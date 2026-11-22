"""Replay native 2P2D egress caps on a single reference trace using only CPU."""

from __future__ import annotations

import argparse
from gzip import open as gzip_open
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_multiwave import aligned_tx_intervals
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS
from pd_kv_transfer_opt.simulator.decode_supply import DECODE_ROLES, decode_samples, fit_decode, native_arms, simulate_decode


PORTS = 8
LOW_CAP, HIGH_CAP = 100, 200


def load_native_archive(path):
    manifest = read_json(path / "manifest.json")
    settings = manifest["settings"]
    workload = manifest["workload"]
    engines = settings["engines"]
    if (settings["D_egress_ports"] != PORTS or settings["D_pool_GiB"] != 128
            or not all(engine["enable_prefix_caching"] for engine in engines.values())
            or not engines["D"]["recompute_scheduler_enable"] or workload["ignore_eos"]
            or workload["route_policy"] != "upstream native without route override"):
        raise ValueError(f"Different native baseline in {path.name}")
    digest, arms = native_arms(path)
    for case, (data, _, _) in arms.items():
        cap = int(case.rsplit("-cap", 1)[1])
        for role, stream, expected in (("D", "shaping.json", cap * 1000 // PORTS),
                                      ("P", "P-shaping.json", 200_000)):
            snapshot, = data[stream]
            readback = snapshot["readback"]
            if set(readback) != {str(i) for i in range(PORTS)} or any(
                    row["configured_Mbps"] != expected or row["current_Mbps"] != expected
                    for row in readback.values()):
                raise ValueError(f"Wrong {role} port settings in {path.name}/{case}")
        if any(event.get("fixed_routes") for event in data["proxy"]):
            raise ValueError(f"Non-native routing in {path.name}/{case}")
        initialization, = data["cache-initialization.json"]
        if any(initialization["after"][key] != 0 for key in ("master_allocated_bytes", "master_key_count")):
            raise ValueError(f"Nonempty initial Store in {path.name}/{case}")
    return manifest, digest, arms


def egress_epochs(arm):
    data, requests, origin = arm
    start = min(row["P_start_s"] for row in requests.values())
    finish = max(row["P_end_s"] for row in requests.values())
    intervals = tuple((max(left, start), min(right, finish), rates)
                      for left, right, _, rates in aligned_tx_intervals(data["telemetry"], round(origin * NS))
                      if max(left, start) < min(right, finish))
    if not intervals or abs(sum(end - begin for begin, end, _ in intervals) - (finish - start)) > .001:
        raise ValueError("D port samples do not cover the reference P HTTP span")
    return intervals


def replay_egress(intervals, cap_gbps, pressure_length=1.0, *, use_short_bursts=False):
    """Preserve each interval's per-port bytes and dilate its common service clock."""
    if not isfinite(cap_gbps) or cap_gbps <= 0 or not isfinite(pressure_length) or pressure_length < 1:
        raise ValueError("Use a positive cap and a pressure-window length multiplier of at least one")
    limit = cap_gbps / PORTS
    dilation, byte_counts = [], [0.0] * PORTS
    pressured_reference_s = near_cap_s = capacity_wait_s = 0.0
    for begin, end, rates in intervals:
        if len(rates) != PORTS or any(not isfinite(r) or r < 0 for r in rates) or end <= begin:
            raise ValueError("Invalid aligned D port rates")
        width = end - begin
        peak = max(rates)
        pressured = peak > LOW_CAP / PORTS
        length = pressure_length if pressured else 1.0
        if pressured:
            pressured_reference_s += width
        offered_span = width * length
        busy_fraction = min(1.0, peak / (HIGH_CAP / PORTS)) if use_short_bursts else 1.0
        service_fraction = max(busy_fraction, peak / limit)
        extra_fraction = service_fraction - busy_fraction
        capacity_wait_s += offered_span * extra_fraction
        if peak >= .85 * limit * service_fraction:
            near_cap_s += offered_span * service_fraction
        for port, rate in enumerate(rates):
            offered_bytes = rate * offered_span * 1e9 / 8
            service_rate = rate / service_fraction if service_fraction else 0.0
            assert service_rate <= limit * (1 + 1e-12)
            assert isclose(offered_bytes, service_rate * offered_span * service_fraction * 1e9 / 8, rel_tol=1e-12)
            byte_counts[port] += offered_bytes
        dilation.append((begin, end, length * (1 + extra_fraction) - 1))
    return dilation, {"D_port_TX_bytes": byte_counts, "D_TX_GB_in_P_span": sum(byte_counts) / 1e9,
                      "reference_pressure_s": pressured_reference_s, "D_any_port_near_cap_s": near_cap_s,
                      "capacity_wait_on_shared_egress_clock_s": capacity_wait_s}


def drained_measurement(requests):
    finish = max(row["end_s"] for row in requests.values())
    duration = finish - min(row["offered_s"] for row in requests.values())
    tokens = sum(row["output_tokens"] for row in requests.values())
    return {"output_tokens": tokens, "duration_s": duration, "throughput_tokens_s": tokens / duration,
            "last_P_reply_s": max(row["P_end_s"] for row in requests.values()),
            "all_natural_EOS": all(row["finish_reason"] == "stop" for row in requests.values())}


def replay_prefill_supply(requests, service, exposed, slots, *, include_requests=False):
    prefill = {key: row["phases_s"]["P_HTTP"] + exposed[key] for key, row in requests.items()}
    sim = simulate_decode(requests, service, prefill_http_s=prefill, slots=slots, include_requests=True)
    events = sim.pop("per_request")
    empty_with_P = {}
    for role in DECODE_ROLES:
        rows = [events[key] for key, request in requests.items() if request["D"] == role]
        prefill_inflight = [(row["P_start_s"], row["P_end_s"]) for row in rows]
        generating = [(row["first_s"], row["end_s"]) for row in rows]
        covered = interval_union((max(a, c), min(b, d)) for a, b in prefill_inflight
                                 for c, d in generating if max(a, c) < min(b, d))
        empty_with_P[role] = max(0.0, interval_union(prefill_inflight) - covered)
    last_P = max(row["P_end_s"] for row in events.values())
    result = {**sim, "duration_s": sim["finish_s"] - min(row["offered_s"] for row in requests.values()),
              "last_P_reply_s": last_P, "drain_after_last_P_reply_s": sim["finish_s"] - last_P,
              "P_HTTP_added_s_p50_p95_max": [percentile(exposed.values(), .5), percentile(exposed.values(), .95),
                                              max(exposed.values())],
              "D_not_generating_with_P_inflight_s": empty_with_P}
    if include_requests:
        result["per_request"] = events
    return result


def run_bandwidth(requests, service, intervals, cap, slots, pressure_length=1.0, *,
                  use_short_bursts=False, include_requests=False):
    dilation, transport = replay_egress(intervals, cap, pressure_length, use_short_bursts=use_short_bursts)
    exposed = {key: sum(max(0.0, min(row["P_end_s"], end) - max(row["P_start_s"], start)) * stretch
                        for start, end, stretch in dilation if stretch)
               for key, row in requests.items()}
    return {**replay_prefill_supply(requests, service, exposed, slots, include_requests=include_requests), **transport}


def compare_pair(low_arm, high_arm, service, slots, pressure_lengths, *, include_requests=False):
    reference = high_arm[1]
    observed_low = low_arm[1]
    intervals = egress_epochs(high_arm)

    def prediction(length, short_bursts=False):
        low = run_bandwidth(reference, service, intervals, LOW_CAP, slots, length,
                            use_short_bursts=short_bursts, include_requests=include_requests)
        high = run_bandwidth(reference, service, intervals, HIGH_CAP, slots, length,
                             use_short_bursts=short_bursts, include_requests=include_requests)
        assert low["D_port_TX_bytes"] == high["D_port_TX_bytes"]
        return {"pressure_window_length_multiple": length, "100": low, "200": high,
                "low_throughput_loss_fraction": 1 - high["duration_s"] / low["duration_s"],
                "finish_gap_s": low["duration_s"] - high["duration_s"]}

    base = prediction(1.0)
    burst = prediction(1.0, short_bursts=True)

    measured = {cap: drained_measurement(requests) for cap, requests in (("100", observed_low), ("200", reference))}
    for cap, arm in (("100", low_arm), ("200", high_arm)):
        measured[cap]["D_TX_GB_in_P_span"] = sum((end - start) * sum(rates) / 8
                                                  for start, end, rates in egress_epochs(arm))
    same_P = sum(row["P"] == observed_low[key]["P"] for key, row in reference.items())
    same_D = sum(row["D"] == observed_low[key]["D"] for key, row in reference.items())
    low_tx, high_tx = (measured[cap]["D_TX_GB_in_P_span"] for cap in ("100", "200"))
    comparable = (same_P == len(reference) and abs(low_tx - high_tx) <= .05 * max(low_tx, high_tx)
                  and all(row["all_natural_EOS"] for row in measured.values()))
    measured_gap = measured["100"]["duration_s"] - measured["200"]["duration_s"]
    report = {"requests": len(reference), "same_observed_P": same_P, "same_observed_D": same_D,
              "comparable_P_routes_wire_bytes_and_EOS": comparable, "hardware": measured,
              "hardware_low_throughput_loss_fraction": 1 - measured["100"]["throughput_tokens_s"] /
              measured["200"]["throughput_tokens_s"], "hardware_finish_gap_s": measured_gap,
              "prediction": base, "short_burst_sensitivity": burst,
              "longer_pressure_windows": [prediction(length) for length in pressure_lengths if length > 1.0],
              "predicted_finish_gap_minus_hardware_gap_s": base["finish_gap_s"] - measured_gap}
    if comparable:
        # Compare the predicted shift to an oracle using measured P timings only after prediction.
        fixed = simulate_decode(reference, service, slots=slots,
                                prefill_http_s={key: row["phases_s"]["P_HTTP"] for key, row in observed_low.items()})
        anchored = simulate_decode(reference, service, slots=slots)
        conditional_gap = fixed["finish_s"] - anchored["finish_s"]
        report["measured_P_only_fixed_reference_output_and_D_finish_gap_s"] = conditional_gap
        report["predicted_finish_gap_minus_measured_P_only_gap_s"] = base["finish_gap_s"] - conditional_gap
    return report


def check_get_methods(path, baseline_commit):
    manifest, _, arms = load_native_archive(path)
    summary = path / "native-method-summary.json"
    if (manifest["baseline_commit"] != baseline_commit or not manifest["independent_method_pass"]
            or sha256(summary) != manifest["artifacts"][summary.name]["sha256"]):
        raise ValueError("An independently archived and verified method pass is required")
    report = read_json(summary)
    batches = report["matched_four_rank_GET_batches"]
    rows = []
    for batch in batches:
        low, high = (batch[f"{cap}_Gbps"]["slowest_GET_sum_s"] for cap in (LOW_CAP, HIGH_CAP))
        prediction = high * HIGH_CAP / LOW_CAP
        rows.append({"P": batch["P"], "request_ids": batch["request_ids"], "bytes": batch["bytes"],
                     "100_observed_GET_s": low, "200_observed_GET_s": high, "100_predicted_GET_s": prediction})
    reference_case = f"b1-cap{HIGH_CAP}"
    data, _, _ = arms[reference_case]
    origin, = (row["anchor"]["unix_ns"] for row in data["client"] if row["type"] == "client_origin")
    trace = path / "native-method-events.jsonl.gz"
    if sha256(trace) != report["method_events_sha256"]:
        raise ValueError("The archived method events do not match the summary")
    get_windows = []
    with gzip_open(trace, "rt", encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            event = row["data"]
            if (row["case"] == reference_case and event["type"] == "store_end" and event["instance"].startswith("P")
                    and event["operation"] == "batch_get_into_multi_buffers"):
                there = event["unix_ns"] - origin
                get_windows.append(((there - event["elapsed_ns"]) / NS, there / NS))
    concurrent, seconds = 0.0, 0.0
    for begin, end, rates in egress_epochs(arms[reference_case]):
        if max(rates) > LOW_CAP / PORTS:
            seconds += end - begin
            concurrent += interval_union((max(begin, a), min(end, b)) for a, b in get_windows
                                         if max(begin, a) < min(end, b))
    return {"summary_sha256": sha256(summary), "matched_batches": len(rows),
            "lower_cap_GET_slower_batches": sum(row["100_observed_GET_s"] > row["200_observed_GET_s"] for row in rows),
            "100_prediction_MAE_s": mean(abs(row["100_predicted_GET_s"] - row["100_observed_GET_s"]) for row in rows),
            "matched_batch_results": rows, "reference_any_P_synchronous_GET_active_s": interval_union(get_windows),
            "reference_time_port_above_12_5_Gbps_s": seconds,
            "reference_GET_overlap_with_port_above_12_5_Gbps_s": concurrent}


def analyze(calibration, holdouts, method_archives, pressure_lengths):
    fits, training_digests = [], {}
    deployment = None

    def validated(path):
        nonlocal deployment
        manifest, digest, arms = load_native_archive(path)
        settings = {key: value for key, value in manifest["settings"].items() if key != "serving_observer"}
        identity = (manifest["baseline_commit"], manifest["verified_published_tree"], settings,
                    manifest["workload"]["source_revision"], manifest["workload"]["source_repo"], manifest["software"])
        if deployment is None:
            deployment = identity
        if deployment != identity or manifest["independent_method_pass"]:
            raise ValueError(f"Expected comparable formal timing in {path.name}")
        return digest, arms, manifest["settings"]["engines"]["D"]["max_num_seqs"]

    for path in calibration:
        digest, arms, _ = validated(path)
        if digest in training_digests.values():
            raise ValueError("Duplicate decode calibration trace")
        training_digests[path.name] = digest
        for arm in arms.values():
            fits.extend(decode_samples(*arm))
    decode = fit_decode(fits)
    reports = {}
    for path in holdouts:
        digest, arms, slots = validated(path)
        if digest in training_digests.values():
            raise ValueError("A held-out trace must be distinct from decode calibration")
        orders = sorted({name.split("-cap", 1)[0] for name in arms})
        reports[path.name] = {"trace_sha256": digest,
                              "orders": {order: compare_pair(arms[f"{order}-cap{LOW_CAP}"], arms[f"{order}-cap{HIGH_CAP}"],
                                                             decode, slots, pressure_lengths) for order in orders}}
    return {"model": "native_2p2d_trace_conditioned_egress_replay/v1", "baseline_commit": deployment[0],
            "source_revision": deployment[3], "decode_training_trace_sha256": training_digests,
            "decode_service_fit": decode,
            "method_checks": {path.name: check_get_methods(path, deployment[0]) for path in method_archives},
            "holdouts": reports,
            "scope": ("Use only the 200 Gbps arm for simulated input timing, offered work, P/D roles, natural output and MAC service. "
                      "Compare to the 100 Gbps arm after prediction. Per-port MAC service clocks are shared within each epoch. "
                      "Expose only epoch dilation inside each reference P HTTP interval and re-gate successors on simulated decode. "
                      "Short-burst sensitivity keeps bytes, concentrating each coarse sample at the reference cap. "
                      "Virtual longer windows repeat service time and bytes in epochs with any reference port above 12.5 Gbps at unchanged offered rates. "
                      "The replay conditions on reference routing, cache/traffic and output. It does not generate new APC hits, Store residency, "
                      "new packet demand or model tokens when workload or configuration changes.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, action="append", required=True)
    parser.add_argument("--holdout", type=Path, action="append", required=True)
    parser.add_argument("--methods", type=Path, action="append", default=[])
    parser.add_argument("--pressure-length", type=float, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lengths = sorted(set([1.0, *args.pressure_length]))
    if any(not isfinite(value) or value < 1.0 for value in lengths):
        parser.error("Each pressure length must be finite and at least one")
    write_json(args.output, analyze(args.calibration, args.holdout, args.methods, lengths))


if __name__ == "__main__":
    main()
