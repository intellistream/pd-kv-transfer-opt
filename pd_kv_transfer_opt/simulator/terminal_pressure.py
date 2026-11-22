"""Screen measured Terminal-Bench workloads and replay one without scaling traffic."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from gzip import open as gzip_open
from itertools import accumulate
import json
from pathlib import Path

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS
from pd_kv_transfer_opt.simulator.decode_supply import decode_samples, fit_decode
from pd_kv_transfer_opt.simulator.rdma_replay import (HIGH_CAP, LOW_CAP, PORTS, compare_pair, drained_measurement,
                                                     egress_epochs, load_native_archive, replay_prefill_supply)


CASES = ["b1-cap100", "b1-cap200", "b2-cap200", "b2-cap100"]
ORDERS = ("b1", "b2")


def peak_window(intervals, window):
    boundaries = [intervals[0][0], *(end for _, end, _ in intervals)]
    rates = [sum(row) for _, _, row in intervals]
    start, stop = boundaries[0], boundaries[-1]
    if stop - start < window:
        return sum((end - begin) * sum(row) for begin, end, row in intervals) / window
    integral = [0.0, *accumulate((b - a) * rate for (a, b, _), rate in zip(intervals, rates, strict=True))]

    def until(at):
        index = min(len(rates), bisect_right(boundaries, at) - 1)
        return integral[index] + (at - boundaries[index]) * rates[index] if index < len(rates) else integral[-1]

    # The integral in a moving fixed-width window is linear between these breakpoints.
    starts = {t for edge in boundaries for t in (edge, edge - window) if start <= t <= stop - window}
    return max((until(t + window) - until(t)) / window for t in starts)


def pressure(arm, window, cap=HIGH_CAP):
    intervals = egress_epochs(arm)
    runs = []
    by_port = [0.0] * PORTS
    before_successors = 0.0
    any_near_cap, aggregate_near_cap = 0.0, 0.0
    first_successor = min((row["offered_s"] for row in arm[1].values() if row["predecessor"] is not None),
                          default=None)
    for start, end, rates in intervals:
        for port, rate in enumerate(rates):
            by_port[port] += (end - start) * rate / 8
        aggregate = sum(rates)
        if first_successor is not None:
            before_successors += max(0.0, min(end, first_successor) - start) * aggregate / 8
        if aggregate >= .85 * cap:
            aggregate_near_cap += end - start
        if max(rates) >= .85 * cap / PORTS:
            any_near_cap += end - start
        if aggregate < LOW_CAP:
            continue
        if runs and start <= runs[-1][1] + 1e-9:
            runs[-1] = runs[-1][0], end
        else:
            runs.append((start, end))
    return {"peak_window_average_Gbps": peak_window(intervals, window),
            "D_TX_GB_in_P_span": sum(by_port), "D_TX_GB_by_port": by_port,
            "D_TX_GB_before_first_planned_successor": before_successors if first_successor is not None else None,
            "aggregate_at_least_100_Gbps_s": interval_union(runs),
            "longest_contiguous_at_least_100_Gbps_s": max((end - start for start, end in runs), default=0.0),
            "aggregate_near_applied_cap_s": aggregate_near_cap, "any_port_near_applied_cap_s": any_near_cap,
            "finish_reasons": dict(Counter(row["finish_reason"] for row in arm[1].values()))}


def compatible(manifest, plan):
    return (manifest.get("baseline_commit") == plan["baseline_commit"]
            and manifest.get("workload", {}).get("source_revision") == plan["source_revision"])


def screen(results, plan):
    candidates = []
    for file in sorted(results.glob(plan["manifest_pattern"])):
        manifest = read_json(file)
        if (not compatible(manifest, plan) or manifest.get("case_order") != CASES
                or manifest.get("independent_method_pass") is not False
                or not str(manifest.get("shaping", "")).startswith("D egress only")):
            continue
        _, digest, arms = load_native_archive(file.parent)
        high = {order: pressure(arms[f"{order}-cap{HIGH_CAP}"], plan["screening_window_s"]) for order in ORDERS}
        workload = manifest["workload"]
        eligibility = (any(row["predecessor"] is not None for row in workload["request_sources"])
                       and all(section["finish_reasons"] == {"stop": workload["requests"]} for section in high.values()))
        candidates.append({"archive": file.parent.name, "trace_sha256": digest, "requests": workload["requests"],
                           "session_windows": workload["sessions"], "high_references": high,
                           "natural_EOS_with_predecessors": eligibility,
                           "conservative_peak_window_Gbps": min(row["peak_window_average_Gbps"] for row in high.values())})
    eligible = [row for row in candidates if row["natural_EOS_with_predecessors"]]
    selected = max(eligible, key=lambda row: row["conservative_peak_window_Gbps"])
    return selected, candidates


def supply(requests, timing, since):
    end = since + 10.0
    successors = [timing[key] for key, row in requests.items() if row["predecessor"] is not None]
    return {"window_s": [since, end],
            "successor_P_delivered_by_window_end": sum(row["P_end_s"] <= end for row in successors),
            "successor_D_first_token_by_window_end": sum(row["first_s"] <= end for row in successors),
            "last_successor_P_reply_s": max(row["P_end_s"] for row in successors)}


def planned_recovery_start(requests):
    return min(row["offered_s"] for row in requests.values() if row["predecessor"] is not None)


def historical_r28(archive):
    path = archive / "summary.json"
    summary = read_json(path)
    return {"archive": archive.name, "summary_sha256": sha256(path), "orders": {
        order: {"100": summary["arms"][f"{order}-cap100"]["total"],
                "200": summary["arms"][f"{order}-cap200"]["total"], **summary["pairs"][order]}
        for order in ORDERS}}


def replay_selected(arms, service, slots, history, window):
    pairs = {}
    for order in ORDERS:
        low, high = (arms[f"{order}-cap{cap}"] for cap in (LOW_CAP, HIGH_CAP))
        since = planned_recovery_start(high[1])
        pair = compare_pair(low, high, service, slots, [1.0], include_requests=True)
        for key, section in (("100", low), ("200", high)):
            pair["hardware"][key]["supply"] = supply(section[1], section[1], since)
            pair["hardware"][key]["pressure"] = pressure(section, window, int(key))
            for mode in ("prediction", "short_burst_sensitivity"):
                modeled = pair[mode][key]
                modeled["supply"] = supply(high[1], modeled.pop("per_request"), since)
        target = history["orders"][order]["loss_100_relative_to_200"]
        faster = pair["prediction"]["200"]["duration_s"]
        pair["seconds_needed_at_same_output_for_r28_loss"] = faster * target / (1 - target)
        pairs[order] = pair
    return pairs


def compare_arrivals(path, workload, window):
    manifest, digest, arms = load_native_archive(path)

    def inputs(plan):
        return {row["id"]: {k: v for k, v in row.items() if k != "offered_s"} for row in plan["request_sources"]}

    if (manifest["workload"]["parent_requests_sha256"] != workload["parent_requests_sha256"]
            or inputs(manifest["workload"]) != inputs(workload)):
        return None
    return {"archive": path.name, "trace_sha256": digest, "arms": {
        case: {"first_planned_successor_s": planned_recovery_start(arm[1]),
               **drained_measurement(arm[1]), "pressure": pressure(arm, window, int(case.rsplit("-cap", 1)[1]))}
        for case, arm in arms.items()}}


def overlap(start, end, intervals):
    return interval_union((max(start, a), min(end, b)) for a, b in intervals if max(start, a) < min(end, b))


def profiled_recovery(path, workload, service, slots):
    manifest, digest, arms = load_native_archive(path)
    if (not manifest["independent_method_pass"]
            or manifest["workload"]["request_sources"] != workload["request_sources"]):
        raise ValueError("The method trace does not use the selected frozen workload")
    events_path = path / "native-method-events.jsonl.gz"
    evidence_path = path / "native-method-summary.json"
    for file in (events_path, evidence_path):
        if sha256(file) != manifest["artifacts"][file.name]["sha256"]:
            raise ValueError(f"Method trace hash mismatch for {file.name}")
    evidence = read_json(evidence_path)
    high_key = f"b1-cap{HIGH_CAP}"
    data, reference, _ = arms[high_key]
    origin, = (row["anchor"]["unix_ns"] for row in data["client"] if row["type"] == "client_origin")
    windows = defaultdict(list)
    with gzip_open(events_path, "rt", encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            event = row["data"]
            if (row["case"] == high_key and event["type"] == "store_end" and event["instance"].startswith("P")
                    and event["operation"] == "batch_get_into_multi_buffers" and event["negative_returns"] == 0):
                end = (event["unix_ns"] - origin) / NS
                windows[event["instance"]].append((end - event["elapsed_ns"] / NS, end))
    all_windows = [window for group in windows.values() for window in group]
    if not all_windows:
        raise ValueError("The independent method pass has no synchronous GET calls")
    intervals = egress_epochs(arms[high_key])
    aligned = [(a, b, rates, overlap(a, b, all_windows)) for a, b, rates in intervals]
    since = planned_recovery_start(reference)
    result = {"archive": path.name, "event_trace_sha256": digest, "method_events_sha256": sha256(events_path),
              "matches_four_rank": evidence["matched_four_rank_GET_batches"], "measured": {}}
    for cap in (LOW_CAP, HIGH_CAP):
        case = f"b1-cap{cap}"
        observed = arms[case][1]
        result["measured"][str(cap)] = {**drained_measurement(observed),
                                        "synchronous_GET_wall_union_s": evidence["cases"][case]["P_GET_active_union_s"],
                                        "GET_descriptor_GB": evidence["cases"][case]["P_GET_descriptor_bytes"] / 1e9,
                                        "supply": supply(observed, observed, since)}

    for name, use_local in (("same_P_GET_exposure", True), ("shared_GET_exposure", False)):
        result[name] = {}
        for cap in (LOW_CAP, HIGH_CAP):
            delay_per_epoch = []
            for left, right, rates, active_s in aligned:
                if active_s == 0:
                    continue
                peak = max(rates)
                width = right - left
                active_service = min(width, max(active_s, width * peak / (HIGH_CAP / PORTS)))
                delay = max(0.0, width * peak / (cap / PORTS) - active_service)
                if delay:
                    delay_per_epoch.append((left, right, delay / active_s))
            exposed = {key: sum(overlap(max(left, row["P_start_s"]), min(right, row["P_end_s"]),
                                        windows[row["P"]] if use_local else all_windows) * ratio
                                for left, right, ratio in delay_per_epoch)
                       for key, row in reference.items()}
            predicted = replay_prefill_supply(reference, service, exposed, slots, include_requests=True)
            predicted["supply"] = supply(reference, predicted.pop("per_request"), since)
            result[name][str(cap)] = predicted
    return result


def run(plan, results, output):
    selected, candidates = screen(results, plan)
    archive = results / selected["archive"]
    manifest, digest, arms = load_native_archive(archive)
    workload = manifest["workload"]
    settings = manifest["settings"]
    print(f"Screened {len(candidates)} workloads and selected {archive.name}", flush=True)
    calibration, digests = [], {}
    for name in plan["decode_calibration"]:
        other, cost_digest, measured = load_native_archive(results / name)
        if (cost_digest == digest or not compatible(other, plan) or other["settings"] != settings):
            raise ValueError("Calibration conditions must match and leave the selected trace out")
        digests[name] = cost_digest
        for arm in measured.values():
            calibration.extend(decode_samples(*arm))
    service = fit_decode(calibration)
    slots = settings["engines"]["D"]["max_num_seqs"]
    history = historical_r28(results / plan["historical_comparison"])
    window = plan["screening_window_s"]
    pairs = replay_selected(arms, service, slots, history, window)
    arrivals = compare_arrivals(results / plan["same_inputs_arrival_control"], workload, window)
    observed = archive.with_name(archive.name + "-methods")
    methods = profiled_recovery(observed, workload, service, slots) if observed.is_dir() else None
    prompt_widths = [row["prompt_tokens"] for row in workload["request_sources"]]
    workload_file = output.parent / "selected-workload.json"
    write_json(workload_file, {"source_archive": archive.name, "source_manifest_sha256": sha256(archive / "manifest.json"),
                              "baseline_commit": manifest["baseline_commit"], "settings": settings,
                              "source_ledger_sha256": workload["selection"].get("source_ledger_sha256"),
                              "workload": {key: workload[key] for key in ("source_repo", "source_revision", "source_workload_sha256",
                                           "parent_requests_sha256", "requests", "sessions", "ignore_eos", "temperature",
                                           "route_policy", "request_sources")}})
    write_json(output, {"plan": plan, "selection": selected, "screened": candidates,
                        "selected_workload_sha256": sha256(workload_file), "decode_calibration_traces": digests,
                        "prompt_tokens_min_p50_max": [min(prompt_widths), percentile(prompt_widths, .5), max(prompt_widths)],
                        "decode_service_fit": service, "selected_bandwidth_pairs": pairs,
                        "independent_method_replay": methods, "same_inputs_different_arrivals": arrivals,
                        "historical_r28": history,
                        "scope": ("Selection uses only the two measured 200 Gbps references. Exact archived request bodies are "
                                  "identified by hashes. Every pair reuses reference native placements, observed cache/egress, "
                                  "natural EOS output, offers and dependencies. There is no traffic or arrival multiplier. The "
                                  "method replay uses real GET activity as a pacing constraint and exposes delay on each request's "
                                  "local P or on either P. It holds the 200 Gbps trace's group shape and does not predict changes "
                                  "in scheduling, placement or caching caused by a different bandwidth.")})
    print(f"Completed local comparison for {workload['sessions']} sessions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(read_json(args.plan), args.results, args.output)


if __name__ == "__main__":
    main()
