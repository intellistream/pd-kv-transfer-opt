"""Attribute Terminal-Bench output to measured shared-link pressure."""

import argparse
from bisect import bisect_right
from itertools import pairwise
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_multiwave import aligned_tx_intervals
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, gauge, read_trace, requests_for
from pd_kv_transfer_opt.profiling.screen_mac_demand import align_prefill


def background_rates(rows, origin):
    times = [(row["monotonic_ns"] - origin) / NS for row in rows]
    deltas = [tuple(b - a for a, b in zip(first["received_bytes"], last["received_bytes"], strict=True))
              for first, last in pairwise(rows)]
    spans = []
    for (left, right), counts in zip(pairwise(times), deltas, strict=True):
        if not left < right or len(counts) != 8 or min(counts) < 0:
            raise ValueError("Invalid background receiver counters")
        spans.append((left, right, [8 * count / (right - left) / NS for count in counts]))
    return times, spans


def overlap(left, right, begin, end):
    return max(0, min(right, end) - max(left, begin))


def volumes_by_rail(series, begin, end):
    volumes = [0.0] * 8
    for left, right, rates in series:
        seconds = overlap(left, right, begin, end)
        for rail, rate in enumerate(rates):
            volumes[rail] += seconds * rate / 8
    return volumes


def decode_supply_overlap(samples, profiles, origin, contention):
    result = {}
    for role in ("D0", "D1"):
        points = [((row["monotonic_ns"] - origin) / NS, gauge(row["values"], "num_requests_running"))
                  for sample in samples for row in sample.get("engines", [])
                  if row["role"] == role and row.get("values")]
        first_token_waits = [(row["D_start_s"], row["first_s"]) for row in profiles.values() if row["D"] == role]
        idle_with_wait = []
        for (left, running), (right, _) in pairwise(points):
            if running == 0 and right - left <= 6:
                idle_with_wait.extend((max(left, begin), min(right, end)) for begin, end in first_token_waits
                                      if overlap(left, right, begin, end))
        result[role] = {"without_running_with_pending_first_token_s": interval_union(idle_with_wait),
                        "overlap_with_contention_s": interval_union(
                            (max(left, begin), min(right, end)) for left, right in idle_with_wait
                            for begin, end in contention if overlap(left, right, begin, end))}
    return result


def shared_wire(wire_samples, pressure_samples, origin, begin, end, cap, calibrated_Gbps, sender="P"):
    times, pressure = background_rates(pressure_samples, origin)
    wire = list(aligned_tx_intervals(wire_samples, origin))
    duration = end - begin
    wire_spans = [(left, right, rates) for left, right, _, rates in wire]
    if min(times[-1], wire[-1][1]) < end - 1 or max(times[0], wire[0][0]) > begin + 1:
        raise ValueError("Incomplete shared-link counter coverage")
    mac = volumes_by_rail(wire_spans, begin, end)
    background = volumes_by_rail(pressure, begin, end)
    near_cap = [(max(left, begin), min(right, end)) for left, right, rate, _ in wire
                if rate >= 0.90 * cap and overlap(left, right, begin, end)]
    contention = []
    for left, right, rate, _ in wire:
        if rate < 0.90 * cap or not overlap(left, right, begin, end):
            continue
        index = max(0, bisect_right(times, left) - 1)
        while index < len(pressure) and pressure[index][0] < right:
            start, stop, rates = pressure[index]
            lo, hi = max(left, start, begin), min(right, stop, end)
            if lo < hi and sum(rates) < 0.95 * calibrated_Gbps:
                contention.append((lo, hi))
            index += 1
    return {"window_s": [begin, end], f"{sender}_MAC_GB_by_rail": mac, "background_payload_GB_by_rail": background,
            f"{sender}_MAC_Gbps": 8 * sum(mac) / duration, "background_payload_Gbps": 8 * sum(background) / duration,
            f"{sender}_MAC_minus_background_payload_Gbps": 8 * (sum(mac) - sum(background)) / duration,
            "MAC_above_90pct_cap_s": interval_union(near_cap),
            "MAC_above_90pct_cap_and_background_below_95pct_calibration_s": interval_union(contention)}, contention


def check_release(arm, name):
    state, = arm[name]
    if state["stage"] != "finished" or sorted(state["unregister"].values()) != [0] * 8:
        raise ValueError("Incomplete background buffer release")


def reverse_wire(arm, origin, report, tail, window_start, window_end):
    mono, unix = origin["monotonic_ns"], origin["unix_ns"]
    receivers = [{**sample, "monotonic_ns": mono + sample["unix_ns"] - unix} for sample in arm["P-pressure-bytes"]]
    times, pressure = background_rates(receivers, mono)
    if times[0] > -5 or times[-1] < -1:
        raise ValueError("Missing shaped quiet-link reference for D egress")
    quiet_rate = 8 * sum(volumes_by_rail(pressure, -5, -1)) / 4
    full, contention = shared_wire(arm["telemetry"], receivers, mono, 0, report["duration_s"], report["cap_Gbps"], quiet_rate, "D")
    full["contention_during_tail_P_HTTP_s"] = interval_union(
        (max(left, row["P_start_s"]), min(right, row["P_end_s"])) for left, right in contention for row in tail
        if overlap(left, right, row["P_start_s"], row["P_end_s"]))
    early, _ = shared_wire(arm["telemetry"], receivers, mono, window_start, window_end, report["cap_Gbps"], quiet_rate, "D")
    calibration, = arm["P-pressure-calibration.json"]
    check_release(arm, "P-pressure-state.json")
    return {"pre_shaping_calibration": calibration, "shaped_quiet_payload_Gbps": quiet_rate, "full_run": full,
            f"{window_start:g}_{window_end:g}_s": early}


def analyze(directory):
    manifest = read_json(directory / "manifest.json")
    path = directory / manifest["event_trace"]
    digest = sha256(path)
    summary = read_json(directory / "summary.json")["arms"]
    if digest != manifest["artifacts"][path.name]["sha256"] or "background_condition" not in manifest:
        raise ValueError("Expected an intact shared-link experiment")
    window_start, window_end = manifest["workload"].get("pressure_window_s", (30, 90))
    cases = {}
    for case, arm in read_trace(path).items():
        report = summary[case]
        origin, = [row["anchor"] for row in arm["client"] if row["type"] == "client_origin"]
        calibration, = arm["pressure-calibration.json"]
        calibrated_Gbps = sum(calibration["delivered_Gbps_by_rail"])
        profiles = requests_for(arm, manifest["workload"]["request_sources"], origin["monotonic_ns"] / NS)
        p_mac = align_prefill(arm["P-telemetry"], origin)
        background = arm["pressure-bytes"]
        spans, contention = shared_wire(p_mac, background, origin["monotonic_ns"], 0, report["duration_s"],
                                        report["cap_Gbps"], calibrated_Gbps)
        tail = [profiles[key] for key in report["observed_dependencies"]["tail_chain"]]
        spans["contention_during_tail_D_to_first_s"] = interval_union(
            (max(left, row["D_start_s"]), min(right, row["first_s"])) for left, right in contention for row in tail
            if overlap(left, right, row["D_start_s"], row["first_s"]))
        early, _ = shared_wire(p_mac, background, origin["monotonic_ns"], window_start, window_end, report["cap_Gbps"], calibrated_Gbps)
        check_release(arm, "pressure-state.json")
        cases[case] = {"throughput_token_s": report["throughput_tokens_s"], "output_tokens": report["output_tokens"],
                       "drained_duration_s": report["duration_s"], "calibration": calibration, "full_run": spans,
                       f"{window_start:g}_{window_end:g}_s": early, "D_MAC_TX_GB": report["D_TX_GB"],
                       "D_supply": decode_supply_overlap(arm["telemetry"], profiles, origin["monotonic_ns"], contention)}
        if "D_to_P" in manifest.get("background_directions", ()):
            cases[case]["D_to_P"] = reverse_wire(arm, origin, report, tail, window_start, window_end)
    return {"event_trace_sha256": digest, "arms": cases,
            "scope": "P MAC minus background completed payload includes protocol overhead and other P traffic. "
                     "D-to-first overlap bounds possible transfer waits and includes decode scheduling."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(args.archive))


if __name__ == "__main__":
    main()
