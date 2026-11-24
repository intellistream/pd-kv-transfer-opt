"""Rebuild the native 200 Gbps Terminal-Bench completion paths from archived traces."""

import argparse
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from statistics import median

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.compare_session_runs import D_ROLES, DECODE, timeline
from pd_kv_transfer_opt.profiling.profile_closed_sessions import (
    NS, check_observed, gauge, measure_ports, phase_times, read_trace, requests_for,
)


ARCHIVES = {
    "r43": ("2026-11-20-r43-native-decode-apc", "r43-events.jsonl.gz"),
    "r43_methods": ("2026-11-20-r43-native-decode-apc-methods", "r43-method-events.jsonl.gz"),
    "r45": ("2026-11-21-r45-r28-symmetric", "r45-events.jsonl.gz"),
    "r48": ("2026-11-21-r48-terminal-handoff-extended-symmetric", "r48-events.jsonl.gz"),
    "r69": ("2026-11-22-r69-terminal-source-restore-short-guard", "r69-events.jsonl.gz"),
    "r78": ("2026-11-22-r78-terminal-source-restore-stable-tail", "r78-events.jsonl.gz"),
}


def sampled_decode_windows(telemetry, requests, origin):
    """Use client boundaries and stable sampled D width to bound observable windows."""
    points = defaultdict(list)
    for sample in telemetry:
        for engine in sample.get("engines", []):
            role = engine["role"]
            if role in D_ROLES:
                values = engine.get("values", {})
                readings = (gauge(values, field) for field in
                            ("num_requests_running", "num_requests_waiting", "generation_tokens_total"))
                points[role].append((engine["monotonic_ns"] / NS - origin, *readings))
    for role, samples in points.items():
        cohort = {key: row for key, row in requests.items() if row["D"] == role}
        for (a, width, waiting, tokens), (b, next_width, next_wait, next_tokens) in pairwise(samples):
            if (a < 0 or not 0 < b - a <= 6 or width is None or width <= 0
                    or width != next_width or waiting != 0 or next_wait != 0 or tokens is None
                    or next_tokens is None or tokens >= next_tokens):
                continue
            active = [key for key, row in cohort.items() if row["D_start_s"] < b and row["end_s"] > a]
            if len(active) != width or any(cohort[key]["first_s"] > a or cohort[key]["end_s"] < b for key in active):
                continue
            yield width, active, b - a, next_tokens - tokens


def width_band(width):
    return "1" if width == 1 else "2–4" if width <= 4 else "5–8" if width <= 8 else "9–16"


def decode_windows(telemetry, requests, origin):
    per_request = defaultdict(Counter)
    pace = defaultdict(list)
    for width, active, span, tokens in sampled_decode_windows(telemetry, requests, origin):
        tag = width_band(width)
        for key in active:
            per_request[key][tag] += span
        pace[tag].append(1000 * span * width / tokens)
    return per_request, {tag: {"intervals": len(ms), "median_effective_step_ms": median(ms)}
                         for tag, ms in pace.items()}


def analyze_arm(raw, workload):
    origin, = (row["monotonic_s"] for row in raw["client"] if row["type"] == "client_origin")
    requests = requests_for(raw, workload, origin)
    costs = phase_times(requests)
    observed = timeline(requests, costs)
    chain = check_observed(requests)
    path = chain["tail_chain"]
    per_request, pace = decode_windows(raw["telemetry"], requests, origin)
    stop = max(row["end_s"] for row in requests.values())
    _, _, hot, peak = measure_ports(raw["telemetry"], round(origin * NS), stop, 200)

    def reduce(reductions, phase, fraction):
        changed = timeline(requests, {
            key: {**row, phase: row[phase] - fraction * reductions[key]} for key, row in costs.items()
        })
        return {**changed, "gain_pct": 100 * (changed["throughput_tokens_s"] / observed["throughput_tokens_s"] - 1)}

    full = {phase: reduce({key: row[phase] for key, row in costs.items()}, phase, .1)
            for phase in ("P_HTTP", "D_to_first", DECODE)}
    one = {key: per_request[key]["1"] for key in costs}
    low = {key: one[key] + per_request[key]["2–4"] for key in costs}
    return {
        "observed": observed, "requests": len(requests),
        "finish_reasons": dict(Counter(row["finish_reason"] for row in requests.values())),
        "critical_chain": [{"request": key, "P": requests[key]["P"], "D": requests[key]["D"],
                            "output_tokens": requests[key]["output_tokens"], "finish_reason": requests[key]["finish_reason"]}
                           for key in path],
        "critical_chain_s": chain["tail_chain_phase_sum_s"],
        "confirmed_D_width_s_on_critical_chain": {tag: sum(per_request[key][tag] for key in path)
                                                 for tag in ("1", "2–4", "5–8", "9–16")},
        "D_width_sampled_effective_pace": pace,
        "D_MAC_peak_1s_Gbps": peak, "D_at_least_one_hot_port_s": sum(b - a for a, b in hot),
        "phase_residence_minus_10pct": full,
        "only_confirmed_D_width_1_minus_20pct": reduce(one, DECODE, .2),
        "only_confirmed_D_width_1_to_4_minus_20pct": reduce(low, DECODE, .2),
    }


def analyze(results):
    output = {}
    for name, (archive, trace_file) in ARCHIVES.items():
        root = results / archive
        manifest = read_json(root / "manifest.json")
        trace = root / trace_file
        digest = sha256(trace)
        artifacts = manifest.get("artifacts", manifest.get("files", {}))
        if digest != artifacts[trace_file]["sha256"]:
            raise ValueError(f"Archive digest mismatch in {archive}")
        workload = manifest["workload"]["request_sources"]
        cases = {key: analyze_arm(raw, workload) for key, raw in read_trace(trace).items() if key.endswith("cap200")}
        if not cases:
            raise ValueError(f"No 200 Gbps arm in {archive}")
        output[name] = {"archive": archive, "trace_sha256": digest, "cases": cases}
    return {"archives": output,
            "interpretation": "Fixed observed per-request work, placements and phase durations with original session "
                              "dependencies. Width estimates require an unchanged client cohort and agreeing endpoint "
                              "counters. They identify sampled intervals, not exact device batches. Stage reductions "
                              "are sensitivity calculations without rerouting, rebatching or a device intervention."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, analyze(args.results))


if __name__ == "__main__":
    main()
