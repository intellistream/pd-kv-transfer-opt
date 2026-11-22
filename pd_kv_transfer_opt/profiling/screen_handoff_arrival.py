"""Separate offered arrival rate from cohort size using r43 method traces."""

import argparse
from gzip import open as gzip_open
import json
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.profile_closed_sessions import read_trace
from pd_kv_transfer_opt.profiling.screen_bidirectional_workload import P_RATE
from pd_kv_transfer_opt.profiling.screen_handoff_cohort import is_witness, profile, simulate


def arrival_plan(parent, specs, rate, order):
    source = {row["id"]: row for row in parent["request_sources"]}
    # Build from the pinned manifest's order so every predecessor is listed first.
    by_session = {task: [] for task in order}
    keep = {row["id"] for row in specs}
    for row in source.values():
        if row["id"] in keep:
            by_session[row["session_id"]].append(row)
    arrival = {task: index / rate for task, index in order.items()}
    request_sources = [{**row, "original_offered_s": row["offered_s"], "offered_s": arrival[task]}
                       for task in order for row in by_session[task]]
    return {**{key: parent[key] for key in ("source_repo", "source_revision", "source_workload_sha256", "tokenizer_fingerprint",
                                            "ignore_eos", "temperature", "tool_gap_policy", "max_tokens_policy", "route_policy")},
            "parent_r42_sha256": parent["workload_sha256"], "requests": len(keep), "sessions": len(order),
            "last_session_start_s": max(arrival.values()), "pressure_window_s": [30, 90],
            "arrival_policy": f"Uniform {rate:g} roots/s, preserving all selected causal links and native generation",
            "root_offer_rate_per_s": rate, "request_sources": request_sources}


def screen(samples, workload):
    rows = workload["request_sources"]
    offers = {row["id"]: row["offered_s"] for row in rows}
    results = {}
    for case, (actual, _, background) in samples.items():
        witnesses = {key for key in offers if is_witness(actual[key])}
        selected_cold_input = sum(actual[key]["P_computed"] for key in offers if actual[key]["predecessor"] is None)
        estimates = {}
        for cap in (100, 200):
            output = simulate(actual, rows, witnesses, offers, background, cap, True)
            peaks = output.pop("offered_peaks")
            output.update(P_aggregate_peak_Gbps={window: point["Gbps"] for window, point in peaks["aggregate"].items()},
                          hottest_P_rail_peak_Gbps={window: max(rail[window]["Gbps"] for rail in peaks["each_logical_source_rail"])
                                                   for window in peaks["aggregate"]})
            estimates[str(cap)] = output
        results[case] = {"root_computed_tokens": selected_cold_input, "cold_P_service_floor_s_at_44k": selected_cold_input / P_RATE,
                         "witness_requests": len(witnesses), "cases": estimates}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method_archive", type=Path)
    parser.add_argument("handoff_archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.method_archive / "manifest.json")
    trace = args.method_archive / "r43-method-events.jsonl.gz"
    handoff = read_json(args.handoff_archive / "summary.json")
    base = args.handoff_archive / "workload.json"
    ledger = args.handoff_archive / "prefix-ledger.json.gz"
    rank_map = args.handoff_archive / "decode-worker-ranks.json"
    if (sha256(trace) != manifest["trace_sha256"] or sha256(base) != handoff["candidate_plan_sha256"]
            or sha256(ledger) != handoff["prefix_ledger_sha256"] or sha256(rank_map) != handoff["rank_map_sha256"]):
        raise ValueError("Source trace or candidate fingerprint changed")
    with gzip_open(ledger, "rt", encoding="utf-8") as file:
        prefix = json.load(file)
    parent = manifest["workload"]
    specs = {row["id"]: row for row in parent["request_sources"]}
    method = read_json(args.method_archive / "summary.json")["cases"]
    ranks = read_json(rank_map)
    samples = {case: profile(arm, method[case], specs, prefix, ranks) for case, arm in read_trace(trace).items()}
    frozen = read_json(base)
    fixed_order = {row["session_id"]: i for i, row in enumerate(sorted(
        (r for r in frozen["request_sources"] if r["predecessor"] is None), key=lambda r: r["offered_s"]))}
    extra = [row["session_id"] for row in specs.values() if row["predecessor"] is None and row["session_id"] not in fixed_order]
    extended_order = {task: i for i, task in enumerate([*fixed_order, *extra])}
    plans = {f"fixed_81_rate_{rate}": arrival_plan(parent, frozen["request_sources"], rate, fixed_order)
             for rate in (8, 16, 32)}
    for rate in (8, 1.6):
        plans[f"expanded_254_rate_{rate:g}"] = arrival_plan(parent, specs.values(), rate, extended_order)
    results = {name: {"requests": plan["requests"], "sessions": plan["sessions"],
                      "last_root_s": plan["last_session_start_s"], "historical_samples": screen(samples, plan)}
               for name, plan in plans.items()}
    write_json(args.output / "workload-fast.json", plans["fixed_81_rate_16"])
    write_json(args.output / "workload-extended.json", plans["expanded_254_rate_8"])
    write_json(args.output / "summary.json", {
        "method_trace_sha256": manifest["trace_sha256"], "r46_plan_sha256": sha256(base), "plans": results,
        "scope": "Changing root times and causal releases only. Stage costs, history-specific hit/miss outcomes, routes and "
                 "generation are frozen. No model feedback from changing P queue capacity or route/cache choices."})


if __name__ == "__main__":
    main()
