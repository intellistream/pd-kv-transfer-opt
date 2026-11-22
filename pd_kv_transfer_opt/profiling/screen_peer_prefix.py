"""Bound peer-P Attention reuse and replay r19's measured per-rank GET order."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_native_prefill_path import GET, caller_timing
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import (
    ATTENTION_OBJECT_BYTES, classify_layout, read_bundle,
)
from pd_kv_transfer_opt.profiling.analyze_natural_restore_ab import route
from pd_kv_transfer_opt.profiling.audit_restore_worker_costs import get_case_nodes
from pd_kv_transfer_opt.profiling.compare_session_runs import ARCHIVES
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, read_trace, requests_for


PAGE_TOKENS = 1024
PEER_SETUP_S = 0.010
PEER_RATE_SCENARIOS = (7.5, 8.5, 10)


def prefix_pages(spec, name, local, external):
    row = spec[name]
    previous = spec[row["predecessor"]]
    lcp = min(row["source_prompt_lcp_to_previous"], previous["prompt_tokens"])
    return max(0, min(lcp, local + external) // PAGE_TOKENS - local // PAGE_TOKENS)


def layout(row):
    components = classify_layout(row)
    if components is None:
        raise ValueError("Unexpected Attention/GDN descriptor layout")
    return components


def natural_bytes(root, spec, run):
    folder = root / "results" / ARCHIVES[run]
    trace = folder / f"{run}-events.jsonl.gz"
    manifest = read_json(folder / "manifest.json")
    reports = {}
    for case, arm in read_trace(trace).items():
        origin, = (row["monotonic_s"] for row in arm["client"] if row["type"] == "client_origin")
        requests = requests_for(arm, manifest["workload"]["request_sources"], origin)
        events = arm["p_native"]
        _, reads, _ = caller_timing(events)
        lookups = defaultdict(set)
        for event in events:
            if event["type"] == "pool_match_tokens" and event["external_tokens"] > 0:
                lookups[event["request_id"]].add((event["local_tokens"], event["external_tokens"]))
        total = peer = switched = switched_bytes = 0
        for name, by_prefill in reads.items():
            request = requests[name]
            calls = by_prefill[request["P"]]
            attention = sum(layout(row)[0] for row in calls)
            size = sum(row["payload_bytes"] for row in calls)
            total += size
            previous = request["predecessor"]
            if previous is None or requests[previous]["P"] == request["P"]:
                continue
            if len(lookups[name]) != 1:
                raise ValueError(f"Missing unique external-match bounds for {name} in {case}")
            (local, external), = lookups[name]
            peer += min(attention, 4 * ATTENTION_OBJECT_BYTES * prefix_pages(spec, name, local, external))
            switched += 1
            switched_bytes += size
        reports[case] = {"requests_switching_P": switched, "GET_bytes": total,
                         "GET_bytes_after_P_switch": switched_bytes, "peer_upper_bound_bytes": peer}
    return {"trace_sha256": sha256(trace), "cases": reports}


def r19_case(nodes, end_rows, target_bytes, peer_GB_s=None, coverage=1.0, eligible=None):
    savings = defaultdict(float)
    for node in nodes:
        before, after = [], []
        for rank, forward in node["forwards"].items():
            old = forward["forward_enter_ns"] / NS
            delta = 0.0
            for request, read in node["reads"].get(rank, {}).items():
                if eligible is not None and request not in eligible:
                    continue
                borrowed = coverage * target_bytes[read.end]
                if not borrowed:
                    continue
                original = (read.end - read.begin) / NS
                remaining = original * (1 - borrowed / read.size)
                peer = PEER_SETUP_S + borrowed / (peer_GB_s * 1e9) if peer_GB_s else 0.0
                delta += original - max(remaining, peer) if peer_GB_s else original
            before.append(old)
            after.append(old - delta)
        batch_savings = max(before) - max(after)
        for request in node["names"]:
            savings[request] += batch_savings

    end_times = {row["id"]: row["end_s"] - savings[row["id"]] for row in end_rows}
    terminal = max(end_times, key=end_times.get)
    duration = end_times[terminal] - min(row["offered_s"] for row in end_rows)
    output = sum(row["completion_tokens"] for row in end_rows)
    return {"duration_s": duration, "output_tokens": output, "throughput_tokens_s": output / duration,
            "last_request": terminal.split("~", 1)[-1]}


def wave(root, spec):
    directory = root / "results/2026-11-18-r19-natural-ab"
    summary = read_json(directory / "summary.json")
    p_path, d_path = (directory / f"r19-{role}.jsonl.gz" for role in "PD")
    prefill, decode = read_bundle(p_path), read_bundle(d_path)
    cases = get_case_nodes(prefill, decode)
    get_rows = {row["monotonic_ns"]: row for row in prefill["native"]
                if row["type"] == "store_end" and row["operation"] == GET}
    routes = {row["id"]: route(row) for row in decode["proxy"] if row["type"] == "prefill_begin"}
    reports = {}
    for pair in summary["paired_rounds"]:
        cap = pair["D_egress_cap_Gbps"]
        case = f"a-{pair['block']}-cap{cap}"
        actual = pair["baseline"]
        details = actual["per_request"]
        budget = {}
        for node in cases[case]:
            for reads in node["reads"].values():
                for request, read in reads.items():
                    name = request.split("~", 1)[-1]
                    previous = spec[name]["predecessor"]
                    row = details[name]
                    source_prefix = f"{case}~{previous}"
                    pages = prefix_pages(spec, name, row["P_local_prefix_tokens"], row["P_external_prefix_tokens"])
                    attention, _ = layout(get_rows[read.end])
                    budget[read.end] = 0
                    if routes[source_prefix] != routes[request]:
                        budget[read.end] = min(attention, pages * ATTENTION_OBJECT_BYTES)
        ends = [row for row in decode["client"] if row["type"] == "request_end" and row["case"] == case]
        original = r19_case(cases[case], ends, budget, coverage=0)
        if abs(original["duration_s"] - actual["drain_s"]) > 1e-6:
            raise ValueError(f"Replay of the observed duration did not close in {case}")
        terminal = max(ends, key=lambda row: row["end_s"])["id"]
        reports[case] = {"cap_Gbps": cap, "observed": original, "GET_bytes": actual["get_bytes"],
                         "peer_upper_bound_bytes": sum(budget.values()),
                         "own_GET_removed": r19_case(cases[case], ends, budget, eligible={terminal}),
                         "all_candidate_GET_removed": r19_case(cases[case], ends, budget),
                         "assumed_path": {str(rate): r19_case(cases[case], ends, budget, rate)
                                          for rate in PEER_RATE_SCENARIOS},
                         "coverage_at_10_GB_s": {str(coverage): r19_case(cases[case], ends, budget, 10, coverage)
                                                 for coverage in (0.5, 0.75, 0.9)}}
    means = {str(cap): {str(rate): mean(row["assumed_path"][str(rate)]["throughput_tokens_s"]
                                       for row in reports.values() if row["cap_Gbps"] == cap)
                        for rate in PEER_RATE_SCENARIOS} for cap in (100, 200)}
    return {"source_trace_sha256": {path.name: sha256(path) for path in (p_path, d_path)},
            "setup_s": PEER_SETUP_S, "cases": reports, "conditional_average_rate_tokens_s": means}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.repo / "configs/workload-natural-r10.json"
    workload = {row["id"]: row for row in read_json(source)["requests"]}
    write_json(args.output, {"source_workload_sha256": sha256(source),
                             "r19": wave(args.repo, workload),
                             **{run: natural_bytes(args.repo, workload, run) for run in ("r33", "r39")}})


if __name__ == "__main__":
    main()
