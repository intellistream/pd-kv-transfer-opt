"""Connect peer send pressure, D-ready work, and drained throughput."""

import argparse
from collections import defaultdict
from itertools import pairwise
from math import isclose
from pathlib import Path
from statistics import mean

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_native_prefill_path import GET, clipped_union, report
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution, interval_union
from pd_kv_transfer_opt.profiling.analyze_packed_natural_replay import compare_prefixes, decode_work_and_cadence
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, compare_arms, measure_ports, read_trace, requests_for


def supply_under_pressure(telemetry, profiles, detail, cap_Gbps, origin):
    """Measure readiness and port pressure with D-host clocks only."""
    last = max(row["end_s"] for row in profiles.values())
    ports, aggregate_hot, any_hot, _ = measure_ports(telemetry, round(origin * NS), last, cap_Gbps)
    last_P = max(row["P_end_s"] for row in profiles.values())
    tail_GB = sum((end - max(start, last_P)) * rate / 8 for start, end, rate, _ in ports if end > last_P)
    per_role = {}
    for role in ("D0", "D1"):
        events = defaultdict(lambda: [0, 0])
        requests = {key: row for key, row in profiles.items() if row["D"] == role}
        first = min(detail[key]["D_KV_ready_s"] for key in requests)
        for key, row in requests.items():
            events[row["P_start_s"]][0] += 1
            events[row["P_end_s"]][0] -= 1
            events[detail[key]["D_KV_ready_s"]][1] += 1
            events[row["end_s"]][1] -= 1
        waiting_P, ready = 0, 0
        low = {0: [], 1: [], 3: [], 7: []}
        for start, end in pairwise(sorted(events)):
            adds_P, adds_D = events[start]
            waiting_P += adds_P
            ready += adds_D
            if start >= first and waiting_P:
                for limit, spans in low.items():
                    if ready <= limit:
                        spans.append((start, end))
        per_role[role] = {
            str(limit): {"P_inflight_s": sum(right - left for left, right in spans),
                         "with_aggregate_hot_s": sum(clipped_union(aggregate_hot, span) for span in spans),
                         "with_any_port_hot_s": sum(clipped_union(any_hot, span) for span in spans)}
            for limit, spans in low.items()
        }
    return {"D_TX_after_last_P_reply_GB": tail_GB,
            "after_first_D_KV_ready_with_P_inflight_and_at_most_this_many_ready": per_role}


def add_reported_throughput(row, output_tokens):
    return {**row, "observed_throughput_tokens_s": output_tokens / row["original_finish_s"],
            "modeled_throughput_tokens_s": output_tokens / row["counterfactual_finish_s"]}


def same_byte_gets(first, second):
    pairs = [(a, second[key]) for key, a in first.items()
             if a["P_direct_GET_bytes"] > 0
             and all(a[field] == second[key][field] for field in
                     ("P", "D", "P_direct_GET_calls", "P_direct_GET_bytes"))]
    return {"requests": len(pairs), "paired_payload_bytes": sum(a["P_direct_GET_bytes"] for a, _ in pairs),
            "Store_s_a_minus_b": distribution(a["P_direct_slowest_rank_Store_s"] - b["P_direct_slowest_rank_Store_s"]
                                              for a, b in pairs),
            "P_HTTP_s_a_minus_b": distribution(a["route_P_HTTP_s"] - b["route_P_HTTP_s"] for a, b in pairs),
            "D_ready_after_proxy_s_a_minus_b": distribution(
                (a["D_KV_ready_s"] - a["D_start_s"]) - (b["D_KV_ready_s"] - b["D_start_s"]) for a, b in pairs)}


def analyze(trace, manifest):
    digest = sha256(trace)
    if manifest["files"][trace.name]["sha256"] != digest:
        raise ValueError("The raw-event archive does not match its manifest")
    raw = read_trace(trace)
    if raw.keys() != set(manifest["case_order"]):
        raise ValueError("The trace and manifest describe different replay arms")
    cases, profiles, outputs = {}, {}, {}
    for name in manifest["case_order"]:
        arm = raw[name]
        cap = manifest["cases"][name]["cap_Gbps"]
        origin, = (r["monotonic_s"] for r in arm["client"] if r["type"] == "client_origin")
        profiles[name] = requests_for(arm, manifest["workload"]["request_sources"], origin)
        cp = profiles[name]
        outputs[name] = {r["id"]: r for r in arm["client"] if r["type"] == "request_end"}
        observed = report(arm, manifest, cap)
        base = observed["baseline"]
        observed["Store_GET"]["activity_union_s"] = interval_union(
            (row["monotonic_ns"] - row["elapsed_ns"], row["monotonic_ns"])
            for row in arm["p_native"] if row["type"] == "store_end" and row["operation"] == GET) / NS
        for field in ("conditional_Store_windows", "conditional_P_windows", "conditional_D_windows"):
            observed[field] = {label: add_reported_throughput(row, base["output_tokens"])
                               for label, row in observed[field].items()}
        observed["D_supply_under_port_pressure"] = supply_under_pressure(
            arm["telemetry"], cp, observed["per_request"], cap, origin)
        expected = manifest["cases"][name]["throughput_receipt"]
        if (not isclose(base["throughput_tokens_s"], expected["output_tokens_per_s"], rel_tol=1e-10)
                or base["output_tokens"] != expected["output_tokens"]):
            raise ValueError(f"Wrong output totals or timing in {name}")
        cases[name] = observed

    pairs = {}
    for label, (a, b) in manifest["comparisons"].items():
        if a not in cases or b not in cases:
            continue
        left, right = cases[a]["baseline"], cases[b]["baseline"]
        p = compare_arms(profiles[a], profiles[b], left, right)
        p["cases"] = [a, b]
        p["throughput_tokens_s"] = [left["throughput_tokens_s"], right["throughput_tokens_s"]]
        p["throughput_change_percent"] = 100 * (right["throughput_tokens_s"] / left["throughput_tokens_s"] - 1)
        p["generation_work_vs_cadence_s"] = decode_work_and_cadence(profiles[a], profiles[b])
        p["output_sequence_agreement"] = compare_prefixes(outputs[a], outputs[b])
        p["same_byte_gets"] = same_byte_gets(cases[a]["per_request"], cases[b]["per_request"])
        pairs[label] = p
    balance = {}
    if pairs and pairs.keys() == manifest["comparisons"].keys():
        for cap in (100, 200):
            replicas = [v["baseline"] for key, v in cases.items() if manifest["cases"][key]["cap_Gbps"] == cap]
            rates = [v["throughput_tokens_s"] for v in replicas]
            balance[str(cap)] = {"mean_throughput_tokens_s": mean(rates),
                                 "same_cap_throughput_range_tokens_s": [min(rates), max(rates)],
                                 "mean_elapsed_s": mean(v["duration_s"] for v in replicas)}
    return {"trace_sha256": digest, "cases": cases, "comparisons": pairs, "counterbalanced_caps": balance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(args.trace, read_json(args.manifest)))


if __name__ == "__main__":
    main()
