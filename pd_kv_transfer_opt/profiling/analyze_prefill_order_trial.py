"""Audit native, SJF and history-score arms at one D egress cap."""

import argparse
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.profile_closed_sessions import (
    NS, compare_arms, describe, measure_arms, read_trace, requests_for,
)


PAIRS = (("a1-native", "c1-history"), ("a2-native", "c1-history"),
         ("a1-native", "b1-sjf"), ("a2-native", "b2-sjf"), ("a1-native", "a2-native"))


def audit_order(case, events, traffic):
    mode = case.split("-", 1)[-1]
    roots = {key for key, row in traffic.items() if row["predecessor"] is None}
    starts = {}
    decisions = {}
    for row in events:
        if (row["case"], row["mode"]) != (case, mode):
            raise ValueError(f"The P worker received mismatched arm advice in {case}")
        if row["type"] == "prefill_order_root_started":
            key = row["request_id"]
            if key in starts or key not in roots or traffic[key]["P"] != row["instance"]:
                raise ValueError(f"Invalid cold-root admission audit for {key}")
            starts[key] = {"P": row["instance"], "D": traffic[key]["D"],
                           "wait_to_first_chunk_s": (row["monotonic_ns"] - row["arrival_ns"]) / NS,
                           "P_end_s": traffic[key]["P_end_s"], "first_s": traffic[key]["first_s"]}
        elif row["type"] == "prefill_order_decision":
            if row["instance"] in decisions or sorted(row["before"]) != sorted(row["after"]):
                raise ValueError(f"Expected a single permutation on {row['instance']} in {case}")
            decisions[row["instance"]] = row
    if set(starts) != roots:
        raise ValueError(f"Missing P root dispatch receipts in {case}: {sorted(roots - starts.keys())}")
    confirmed = set(decisions) == (set() if mode == "native" else {"P0", "P1"})
    return {"receipt_complete": confirmed, "mode": mode,
            "instances_reordered": sum(row["before"] != row["after"] for row in decisions.values()),
            "wait_to_first_chunk_s": describe([row["wait_to_first_chunk_s"] for row in starts.values()]),
            "decisions": decisions, "roots": starts}


def analyze(path, manifest):
    spec = manifest["workload"]["request_sources"]
    if manifest["workload"]["ignore_eos"] or len(spec) != 170:
        raise ValueError("Require the full natural EOS workload")
    cap_gbps = manifest.get("D_egress_cap_Gbps", 100)
    if cap_gbps not in (100, 200):
        raise ValueError(f"Unexpected D egress cap: {cap_gbps}")
    port_rate_mbps = cap_gbps * 1000 // 8
    phases, reports = {}, {}
    for case, raw in read_trace(path).items():
        origin, = (row["monotonic_s"] for row in raw["client"] if row["type"] == "client_origin")
        traffic = phases[case] = requests_for(raw, spec, origin)
        if any(row.get("fixed_routes") or row.get("http_connection_reuse_disabled") for row in raw["proxy"]):
            raise ValueError(f"Unexpected route or proxy connection override in {case}")
        configuration, = raw["shaping.json"]
        restored, = raw["restore.json"]
        init, = raw["cache-initialization.json"]
        if (configuration["aggregate_Gbps"] != cap_gbps or len(configuration["readback"]) != 8
                or any(port["configured_Mbps"] != port_rate_mbps or port["current_Mbps"] != port_rate_mbps
                       for port in configuration["readback"].values())
                or not restored["restored"] or init["after"]["master_key_count"] != 0):
            raise ValueError(f"Unverified shaping or Store reset in {case}")
        report = measure_arms(raw, traffic, origin, cap_gbps, {"P": 16, "D": 16})
        report["P_order"] = audit_order(case, raw["prefill_order"], traffic)
        by_session = {}
        for row in traffic.values():
            session = row["session_id"]
            by_session[session] = max(by_session.get(session, 0), row["end_s"])
        report["last_sessions_s"] = dict(sorted(by_session.items(), key=lambda item: -item[1])[:3])
        reports[case] = report
    pairs = {}
    for left, right in PAIRS:
        if not {left, right} <= reports.keys():
            continue
        a, b = reports[left], reports[right]
        delta = compare_arms(phases[left], phases[right], a, b)
        pairs[f"{left}_vs_{right}"] = {
            "both_policy_receipts_complete": a["P_order"]["receipt_complete"] and b["P_order"]["receipt_complete"],
            "throughput_pct": 100 * (b["throughput_tokens_s"] / a["throughput_tokens_s"] - 1),
            "output_pct": 100 * (b["output_tokens"] / a["output_tokens"] - 1),
            "elapsed_savings_s": a["duration_s"] - b["duration_s"],
            "observed_phase_swap": delta,
        }
    return {"run_id": manifest["run_id"], "trace_sha256": sha256(path), "arms": reports, "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(args.events, read_json(args.manifest)))


if __name__ == "__main__":
    main()
