"""Measure cache revisit opportunities and port pressure in the shared-document replay."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_multiwave import aligned_tx_intervals
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, read_trace, requests_for
from pd_kv_transfer_opt.profiling.screen_mac_demand import align_prefill


def visits(profiles, specs, role, begin, complete):
    """Check for a prior completed request for the same document and engine."""
    first_finished = {}
    revisits = set()
    by_engine = defaultdict(set)
    for key, row in sorted(profiles.items(), key=lambda entry: (entry[1][begin], entry[0])):
        doc, engine = specs[key]["document_id"], row[role]
        pair = (doc, engine)
        previous = first_finished.get(pair, float("inf"))
        if previous <= row[begin]:
            revisits.add(key)
        first_finished[pair] = min(previous, row[complete])
        by_engine[engine].add(doc)
    return revisits, {engine: len(documents) for engine, documents in by_engine.items()}


def tx_window(samples, origin, cap, start, stop):
    hot = []
    rails = [0.0] * 8
    covered = 0.0
    for left, right, aggregate, rates in aligned_tx_intervals(samples, origin):
        left, right = max(left, start), min(right, stop)
        if left >= right:
            continue
        duration = right - left
        covered += duration
        for port, rate in enumerate(rates):
            rails[port] += rate * duration / 8
        if aggregate >= 0.85 * cap:
            hot.append((left, right))
    return {"s": [start, stop], "covered_s": covered, "tx_by_port_GB": rails,
            "aggregate_Gbps": sum(rails) * 8 / covered if covered else None,
            "aggregate_near_cap_s": interval_union(hot)}


def analyze(directory):
    manifest = read_json(directory / "manifest.json")
    workload = manifest["workload"]
    if workload["workload_type"] != "narrativeqa_document_questions":
        raise ValueError("Expected the audited shared-document workload")
    trace = directory / manifest["event_trace"]
    if sha256(trace) != manifest["artifacts"][trace.name]["sha256"]:
        raise ValueError("Trace checksum mismatch")
    specs = {row["id"]: row for row in workload["request_sources"]}
    reports = read_json(directory / "summary.json")["arms"]
    start, offered_stop = workload["pressure_window_s"]
    arms = {}
    for case, arm in read_trace(trace).items():
        anchor, = [row["anchor"] for row in arm["client"] if row["type"] == "client_origin"]
        origin = anchor["monotonic_ns"]
        profiles = requests_for(arm, list(specs.values()), origin / NS)
        p_revisited, p_docs = visits(profiles, specs, "P", "P_start_s", "P_end_s")
        d_revisited, d_docs = visits(profiles, specs, "D", "D_start_s", "end_s")
        report = reports[case]
        wave_ids = defaultdict(list)
        for key, row in specs.items():
            wave_ids[row["wave"]].append(key)
        waves = {}
        for wave, ids in wave_ids.items():
            rows = [profiles[key] for key in ids]
            waves[wave] = {"requests": len(rows), "P_return_range_s": [min(p["P_end_s"] for p in rows), max(p["P_end_s"] for p in rows)],
                           "P_revisited_prior_doc": sum(key in p_revisited for key in ids),
                           "D_revisited_prior_doc": sum(key in d_revisited for key in ids),
                           "P_revisited_while_D_has_no_finished_prior_doc": sum(key in p_revisited - d_revisited for key in ids),
                           "P_HTTP_p50_s": median(row["phases_s"]["P_HTTP"] for row in rows),
                           "D_to_first_p50_s": median(row["phases_s"]["D_to_first"] for row in rows),
                           "output_tokens_mean": sum(row["output_tokens"] for row in rows) / len(rows),
                           "output_tokens_p50": median(row["output_tokens"] for row in rows)}
        lead_in = [profiles[key] for key in wave_ids[0]]
        last_p = max(row["P_end_s"] for row in profiles.values())
        duration = report["duration_s"]
        network = {"P": align_prefill(arm["P-telemetry"], anchor), "D": arm["telemetry"]}
        windows = {"planned_pressure": (start, offered_stop), "from_pressure_start_to_last_P_return": (start, last_p)}
        arms[case] = {"output_tokens": report["output_tokens"], "duration_s": duration,
                      "drained_throughput_tokens_s": report["throughput_tokens_s"],
                      "lead_in_P_returns_before_pressure": sum(row["P_end_s"] <= start for row in lead_in),
                      "lead_in_all_responses_before_pressure": sum(row["end_s"] <= start for row in lead_in),
                      "P_document_counts_by_engine": p_docs, "D_document_counts_by_engine": d_docs,
                      "later_questions_not_dispatched_until_after_planned_pressure": sum(row["dispatch_s"] > offered_stop for row in profiles.values()),
                      "waves": waves, "windows": {key: {role: tx_window(samples, origin, report["cap_Gbps"], left, right)
                                                          for role, samples in network.items()}
                                                 for key, (left, right) in windows.items()}}
    return {"arms": arms, "trace_sha256": sha256(trace),
            "scope": "A prior completed request on the same engine makes a cache hit possible. Cache counters and "
                     "measured payloads determine actual reuse. MAC samples include background Store writes and protocol overhead."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(args.archive))


if __name__ == "__main__":
    main()
