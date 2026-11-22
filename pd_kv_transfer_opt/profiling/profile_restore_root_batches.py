"""Compare the actual recovery traffic and supply over the timed root and successor waves."""

import argparse
from collections import Counter
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union
from pd_kv_transfer_opt.profiling.profile_closed_sessions import (
    ROLES, measure_ports, read_trace, requests_for,
)


TOKEN_SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")


def prompt_sources(samples, time_s, origin):
    latest = {}
    for sample in samples:
        for engine in sample.get("engines", ()):
            at_s = engine["monotonic_ns"] / 1e9 - origin
            values = engine.get("values", {})
            if at_s > time_s or not values:
                continue
            role = engine["role"]
            if role not in latest or at_s > latest[role]["sample_at_s"]:
                counts = {name: sum(count for key, count in values.items()
                                    if key.startswith("vllm:prompt_tokens_by_source_total{") and f'source="{name}"' in key)
                          for name in TOKEN_SOURCES}
                latest[role] = {"sample_at_s": at_s, "tokens": counts}
    return latest


def measure_phase(trace, rows, origin, intervals, hot, begin, end):
    def clip(left, right):
        return max(0.0, min(right, end) - max(left, begin))

    before = prompt_sources(trace["telemetry"], begin, origin)
    after = prompt_sources(trace["telemetry"], end, origin)
    source_changes = {
        role: {"sample_window_s": [before[role]["sample_at_s"], after[role]["sample_at_s"]],
               "tokens": {name: after[role]["tokens"][name] - before[role]["tokens"][name]
                          for name in TOKEN_SOURCES}}
        for role in ROLES if role in before and role in after
    }
    tx = [sum(clip(left, right) * rates[j] / 8 for left, right, _, rates in intervals) for j in range(8)]
    hot_s = interval_union((max(left, begin), min(right, end)) for left, right in hot if clip(left, right))
    bins, = trace["client/output-bins.json"]
    return {"window_s": [begin, end], "D_TX_GB": sum(tx), "D_TX_GB_by_port": tx,
            "D_sample_seconds": sum(clip(left, right) for left, right, _, _ in intervals),
            "D_any_port_hot_s": hot_s, "prompt_source_sample_delta": source_changes,
            "P_HTTP_returns": sum(begin <= row["P_end_s"] < end for row in rows.values()),
            "D_first_tokens": sum(begin <= row["first_s"] < end for row in rows.values()),
            "client_output_tokens_in_overlapping_time_bins": sum(
                point["n_tokens"] for point in bins if begin <= point["at_s"] < end)}


def analyze_shared_history(workload, cases, digest):
    start = workload["pressure_window_s"][0]
    groups = workload["selection"]["shared_producer_groups"]
    arms, profiles = {}, {}
    for case, trace in cases.items():
        origin, = (row["monotonic_s"] for row in trace["client"] if row["type"] == "client_origin")
        rows = requests_for(trace, workload["request_sources"], origin)
        profiles[case] = rows
        tail = max(rows, key=lambda key: rows[key]["end_s"])
        end = rows[tail]["end_s"]
        intervals, _, hot, _ = measure_ports(trace["telemetry"], round(origin * 1e9), end, int(case.rsplit("cap", 1)[1]))
        phases = {name: measure_phase(trace, rows, origin, intervals, hot, left, right)
                  for name, left, right in (("producer_phase", 0, start), ("shared_history_burst_through_drain", start, end))}
        evidence = {}
        for task, group in groups.items():
            producer = rows[group["timed_producer"]]
            supplied = [key for key in group["later_original_requests"] if rows[key]["P"] != producer["P"]
                        and producer["D_start_s"] < rows[key]["P_start_s"]]
            first_return = min((rows[key]["P_end_s"] for key in supplied), default=None)
            admitted = [key for key in supplied if rows[key]["P_start_s"] < first_return]
            evidence[task] = {"producer": group["timed_producer"], "producer_events": producer,
                              "later_opposite_P_after_producer_D_admission": supplied,
                              "later_opposite_P_before_any_opposite_P_HTTP_return": admitted,
                              "upper_attached_shared_GET_bytes_for_early_admissions": len(admitted) * group["bytes_if_one_full_read"],
                              "last_later_end_s": max(rows[key]["end_s"] for key in group["later_original_requests"])}
        outputs = sum(row["output_tokens"] for row in rows.values())
        arms[case] = {"phases": phases, "groups": evidence, "output_tokens": outputs, "duration_s": end,
                      "throughput_tokens_s": outputs / end, "finish_reasons": dict(Counter(row["finish_reason"] for row in rows.values())),
                      "tail_id": tail, "tail": rows[tail]}
    matches = {}
    for order in ("b1", "b2"):
        low, high = (profiles.get(f"{order}-cap{cap}") for cap in (100, 200))
        if low is not None and high is not None:
            matches[order] = {role: sum(low[key][role] == high[key][role] for key in low) for role in ("P", "D")}
    return {"trace_sha256": digest, "arms": arms, "route_matches": matches,
            "admission_bounds_scope": "HTTP times bound overlap candidates. They do not locate the Store commit, lookup, rank reads, or APC publication."}


def analyze(archive):
    manifest = read_json(archive / "manifest.json")
    trace_path = archive / manifest["event_trace"]
    digest = sha256(trace_path)
    if manifest["artifacts"][trace_path.name]["sha256"] != digest:
        raise ValueError("The frozen trace hash does not match the receipt")
    workload = manifest["workload"]
    cases = read_trace(trace_path)
    if "shared_producer_groups" in workload["selection"]:
        return analyze_shared_history(workload, cases, digest)
    root_plan = workload["root_batch_calibration"]
    matches = [link for family in root_plan["checkpoint_matches"].values() for link in family]
    first, later, successor = (root_plan[n] for n in ("first_roots", "later_roots", "successor_offer_start_s"))
    wave_start = root_plan["later_root_offer_start_s"]
    groups = {"first_roots": first, "later_roots": later,
              "successors": [row["id"] for row in workload["request_sources"] if row["predecessor"] is not None]}
    arms, profiles = {}, {}
    for case, trace in cases.items():
        origin, = (row["monotonic_s"] for row in trace["client"] if row["type"] == "client_origin")
        rows = requests_for(trace, workload["request_sources"], origin)
        profiles[case] = rows
        tail = max(rows, key=lambda key: rows[key]["end_s"])
        end = rows[tail]["end_s"]
        intervals, _, hot, _ = measure_ports(trace["telemetry"], round(origin * 1e9), end, int(case.rsplit("cap", 1)[1]))
        windows = {"first_roots_only": (0, wave_start), "later_roots_before_successors": (wave_start, successor),
                   "after_successor_arrivals": (successor, end)}
        details = {name: measure_phase(trace, rows, origin, intervals, hot, *limits) for name, limits in windows.items()}
        evidence = []
        for link in matches:
            a, b = rows[link["earlier_root"]], rows[link["later_root"]]
            evidence.append({**link, "producer_D_first_s": a["first_s"], "producer_end_s": a["end_s"],
                             "later_P_begin_s": b["P_start_s"], "later_P_end_s": b["P_end_s"],
                             "P_changed": a["P"] != b["P"]})
        output = sum(row["output_tokens"] for row in rows.values())
        arms[case] = {"phases": details, "output_tokens": output, "duration_s": end, "throughput_tokens_s": output / end,
                      "finish_reasons": dict(Counter(row["finish_reason"] for row in rows.values())),
                      "tail_id": tail, "tail": rows[tail], "checkpoint_opportunities": evidence,
                      "last_P_delivered_by_group_s": {phase: max(rows[key]["P_end_s"] for key in keys)
                                                       for phase, keys in groups.items()}}
    pairs = {}
    for order in ("b1", "b2"):
        lo, hi = (profiles.get(f"{order}-cap{cap}") for cap in (100, 200))
        if lo is not None and hi is not None:
            pairs[order] = {phase: {role: sum(lo[key][role] == hi[key][role] for key in keys) for role in ("P", "D")}
                            for phase, keys in groups.items()}
    return {"trace_sha256": digest, "phases_use_same_D_clock": True, "arms": arms, "route_matches": pairs,
            "checkpoint_opportunity_scope": "D first token and finish bracket request progress. Store publication and a local Hybrid miss require their own events."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(args.archive))


if __name__ == "__main__":
    main()
