"""Measure device-marked packed copy overlap with the next same-rank Store GET."""

import argparse
from collections import defaultdict
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union, normalized_proxy_id, read_bundle
from pd_kv_transfer_opt.profiling.analyze_natural_restore_ab import CASE, compare
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import stats


MODES = ("packed", "async-packed")
TIMING_MARGIN_NS = 100_000


def device_intersection(wait, get_begin, get_end):
    """Conservative copy-event/GET intersection on the host monotonic timeline."""
    duration = round(wait["device_elapsed_ms"] * 1e6)
    minimum = max(0, duration - TIMING_MARGIN_NS)
    maximum = duration + TIMING_MARGIN_NS
    start_earliest = wait["record_start_ns"]
    start_latest = wait["wait_end_ns"] - minimum
    end_earliest = start_earliest + minimum
    end_latest = wait["wait_end_ns"]
    for lo, hi, begun, done in wait["probes"]:
        if begun:
            start_latest = min(start_latest, hi)
        else:
            start_earliest = max(start_earliest, lo)
        if done:
            end_latest = min(end_latest, hi)
        else:
            end_earliest = max(end_earliest, lo)
    start_earliest = max(start_earliest, end_earliest - maximum)
    start_latest = min(start_latest, end_latest - minimum)
    end_earliest = max(end_earliest, start_earliest + minimum)
    consistent = start_earliest <= start_latest and end_earliest <= end_latest
    lower = (max(get_begin, start_latest), min(get_end, end_earliest))
    lower = lower if consistent and lower[1] > lower[0] else None
    upper = min(maximum, max(0, min(get_end, end_latest) - max(get_begin, start_earliest)))
    return lower, upper, consistent


def measured_copy_overlap(prefill, decode):
    case_by_pid = {}
    names = {}
    for row in decode["proxy"]:
        if row["type"] != "prefill_begin":
            continue
        case, _, name = row["id"].partition("~")
        if CASE.fullmatch(case):
            case_by_pid[row["proxy_request_id"]] = case
            names[row["proxy_request_id"]] = name

    def identity(row, ordinal):
        return row["instance"], row["pid"], row["thread_id"], row["batch"], ordinal

    begins = {}
    gets = {}
    copies = {}
    preflight_times = {}
    waits = []
    eligible = []
    for row in sorted(prefill["native"], key=lambda row: row["monotonic_ns"]):
        kind = row["type"]
        if (kind in ("store_begin", "store_end") and row.get("operation") == "batch_get_into_multi_buffers"
                and normalized_proxy_id(row.get("request_id")) in case_by_pid):
            if kind == "store_begin":
                begins[row["call_id"]] = row["monotonic_ns"]
            else:
                key = identity(row, row["call_ordinal"])
                gets[key] = begins.pop(row["call_id"]), row["monotonic_ns"], normalized_proxy_id(row["request_id"])
        elif kind == "r20_copy_wait" and normalized_proxy_id(row.get("request_id")) in case_by_pid:
            waits.append(row)
        elif kind == "same_head_local_copy" and normalized_proxy_id(row.get("request_id")) in case_by_pid:
            copies[identity(row, row["call_ordinal"])] = row
        elif kind == "r18_packed_preflight":
            preflight_times[identity(row, None)] = row["monotonic_ns"]
        elif kind == "r20_pipeline_batch":
            eligible.append(row)

    cases = defaultdict(list)
    batches = defaultdict(set)
    timed_batches = {(key[0], key[3]): case_by_pid[pid] for key, (_, _, pid) in gets.items()}
    for wait in waits:
        pid = normalized_proxy_id(wait["request_id"])
        case = case_by_pid[pid]
        copy = copies[identity(wait, wait["call_ordinal"])]
        row = dict(sorted(wait.items()), request=names[pid], next_request=None, device_overlap_lower_ns=None,
                   host_enqueue_ms=(wait["enqueue_end_ns"] - copy["copy_begin_ns"]) / 1e6,
                   device_overlap_upper_ns=0, pending_overlap_ns=0, probe_consistent=True)
        cases[case].append(row)
        ordinal = wait["next_call_ordinal"]
        if ordinal is None:
            continue
        begin, end, next_pid = gets[identity(wait, ordinal)]
        if case_by_pid[next_pid] != case:
            raise ValueError("Copy and next GET belong to different timed cases")
        if not wait["enqueue_end_ns"] <= begin <= end <= wait["wait_start_ns"]:
            raise ValueError("Unexpected copy enqueue, successor GET, or wait order")
        lower, upper, consistent = device_intersection(wait, begin, end)
        row.update(next_request=names[next_pid], device_overlap_lower_ns=lower, device_overlap_upper_ns=upper,
                   pending_overlap_ns=end - begin, probe_consistent=consistent)
        batches[case].add((wait["instance"], wait["batch"]))

    output = {}
    async_cases = sorted(row["case"] for row in decode["client_manifest"]
                         if row["arm_mode"] == MODES[1] and CASE.fullmatch(row["case"]))
    for case in async_cases:
        rows = cases[case]
        host = defaultdict(list)
        for row in rows:
            host[row["instance"]].append(row)
        by_prefill = {}
        for instance, items in sorted(host.items()):
            edges = [row for row in items if row["next_request"] is not None]
            lower_intervals = [tuple(row["device_overlap_lower_ns"]) for row in edges if row["device_overlap_lower_ns"]]
            by_prefill[instance] = {
                "rank_edges": len(edges), "rank_edges_with_positive_device_overlap_lower_bound": len(lower_intervals),
                "copy_window_and_GET_union_lower_bound_ms": interval_union(lower_intervals) / 1e6,
                "per_rank_device_overlap_lower_ms": stats((row["device_overlap_lower_ns"][1]
                                                           - row["device_overlap_lower_ns"][0]) / 1e6
                                                          if row["device_overlap_lower_ns"] else 0 for row in edges),
                "per_rank_device_overlap_upper_ms": stats(row["device_overlap_upper_ns"] / 1e6 for row in edges),
                "per_rank_copy_wait_ms": stats((row["wait_end_ns"] - row["wait_start_ns"]) / 1e6 for row in items),
                "per_rank_host_enqueue_ms": stats(row["host_enqueue_ms"] for row in items),
                "per_rank_pending_overlap_ms": stats(row["pending_overlap_ns"] / 1e6 for row in edges),
            }
        gates = [row for row in eligible if timed_batches.get((row["instance"], row["batch"])) == case]
        if not gates:
            raise ValueError(f"Missing pipeline eligibility checks in {case}")
        output[case] = {"by_prefill": by_prefill, "rank_copies_enqueued": len(rows),
                        "batches_with_next_GET": len(batches[case]),
                        "eligible_multi_restore_rank_batches": sum(row["eligible"] for row in gates),
                        "rank_batches_with_alias_fallback": sum(not row["eligible"] and row["jobs"] > 1 for row in gates),
                        "multi_restore_guard_ms": stats((row["monotonic_ns"] - preflight_times[identity(row, None)]) / 1e6
                                                        for row in gates if row["jobs"] > 1),
                        "probe_inconsistencies": sum(not row["probe_consistent"] for row in rows), "copies": rows}
    return output


def analyze(prefill, decode, workload):
    summary = compare(prefill, decode, workload, MODES, allow_pair_fixtures=True)
    overlap = measured_copy_overlap(prefill, decode)
    for pair in summary["paired_rounds"]:
        case = f"b-{pair['block']}-cap{pair['D_egress_cap_Gbps']}"
        controls = tuple(pair[mode] for mode in MODES)
        sync_requests, async_requests = (row["per_request"] for row in controls)
        different_gets = [name for name in sync_requests if sync_requests[name]["get_bytes"] != async_requests[name]["get_bytes"]]
        if (pair["difference_in_native_demand_bytes"] or pair["actual_GET_savings_bytes"]
                or different_gets or pair["requests_with_changed_P_work"]
                or controls[0]["P_to_D_descriptor_bytes"] != controls[1]["P_to_D_descriptor_bytes"]):
            raise ValueError(f"A/B changed physical GET bytes, APC work, or reverse transfer: {case}")
        pair["measured_overlap"] = overlap.get(case)
        for arm in controls:
            arm["copy"]["host_pending_span_ms"] = arm["copy"].pop("elapsed_ms")
            for instance in arm["P_by_instance"].values():
                instance["copy_pending_union_s"] = instance.pop("copy_active_union_s")
            for row in arm["per_request"].values():
                row["P_batch_copy_pending_union_ms"] = row.pop("P_batch_copy_active_union_ms")
        changes = pair["paired_request_stage_delta_packed_minus_baseline"]
        changes["P_batch_copy_pending_union_ms"] = changes.pop("P_batch_copy_active_union_ms")
        pair["paired_request_stage_delta_async_minus_sync"] = pair.pop("paired_request_stage_delta_packed_minus_baseline")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prefill", "decode", "workload", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(read_bundle(args.prefill), read_bundle(args.decode), read_json(args.workload)))


if __name__ == "__main__":
    main()
