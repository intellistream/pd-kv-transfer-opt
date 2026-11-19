"""Audit observed prefill dispatch waits against GETs from older batches."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from pd_kv_transfer_opt.common import write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import (
    interval_union,
    normalized_proxy_id,
    read_bundle,
)
from pd_kv_transfer_opt.profiling.analyze_packed_readiness import (
    client_ids,
    join_forwards,
    request_names,
)
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import stats


COUNTS = (
    "batches",
    "batches_without_own_get",
    "batches_overlapping_older_get_while_waiting_for_worker",
    "overlapping_batches_without_own_get",
    "overlapping_batches_without_pending_predecessor_result_at_dispatch",
    "overlapping_get_free_batches_disjoint_from_all_older_get_batches",
    "overlapping_batches_with_own_get_disjoint_from_all_older_get_batches",
    "request_chunk_rows",
    "first_chunks_with_own_get",
    "first_admission_to_dispatch_intersects_older_get",
    "later_rows_without_own_get_in_overlapping_batches",
    "later_rows_with_result_delivered_before_worker_and_while_older_get_in_flight",
    "first_chunks_delayed_with_unrelated_older_get_and_own_get",
)


def overlap_ms(start, end, intervals):
    clipped = [(max(start, left), min(end, right)) for left, right in intervals
               if max(start, left) < min(end, right)]
    return interval_union(clipped) / 1e6


def observe_waits(nodes, arrivals, case):
    counts = Counter(dict.fromkeys(COUNTS, 0))
    waits, waits_without_overlap = [], []
    earlier_get_overlap, get_free_overlap = [], []
    no_get_witness = None
    early_transfer_witness = None
    previous = {}
    for node in sorted(nodes, key=lambda item: item["batch"]["dispatch_ns"]):
        batch = node["batch"]
        dispatch = batch["dispatch_ns"]
        entry = min(row["worker_enter_ns"] for row in node["forwards"].values())
        older = [other for other in nodes if other["batch"]["instance"] == batch["instance"]
                 and other["batch"]["dispatch_ns"] < dispatch]
        older_reads = [(start, end) for other in older for start, end in other["gets"]]
        overlapping = [other for other in older if overlap_ms(dispatch, entry, other["gets"])]
        overlap = overlap_ms(dispatch, entry, older_reads)
        wait_ms = (entry - dispatch) / 1e6
        waits.append(wait_ms)
        counts["batches"] += 1
        counts["batches_without_own_get"] += not node["gets"]
        if overlap:
            earlier_get_overlap.append(overlap)
            counts["batches_overlapping_older_get_while_waiting_for_worker"] += 1
            if all(previous[rid]["batch"]["ready_ns"] <= dispatch
                   for rid in node["request_ids"] if rid in previous):
                counts["overlapping_batches_without_pending_predecessor_result_at_dispatch"] += 1
            disjoint = all(node["request_ids"].isdisjoint(other["request_ids"]) for other in overlapping)
            if node["gets"]:
                counts["overlapping_batches_with_own_get_disjoint_from_all_older_get_batches"] += disjoint
            else:
                get_free_overlap.append(overlap)
                counts["overlapping_batches_without_own_get"] += 1
                counts["overlapping_get_free_batches_disjoint_from_all_older_get_batches"] += disjoint
                if no_get_witness is None or overlap > no_get_witness["older_get_overlap_ms"]:
                    chunk_history = {}
                    for rid, offset, tokens in node["rows"]:
                        prev = previous.get(rid)
                        prev_chunk = next((row[1:] for row in prev["rows"] if row[0] == rid), None) if prev else None
                        chunk_history[rid.split("~", 1)[1]] = {
                            "previous_offset_and_tokens": prev_chunk,
                            "current_offset_and_tokens": [offset, tokens],
                            "previous_scheduler_result_from_dispatch_ms": (
                                (prev["batch"]["ready_ns"] - dispatch) / 1e6 if prev else None
                            ),
                        }
                    no_get_witness = {"case": case, "instance": batch["instance"], "batch": batch["batch"],
                                      "dispatch_to_worker_ms": wait_ms, "older_get_overlap_ms": overlap,
                                      "previous_chunks": chunk_history,
                                      "older_get_batches": [other["batch"]["batch"] for other in overlapping]}
        else:
            waits_without_overlap.append(wait_ms)

        for rid, _, _ in node["rows"]:
            counts["request_chunk_rows"] += 1
            if rid in previous:
                if overlap and rid not in node["get_starts"]:
                    counts["later_rows_without_own_get_in_overlapping_batches"] += 1
                    prior_result = previous[rid]["batch"]["ready_ns"]
                    counts["later_rows_with_result_delivered_before_worker_and_while_older_get_in_flight"] += (
                        overlap_ms(max(prior_result, dispatch), entry, older_reads) > 0
                    )
            else:
                own_reads = node["get_starts"].get(rid, ())
                counts["first_chunks_with_own_get"] += bool(own_reads)
                counts["first_admission_to_dispatch_intersects_older_get"] += (
                    overlap_ms(arrivals[rid], dispatch, older_reads) > 0
                )
                if own_reads and overlap and all(rid not in other["request_ids"] for other in overlapping):
                    counts["first_chunks_delayed_with_unrelated_older_get_and_own_get"] += 1
                    if early_transfer_witness is None or overlap > early_transfer_witness["older_get_overlap_ms"]:
                        early_transfer_witness = {
                            "case": case, "instance": batch["instance"], "batch": batch["batch"],
                            "request": rid.split("~", 1)[1], "dispatch_to_worker_ms": wait_ms,
                            "older_get_overlap_ms": overlap,
                            "dispatch_to_first_own_get_ms": (min(own_reads) - dispatch) / 1e6,
                            "other_requests_in_successor": sorted(
                                other.split("~", 1)[1] for other in node["request_ids"] if other != rid
                            ),
                        }
            previous[rid] = node

    return {
        "case": case, "counts": dict(counts),
        "dispatch_to_first_worker_ms": stats(waits),
        "dispatch_to_first_worker_without_older_get_overlap_ms": stats(waits_without_overlap),
        "older_get_overlap_while_waiting_ms": stats(earlier_get_overlap),
        "older_get_overlap_while_waiting_for_get_free_batches_ms": stats(get_free_overlap),
        "longest_get_free_batch_witness": no_get_witness,
        "longest_possible_early_transfer_witness": early_transfer_witness,
    }


def audit(prefill, decode):
    names = client_ids(decode)
    arrivals = {names.get(normalized_proxy_id(row["request_id"])): row["arrival_ns"]
                for row in prefill["native"] if row["type"] == "p_chunk_arrival"}
    cases = {row["id"]: manifest["case"] for manifest in decode["client_manifest"] for row in manifest["request_sources"]
             if manifest["phase"] == "target" and manifest["case"].startswith(("a-", "b-"))}
    batches, joined, errors, unmatched = join_forwards(prefill)
    if errors or unmatched or len(batches) != len(joined):
        raise ValueError("Incomplete scheduler/worker join")
    by_case = defaultdict(list)
    observed = set()
    for node in joined:
        node["rows"] = [(names.get(normalized_proxy_id(rid)), offset, count)
                        for rid, offset, count in node["batch"]["rows"]]
        node["request_ids"] = {row[0] for row in node["rows"]}
        matching = {cases.get(rid) for rid in node["request_ids"]}
        if matching == {None}:
            continue
        if len(matching) != 1 or None in matching:
            raise ValueError("Measured batch crosses a case boundary")
        case, = matching
        observed.update(node["request_ids"])
        node["get_starts"] = defaultdict(list)
        node["gets"] = []
        for load in node["loads"].values():
            reads = request_names(load, names)
            if len(reads) != len(load.reads):
                raise ValueError("Missing stable GET request attribution")
            for rid, read in zip(reads, load.reads, strict=True):
                node["get_starts"][rid].append(read.begin)
                node["gets"].append((read.begin, read.end))
        by_case[case].append(node)

    if not by_case or observed != cases.keys():
        raise ValueError("Missing formal targets in joined batches")
    report = [observe_waits(nodes, arrivals, case) for case, nodes in sorted(by_case.items())]
    totals = Counter()
    for row in report:
        totals.update(row["counts"])
    return {"run_id": prefill["scope"][0]["run_id"], "timed_target_requests": len(cases),
            "all_joined_batches": len(joined), "counts": dict(totals), "arms": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("r19_prefill", "r19_decode", "r20_prefill", "r20_decode", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    write_json(args.output, {
        "definitions": {
            "dispatch": "Return from the scheduler's schedule method with a concrete batch output.",
            "scheduler_result": "Entry to scheduler update_from_output, after the previous batch result is collected.",
            "worker_entry": "Earliest host execute_model entry among four ranks of the successor batch.",
            "overlap": "Union of earlier same-instance formal GET windows clipped to the successor dispatch-to-worker interval.",
            "limits": "A scheduler output can contain dependent prefill chunks. Missing a prior scheduler result cannot establish "
                      "when intermediate device state became visible. A new request with its own GET can expose an earlier "
                      "transfer candidate, which does not establish prefill readiness or the allocator/connector release time.",
        },
        "runs": [audit(read_bundle(args.r19_prefill), read_bundle(args.r19_decode)),
                 audit(read_bundle(args.r20_prefill), read_bundle(args.r20_decode))],
    })


if __name__ == "__main__":
    main()
