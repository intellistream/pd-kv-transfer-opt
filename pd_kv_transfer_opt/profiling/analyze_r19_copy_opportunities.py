"""Locate same-batch opportunities to overlap packed distribution with the next GET."""

import argparse
from collections import defaultdict
from pathlib import Path

from pd_kv_transfer_opt.common import write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union, normalized_proxy_id, read_bundle
from pd_kv_transfer_opt.profiling.analyze_packed_readiness import client_ids, join_forwards, replay, request_names
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import stats


HEAD_PAIRS = ((0, 1), (2, 3))
SUBMISSION_OVERHEAD_MS = (0, 10)


def pipelined_forward_ns(node, submit_ns):
    """Hold observed service and pair release delays fixed, with one pending copy per rank."""
    starts = []
    for pair in HEAD_PAIRS:
        loads = [node["loads"][rank] for rank in pair]
        cursor = [load.begin["monotonic_ns"] for load in loads]
        previous = cursor.copy()
        copy_finish = cursor.copy()
        for reads in zip(*(load.reads for load in loads), strict=True):
            get_start = [ready + read.begin - last for ready, read, last in zip(cursor, reads, previous, strict=True)]
            get_finish = [begin + read.end - read.begin for begin, read in zip(get_start, reads, strict=True)]
            if any(read.copy is None for read in reads):
                raise ValueError("Expected matching packed reads on both ranks")
            observed_join = max(read.end for read in reads)
            # The r20 adapter drains the pending copy on both ranks before the next pair exchange.
            new_join = max(*get_finish, *copy_finish)
            copy_start = [new_join + read.copy_begin - observed_join for read in reads]
            copy_duration = [read.returned - read.copy_begin for read in reads]
            copy_finish = [begin + duration for begin, duration in zip(copy_start, copy_duration, strict=True)]
            cursor = [begin + min(submit_ns, duration) for begin, duration in zip(copy_start, copy_duration, strict=True)]
            previous = [read.returned for read in reads]
        for rank, ready, pending, last in zip(pair, cursor, copy_finish, previous, strict=True):
            # Retain the measured gap from the last distribution to this rank's forward.
            starts.append(max(ready, pending) + node["forwards"][rank]["forward_enter_ns"] - last)
    return max(starts)


def observe_intersections(copies, others, instance):
    """Return the wall-time union in each relation, with rank duplicates removed."""
    intersections = defaultdict(list)
    for copy_instance, batch, rank, copy_begin, copy_end in copies:
        if copy_instance != instance:
            continue
        for other_instance, other_batch, other_rank, begin, end in others:
            left, right = max(copy_begin, begin), min(copy_end, end)
            if left >= right:
                continue
            if other_instance != instance:
                key = "other_instance"
            elif other_batch != batch:
                key = "other_batch_same_instance"
            elif other_rank == rank:
                key = "same_rank_same_batch"
            elif other_rank == rank ^ 1:
                key = "same_head_peer_same_batch"
            else:
                key = "other_head_pair_same_batch"
            intersections[key].append((left, right))
    return {key: interval_union(intersections[key]) / 1e6 for key in (
        "same_rank_same_batch", "same_head_peer_same_batch", "other_head_pair_same_batch",
        "other_batch_same_instance", "other_instance")}


def analyze_case(case, nodes, names, targets, client_ends):
    tail = {"TTFT": max(client_ends, key=lambda row: row["first_s"] - row["ready_s"])["id"],
            "drain": max(client_ends, key=lambda row: row["end_s"])["id"]}
    copies, gets, forwards, candidates = [], [], [], []
    restore_batches = 0
    selected = []
    for node in nodes:
        batch = node["batch"]
        clients = {names.get(normalized_proxy_id(rid)) for rid, _, _ in batch["rows"]}
        if clients.isdisjoint(targets):
            continue
        selected.append(node)
        head = batch["instance"], batch["batch"]
        forwards.extend((*head, rank, row["forward_enter_ns"], row["forward_cpu_return_ns"])
                        for rank, row in node["forwards"].items())
        for rank, load in node["loads"].items():
            gets.extend((*head, rank, read.begin, read.end) for read in load.reads)
            copies.extend((*head, rank, read.copy_begin, read.returned)
                          for read in load.reads if read.copy is not None)
        if not any(load.reads for load in node["loads"].values()):
            continue
        restore_batches += 1
        loads = node["loads"]
        if len(loads) != 4 or any(load.verified for load in loads.values()):
            raise ValueError(f"Expected four timed packed loads in {case}")
        order = request_names(loads[0], names)
        if not order or not set(order) <= targets or any(request_names(load, names) != order for load in loads.values()):
            raise ValueError(f"Timed packed load order inconsistent in {case}")
        if len(order) == 1:
            continue
        observed = replay(node)[1]
        # Taking all observed copy time as synchronous submission reproduces the measured trace.
        synchronous_delay = max(read.returned - read.copy_begin for load in loads.values() for read in load.reads)
        if pipelined_forward_ns(node, synchronous_delay) != observed:
            raise ValueError(f"Replay failed to reproduce observed synchronous forward in {case}")
        steps = []
        for idx, (previous, following) in enumerate(zip(order, order[1:])):
            transitions = [(load.reads[idx], load.reads[idx + 1]) for load in loads.values()]
            steps.append({"copy_request": previous.split("~", 1)[-1], "next_get_request": following.split("~", 1)[-1],
                          "copy_active_ms": stats((read.returned - read.copy_begin) / 1e6 for read, _ in transitions),
                          "next_get_after_copy_ms": stats((next_read.begin - read.returned) / 1e6
                                                          for read, next_read in transitions),
                          "next_get_active_ms": stats((next_read.end - next_read.begin) / 1e6
                                                      for _, next_read in transitions)})
        candidates.append({"instance": batch["instance"], "batch": batch["batch"],
                           "critical_requests_in_restore": {kind: rid.split("~", 1)[-1]
                                                            for kind, rid in tail.items() if rid in order},
                           "copy_get_transitions": steps,
                           "replayed_latest_host_forward_advance_ms": {
                               str(overhead): (observed - pipelined_forward_ns(node, overhead * 1_000_000)) / 1e6
                               for overhead in SUBMISSION_OVERHEAD_MS}})
    observations = {}
    for instance in sorted({node["batch"]["instance"] for node in selected}):
        active = [(begin, end) for name, _, _, begin, end in copies if name == instance]
        observations[instance] = {"copy_active_union_ms": interval_union(active) / 1e6,
                                  "copy_intersections_with_GET_ms": observe_intersections(copies, gets, instance),
                                  "copy_intersections_with_host_forward_ms": observe_intersections(copies, forwards, instance)}
    return {"case": case, "timed_requests": len(targets), "matched_batches": len(selected),
            "batches_with_rank0_device_duration": sum(node["batch"].get("device", {}).get("forward_device_ms") is not None
                                                      for node in selected),
            "restore_batches": restore_batches, "multi_restore_batches": len(candidates),
            "observed": observations, "pipeline_candidates": candidates}


def analyze(prefill, decode):
    names = client_ids(decode)
    batches, nodes, errors, unmatched = join_forwards(prefill)
    if errors or unmatched or len(nodes) != len(batches):
        raise ValueError("Prefill batch, load, and rank forward coverage is incomplete")
    target_cases = [row for row in decode["client_manifest"] if row.get("phase") == "target"
                    and row.get("arm_mode") == "packed" and row.get("aggregate_D_egress_cap_Gbps")]
    results = []
    for manifest in sorted(target_cases, key=lambda row: (row["aggregate_D_egress_cap_Gbps"], row["case"])):
        case = manifest["case"]
        targets = {row["id"] for row in manifest["request_sources"]}
        client_ends = [row for row in decode["client"] if row["case"] == case and row["type"] == "request_end"]
        if {row["id"] for row in client_ends} != targets or any(row["status"] != "ok" for row in client_ends):
            raise ValueError(f"Timed client sources or completions disagree in {case}")
        measured = analyze_case(case, nodes, names, targets, client_ends)
        measured["D_egress_cap_Gbps"] = manifest["aggregate_D_egress_cap_Gbps"]
        results.append(measured)
    if not results:
        raise ValueError("Expected packed target cases")
    return {"submission_overhead_sweep_ms": SUBMISSION_OVERHEAD_MS,
            "observed_intersection": "Host monotonic windows. Rank-0 NPU timing gives duration without a shared device-time origin.",
            "replay": "Fixed batches, measured GET and copy service, one pending copy per rank, and a final fence. "
                      "Assumes disjoint destination addresses and unchanged service during overlap.",
            "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefill", "decode", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(read_bundle(args.prefill), read_bundle(args.decode)))


if __name__ == "__main__":
    main()
