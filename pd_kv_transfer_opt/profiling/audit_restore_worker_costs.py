"""Account for r19/r20 restore waits on the last rank to enter host forward."""

import argparse
from collections import defaultdict
from pathlib import Path

from pd_kv_transfer_opt.common import write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import normalized_proxy_id, read_bundle
from pd_kv_transfer_opt.profiling.analyze_packed_readiness import client_ids, join_forwards, request_names
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import stats


def ms(ns):
    return round(ns / 1e6, 6)


def get_case_nodes(prefill, decode):
    names = client_ids(decode)
    cases = {row["id"]: manifest["case"] for manifest in decode["client_manifest"]
             if manifest["phase"] == "target" and manifest["case"].startswith(("a-", "b-"))
             for row in manifest["request_sources"]}
    batches, nodes, errors, unmatched = join_forwards(prefill)
    if errors or unmatched or len(batches) != len(nodes):
        raise ValueError("Incomplete scheduler/worker join")
    by_case = defaultdict(list)
    seen = set()
    for node in nodes:
        node["rows"] = [(names.get(normalized_proxy_id(rid)), offset, tokens)
                        for rid, offset, tokens in node["batch"]["rows"]]
        node["names"] = {name for name, _, _ in node["rows"]}
        matching = {cases.get(name) for name in node["names"]}
        if matching == {None}:
            continue
        if len(matching) != 1 or None in matching:
            raise ValueError("Formal batch contains requests from different cases")
        case, = matching
        seen.update(node["names"])
        node["reads"] = {}
        for rank, load in node["loads"].items():
            sequence = request_names(load, names)
            if load.verified or len(set(sequence)) != len(sequence) or set(sequence) - node["names"]:
                raise ValueError("Validation, duplicate or non-batch read in formal load")
            node["reads"][rank] = dict(zip(sequence, load.reads, strict=True))
        by_case[case].append(node)
    if seen != cases.keys():
        raise ValueError("Missing formal requests in scheduler trace")
    return by_case


def worker_path(node, events):
    rank = max(node["forwards"], key=lambda k: node["forwards"][k]["forward_enter_ns"])
    forward = node["forwards"][rank]
    load = node["loads"][rank]
    enter, start = forward["worker_enter_ns"], load.begin["monotonic_ns"]
    end, fwd = load.end["monotonic_ns"], forward["forward_enter_ns"]
    selected = [row for row in events[forward["pid"], forward["thread_id"], forward["batch"]]
                if start <= row["monotonic_ns"] <= end]
    preflight = next((row for row in selected if row["type"] == "r18_packed_preflight"), None)
    fences = {(row["request_id"], row["call_ordinal"]): row for row in selected if row["type"] == "r20_copy_wait"}

    first_get = load.reads[0].begin
    windows = [(enter, start, "worker_to_load"), (start, first_get, "load_to_first_get"),
               (end, fwd, "load_end_to_forward")]
    pair_peer_get = pair_after_both = 0
    calls = []
    for name, read in node["reads"][rank].items():
        windows.append((read.begin, read.end, "get"))
        call = {"request": name, "get_bytes": read.size, "get_ms": ms(read.end - read.begin)}
        calls.append(call)
        if read.copy is None:
            continue
        pair_begin = read.copy_begin - read.copy["pair_wait_ns"]
        windows.append((pair_begin, read.copy_begin, "pair_exchange"))
        peer = node["reads"][rank ^ 1][name]
        peer_join = peer.copy_begin - peer.copy["pair_wait_ns"]
        peer_get = max(0, min(read.copy_begin, peer_join, peer.end) - max(pair_begin, peer.begin))
        both_joined = max(0, read.copy_begin - max(pair_begin, peer_join))
        pair_peer_get += peer_get
        pair_after_both += both_joined
        call.update(copy_device_ms=read.copy["device_elapsed_ms"], copy_bytes=read.copy["payload_bytes"],
                    pair_exchange_ms=ms(read.copy["pair_wait_ns"]), peer_get_ms=ms(peer_get),
                    pair_after_both_ms=ms(both_joined))
        fence = fences.pop((read.request_id, read.call_ordinal), None)
        if fence is None:
            windows.append((read.copy_begin, read.returned, "sync_copy"))
        else:
            windows.append((read.copy_begin, fence["enqueue_end_ns"], "async_copy_submit"))
            windows.append((fence["wait_start_ns"], fence["wait_end_ns"], "async_copy_fence"))
            call["copy_fence_ms"] = ms(fence["wait_end_ns"] - fence["wait_start_ns"])

    costs = defaultdict(int)
    cursor = enter
    for left, right, phase in sorted(windows):
        if not cursor <= left <= right <= fwd:
            raise ValueError(f"Overlapping or invalid worker interval for {phase}")
        costs["unattributed"] += left - cursor
        costs[phase] += right - left
        cursor = right
    if cursor != fwd or fences:
        raise ValueError("Incomplete worker path accounting")
    return {
        "instance": forward["instance"], "batch": forward["batch"], "last_forward_rank": rank,
        "restore_order": list(node["reads"][rank]), "worker_to_forward_ms": ms(fwd - enter),
        "calls": calls,
        "exclusive_ms": {phase: ms(value) for phase, value in sorted(costs.items())},
        "preflight_plan_ms_within_load_to_first_get": ms(preflight["plan_ns"]) if preflight else None,
        "preflight_pair_ms_within_load_to_first_get": ms(preflight["wait_ns"]) if preflight else None,
        "in_pair_exchange_ms": {"peer_same_get_in_flight": ms(pair_peer_get),
                                "after_both_joined": ms(pair_after_both),
                                "other": ms(costs["pair_exchange"] - pair_peer_get - pair_after_both)},
    }


def inspect_gap(nodes):
    matches = [(node, name) for node in nodes for name in node["reads"].get(0, {})
               if name.endswith("~natural-video-processing:28") and len(node["reads"][0]) == 1]
    (node, name), = matches
    batch = node["batch"]
    dispatch = batch["dispatch_ns"]
    predecessors = [other for other in nodes if other["batch"]["instance"] == batch["instance"]
                    and other["batch"]["dispatch_ns"] < dispatch and other["reads"].get(0)]
    prior = max(predecessors, key=lambda other: other["batch"]["dispatch_ns"])
    old_end = max(read.end for reads in prior["reads"].values() for read in reads.values())
    own_enter = min(row["worker_enter_ns"] for row in node["forwards"].values())
    own = [reads[name] for reads in node["reads"].values()]
    overlap = any(max(old_end, read.begin) < min(own_enter, read.end)
                  for other in nodes for reads in other["reads"].values() for read in reads.values())
    prior_forward = max(max(0, min(own_enter, row["forward_cpu_return_ns"]) - max(old_end, row["forward_enter_ns"]))
                        for row in prior["forwards"].values())
    return {
        "request": name, "instance": batch["instance"], "batch": batch["batch"],
        "predecessor_batch": prior["batch"]["batch"], "scheduled_tokens": sum(row[2] for row in batch["rows"]),
        "request_offset_tokens": node["rows"], "predecessor_request_offset_tokens": prior["rows"],
        "dispatch_to_ms": {
            "predecessor_last_get_end": ms(old_end - dispatch),
            "predecessor_last_host_forward_enter": ms(max(row["forward_enter_ns"] for row in prior["forwards"].values()) - dispatch),
            "predecessor_scheduler_result": ms(prior["batch"]["ready_ns"] - dispatch),
            "first_worker": ms(own_enter - dispatch), "first_own_get": ms(min(read.begin for read in own) - dispatch),
            "last_own_copy_return": ms(max(read.returned for read in own) - dispatch),
        },
        "old_get_end_to_new_worker_ms": ms(own_enter - old_end), "observed_get_overlaps_gap": overlap,
        "longest_single_rank_host_forward_in_gap_ms": ms(prior_forward),
        "four_rank_get_window_ms": ms(max(read.end for read in own) - min(read.begin for read in own)),
        "four_rank_get_bytes": sum(read.size for read in own),
    }


def audit(prefill, decode, include_gap):
    events = defaultdict(list)
    for row in prefill["native"]:
        if row["type"] in {"r18_packed_preflight", "r20_copy_wait"} and "batch" in row:
            events[row["pid"], row["thread_id"], row["batch"]].append(row)
    arms = {}
    for case, nodes in sorted(get_case_nodes(prefill, decode).items()):
        paths = [worker_path(node, events) for node in nodes if any(node["reads"].values())]
        arms[case] = {
            "restore_batches": len(paths),
            "plan_ms_on_last_host_forward_rank": stats(row["preflight_plan_ms_within_load_to_first_get"]
                                                       for row in paths if row["preflight_plan_ms_within_load_to_first_get"] is not None),
            "paths": paths,
        }
        if include_gap and case.startswith("a-"):
            arms[case]["successor_get_gap"] = inspect_gap(nodes)
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("r19_prefill", "r19_decode", "r20_prefill", "r20_decode", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    write_json(args.output, {
        "scope": "Formal target batches. Exclusive host costs start at the selected rank's own worker entry. "
                 "Pair exchange subdivisions and preflight intervals are nested. GET timings are host call durations. "
                 "The gap checks formal P GETs on both instances, not port idleness or safe write readiness.",
        "r19": audit(read_bundle(args.r19_prefill), read_bundle(args.r19_decode), False),
        "r20": audit(read_bundle(args.r20_prefill), read_bundle(args.r20_decode), True),
    })


if __name__ == "__main__":
    main()
