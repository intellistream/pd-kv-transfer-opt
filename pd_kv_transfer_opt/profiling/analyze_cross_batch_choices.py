"""Replay r20's sync batches with one staged GET and an unchanged publication order."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
from math import ceil
from pathlib import Path

from pd_kv_transfer_opt.common import write_json
from pd_kv_transfer_opt.profiling.analyze_packed_readiness import read_bundle, replay
from pd_kv_transfer_opt.profiling.audit_restore_worker_costs import get_case_nodes


STAGING_BYTES = 384 * 1024**2
SCATTER_NS = 8_560_000   # Rounded upper end of the r21 staged sample.
DEADLINES = ("common_host_forward", "first_successor_worker", "first_stock_get")


def free_windows(start, end, busy):
    cursor = start
    for left, right in busy:
        if cursor >= end or left >= end:
            break
        if right <= cursor:
            continue
        if cursor < left:
            yield cursor, left
        cursor = max(cursor, right)
    if cursor < end:
        yield cursor, end


def restore_jobs(node):
    order = list(node["reads"][0])
    if set(node["reads"]) != {0, 1, 2, 3} or any(list(rank) != order for rank in node["reads"].values()):
        raise ValueError("Expected four ranks with the same restore order")
    return [(name, [node["reads"][rank][name] for rank in range(4)]) for name in order]


def ms(ns):
    return round(ns / 1_000_000, 6)


def evaluate(node, jobs, busy, window, cap, factor, prep_ns, bound, baseline):
    start, end = window
    intervals = list(free_windows(start + prep_ns, end, busy))
    candidates = []
    for index, (name, reads) in enumerate(jobs):
        if bound is not None and max(read.size for read in reads) > bound:
            continue
        first = min(read.begin for read in reads)
        gang = max(read.end for read in reads) - first
        floor_ns = ceil(8 * sum(read.size for read in reads) / cap)
        scaled = max(factor, floor_ns / gang)
        offsets = [ceil((read.end - first) * scaled) for read in reads]
        slot = next(((left, right) for left, right in intervals if left + max(offsets) <= right), None)
        if slot is None:
            continue
        launch, slot_end = slot
        complete = [launch + offset for offset in offsets]
        forward = replay(node, staged=(index, complete, SCATTER_NS))[1]
        candidates.append({
            "name": name.split("~", 1)[-1], "ordinal": index, "launch_ns": launch,
            "predicted_get_width_ms": ms(max(offsets)), "ready_slack_ms": ms(slot_end - max(complete)),
            "advance_ms": ms(baseline - forward),
        })
    if not candidates:
        return {"choices": [], "first": None, "longest": None, "critical": None,
                "extra_ms": 0.0, "extra_over_longest_ms": 0.0}
    first = min(candidates, key=lambda row: (row["launch_ns"], row["ordinal"]))
    longest = min(candidates, key=lambda row: (-row["predicted_get_width_ms"], row["launch_ns"], row["ordinal"]))
    critical = min(candidates, key=lambda row: (-row["advance_ms"], row["launch_ns"], row["ordinal"]))
    return {"choices": candidates, "first": first["name"], "longest": longest["name"], "critical": critical["name"],
            "extra_ms": round(critical["advance_ms"] - first["advance_ms"], 6),
            "extra_over_longest_ms": round(critical["advance_ms"] - longest["advance_ms"], 6)}


def observation_windows(node, prior, jobs):
    batch = node["batch"]
    current_entry = min(row["worker_enter_ns"] for row in node["forwards"].values())
    first_get = min(read.begin for _, reads in jobs for read in reads)
    prior_forwards = prior["forwards"].values()
    prior_get_end = max((r.end for rank in prior["reads"].values() for r in rank.values()), default=0)
    # Give all candidates the same release and use both instances when subtracting GET traffic.
    start = max(batch["dispatch_ns"], prior_get_end, max(r["forward_enter_ns"] for r in prior_forwards))
    ends = (
        min(current_entry, min(r["forward_cpu_return_ns"] for r in prior_forwards)),
        min(current_entry, prior["batch"]["ready_ns"]),
        first_get,
    )
    return dict(zip(DEADLINES, ((start, end) for end in ends), strict=True))


def inspect(case, node, prior, busy, tails, cap=None):
    batch = node["batch"]
    jobs = restore_jobs(node)
    baseline = replay(node)[1]
    if baseline != max(row["forward_enter_ns"] for row in node["forwards"].values()):
        raise ValueError("Synchronous trace was not reconstructed exactly")
    windows = observation_windows(node, prior, jobs)
    if cap is None:
        cap = int(case.split("-cap", 1)[1])
    panels = []
    for (limit, window), prep_ms, speed, bound in product(
            windows.items(), (0, 4), (0.75, 1, 1.25), (STAGING_BYTES, None)):
        row = evaluate(node, jobs, busy, window, cap, speed, prep_ms * 1_000_000, bound, baseline)
        panels.append({"deadline": limit, "prep_ms": prep_ms, "service_multiplier": speed,
                       "slot_bytes_per_rank": bound, **row})
    return {
        "case": case, "cap_Gbps": cap, "instance": batch["instance"], "batch": batch["batch"],
        "predecessor": prior["batch"]["batch"],
        "predecessor_get_requests": len(prior["reads"].get(0, {})),
        "scheduled_tokens": batch["total"],
        "request_chunks": [(name.split("~", 1)[-1], offset, tokens) for name, offset, tokens in node["rows"]],
        "contains_TTFT_tail_chunk": tails["ttft"] in node["names"],
        "contains_drain_tail_chunk": tails["drain"] in node["names"],
        "get_free_window_ms": {key: ms(max((right-left for left, right in free_windows(*window, busy)), default=0))
                               for key, window in windows.items()},
        "jobs": [{
            "name": name.split("~", 1)[-1], "bytes_per_rank": [read.size for read in reads],
            "stock_get_width_ms": ms(max(read.end for read in reads) - min(read.begin for read in reads)),
            "local_advance_if_already_staged_ms": ms(
                baseline - replay(node, staged=(index, [0] * 4, SCATTER_NS))[1]),
        } for index, (name, reads) in enumerate(jobs)],
        "replay_sensitivity": panels,
    }


def analyze(prefill, decode):
    if prefill["scope"][0]["run_id"] != decode["scope"][0]["run_id"]:
        raise ValueError("Prefill and decode traces came from different runs")
    by_case = get_case_nodes(prefill, decode)
    groups = []
    count = Counter()
    for case, nodes in sorted(by_case.items()):
        ends = [row for row in decode["client"] if row["type"] == "request_end" and row["case"] == case]
        if len(ends) != 10 or any(row["status"] != "ok" for row in ends):
            raise ValueError(f"Expected ten successful targets for {case}")
        tails = {"ttft": max(ends, key=lambda row: row["first_s"] - row["offered_s"])["id"],
                 "drain": max(ends, key=lambda row: row["end_s"])["id"]}
        kind = "synchronous" if case.startswith("a-") else "asynchronous"
        count[f"{kind}_arms"] += 1
        count[f"{kind}_formal_batches"] += len(nodes)
        count["formal_batches"] += len(nodes)
        busy = sorted((read.begin, read.end) for node in nodes for rank in node["reads"].values()
                      for read in rank.values())
        previous = {}
        for node in sorted(nodes, key=lambda item: item["batch"]["dispatch_ns"]):
            instance = node["batch"]["instance"]
            if len(node["reads"].get(0, ())) > 1:
                count[f"{kind}_multi_restore_batches"] += 1
                if kind == "synchronous":
                    groups.append(inspect(case, node, previous[instance], busy, tails))
            previous[instance] = node
    return {
        "run_id": prefill["scope"][0]["run_id"], "counts": dict(count),
        "scope": "Fixed P batch releases, restore order, rank pair joins, copy durations and bytes. "
                 "The endpoint is the last host forward entry in the successor. GETs run before "
                 "their original slots and still publish in the same order. All selectors use the "
                 "same candidate set. Finite slot screens payload only. The recorded durations are "
                 "oracle inputs, not online estimates. No port headroom, page lease, predecessor slowdown, "
                 "device-kernel overlap, TTFT or throughput effects are inferred from this replay.",
        "synchronous_multi_restore_batches": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("prefill", "decode", "output"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    result = analyze(read_bundle(args.prefill), read_bundle(args.decode))
    result["source_sha256"] = {role: sha256(path.read_bytes()).hexdigest()
                              for role, path in (("P", args.prefill), ("D", args.decode))}
    write_json(args.output, result)


if __name__ == "__main__":
    main()
