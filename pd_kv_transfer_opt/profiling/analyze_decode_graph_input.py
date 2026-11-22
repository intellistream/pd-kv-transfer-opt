"""Join host submissions and device dependencies at the start of a Decode step."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path

from pd_kv_transfer_opt.common import sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_decode_collective_alignment import cycles
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution, interval_union
from pd_kv_transfer_opt.profiling.export_cann import time_ns


MS = 1_000_000
OBSERVABLES = {
    "Enqueue@HcclAllreduce": "collective_enqueue",
    "Dequeue@HcclAllreduce": "collective_dequeue",
    "AscendCL@hcom_allReduce_": "collective_host",
    "AscendCL@aclrtSynchronizeStreamWithTimeout": "stream_sync",
    "AscendCL@aclrtSynchronizeEvent": "host_event_sync",
    "AscendCL@aclrtSynchronizeDeviceWithTimeout": "host_event_sync",
    "AscendCL@aclmdlRIExecuteAsync": "graph_submit",
    "MODEL_EXECUTE": "graph_device",
    "EVENT_WAIT": "device_wait",
}


def indexed_events(path):
    payload = json.loads(path.read_bytes())
    events = payload["traceEvents"] if isinstance(payload, dict) else payload
    grouped = defaultdict(list)
    for row in events:
        if row["ph"] != "X" or row["name"] not in OBSERVABLES:
            continue
        begin = time_ns(row["ts"])
        end = begin + time_ns(row["dur"])
        grouped[OBSERVABLES[row["name"]]].append(
            (begin, end, row["tid"], row.get("args", {}).get("connection_id")))
    return {key: (sorted(r[0] for r in rows), sorted(rows, key=lambda r: r[0])) for key, rows in grouped.items()}


def events_between(index, kind, left, right):
    ticks, rows = index.get(kind, ([], []))
    return rows[bisect_left(ticks, left):bisect_right(ticks, right)]


def host_wait_ns(index, thread, left, right):
    return interval_union((max(begin, left), min(end, right))
                          for kind in ("stream_sync", "host_event_sync")
                          for begin, end, tid, _ in index.get(kind, ([], []))[1]
                          if tid == thread and begin < right and end > left)


def rank_steps(folder):
    comm = folder / "communication.json"
    trace = folder / "trace_view.json"
    groups, widths = cycles(comm)
    if len(groups) < 3:
        raise ValueError(f"Short Decode capture in {folder.name}")
    index = indexed_events(trace)
    clocks = []
    for (anchor, _), (end, sequence) in zip(groups, groups[1:]):
        first, second = sequence[:2]
        first_entry, first_duration, _ = first
        second_entry, second_duration, _ = second
        first_done = first_entry + first_duration
        graph = events_between(index, "graph_submit", anchor, second_entry)
        if len(graph) != 1:
            continue
        submit, submitted, thread, connection = graph[0]
        model = [event for event in events_between(index, "graph_device", submit, second_entry)
                 if connection is not None and event[3] == connection]
        if len(model) != 1:
            continue
        graph_start = model[0][0]
        eager = events_between(index, "collective_host", anchor, first_entry)
        enqueue = events_between(index, "collective_enqueue", anchor, first_entry)
        dequeue = events_between(index, "collective_dequeue", anchor, first_entry)
        sync = [event for event in events_between(index, "stream_sync", anchor, submit)
                if event[2] == thread and event[1] <= submit]
        waited_first = [event for event in events_between(index, "device_wait", first_entry, first_done)
                        if event[0] < first_done and abs(event[1] - first_done) < 80_000]
        row = {
            "previous_allgather_end_ns": anchor, "allgather_end_ns": end,
            "first_allreduce_enter_ns": first_entry, "first_allreduce_done_ns": first_done,
            "second_allreduce_enter_ns": second_entry, "second_allreduce_done_ns": second_entry + second_duration,
            "graph_submit_ns": submit, "graph_submit_return_ns": submitted, "graph_start_ns": graph_start,
            "first_allreduce_in_graph": graph_start <= first_entry,
            "device_wait_near_first_completion_duration_ns": max((r[1] - r[0] for r in waited_first), default=0),
        }
        if sync:
            row["stream_sync_begin_ns"], row["stream_sync_end_ns"] = sync[-1][:2]
        if len(eager) == len(enqueue) == len(dequeue) == 1 and first_entry < graph_start:
            row["eager_enqueue_ns"], row["eager_enqueue_end_ns"] = enqueue[0][:2]
            row["eager_dequeue_ns"] = dequeue[0][0]
            row["eager_host_submit_ns"], row["eager_host_submit_return_ns"] = eager[0][:2]
            row["pre_eager_host_sync_duration_ns"] = host_wait_ns(index, thread, anchor, enqueue[0][0])
            row["post_first_before_graph_sync_duration_ns"] = host_wait_ns(index, thread, first_done, submit)
        clocks.append(row)
    return clocks, {"complete_collective_groups": len(groups), "full_joins": len(clocks), "allreduce_widths": widths,
                    "communication_sha256": sha256(comm), "trace_sha256": sha256(trace)}


def aligned_steps(ranks):
    times = [[row["allgather_end_ns"] for row in rank] for rank in ranks]
    for first in ranks[0]:
        a = first["allgather_end_ns"]
        match = [first]
        for ticks, rows in zip(times[1:], ranks[1:]):
            i = bisect_left(ticks, a)
            candidates = range(max(0, i - 1), min(len(ticks), i + 1))
            if not candidates:
                break
            index = min(candidates, key=lambda j: abs(ticks[j] - a))
            if abs(ticks[index] - a) > 2 * MS:
                break
            match.append(rows[index])
        if len(match) == len(ranks):
            yield match


def summarize(rows):
    measurements = defaultdict(list)
    pairs = []
    latest_ranks = {"first": Counter(), "second": Counter()}
    stage_fields = {
        "latest_first_arrival_ms": "first_allreduce_enter_ns",
        "latest_first_completion_ms": "first_allreduce_done_ns",
        "latest_second_arrival_ms": "second_allreduce_enter_ns",
        "latest_second_completion_ms": "second_allreduce_done_ns",
        "latest_graph_submit_ms": "graph_submit_ns",
        "latest_graph_start_ms": "graph_start_ns",
        "next_logits_allgather_ms": "allgather_end_ns",
    }
    for step in rows:
        anchor = max(row["previous_allgather_end_ns"] for row in step)
        data = {}
        for name, field in stage_fields.items():
            values = [row[field] for row in step]
            data[name] = (max(values) - anchor) / MS
        for number in ("first", "second"):
            field = number + "_allreduce_enter_ns"
            peers = sorted(range(len(step)), key=lambda rank: step[rank][field])
            fast, slow = step[peers[0]], step[peers[-1]]
            latest_ranks[number][peers[-1]] += 1
            data[number + "_arrival_spread_ms"] = (slow[field] - fast[field]) / MS
            data[number + "_last_arrival_to_completion_ms"] = (
                max(row[number + "_allreduce_done_ns"] for row in step) - slow[field]) / MS
            if number == "first" and "eager_enqueue_ns" in fast and "eager_enqueue_ns" in slow:
                data["first_late_minus_early_enqueue_ms"] = (slow["eager_enqueue_ns"] - fast["eager_enqueue_ns"]) / MS
                data["first_late_minus_early_host_sync_ms"] = (
                    slow["pre_eager_host_sync_duration_ns"] - fast["pre_eager_host_sync_duration_ns"]) / MS
                data["first_late_minus_early_queue_dispatch_ms"] = (
                    slow["eager_host_submit_ns"] - slow["eager_enqueue_ns"]
                    - fast["eager_host_submit_ns"] + fast["eager_enqueue_ns"]) / MS
                data["first_late_minus_early_runtime_to_device_ms"] = (
                    slow[field] - slow["eager_host_submit_ns"] - fast[field] + fast["eager_host_submit_ns"]) / MS
            if number == "second" or (fast["first_allreduce_in_graph"] and slow["first_allreduce_in_graph"]):
                data[number + "_late_minus_early_graph_submit_ms"] = (
                    slow["graph_submit_ns"] - fast["graph_submit_ns"]) / MS
                data[number + "_late_minus_early_submit_to_device_start_ms"] = (
                    slow["graph_start_ns"] - slow["graph_submit_ns"] - fast["graph_start_ns"] + fast["graph_submit_ns"]) / MS
                data[number + "_late_minus_early_graph_prefix_ms"] = (
                    slow[field] - slow["graph_start_ns"] - fast[field] + fast["graph_start_ns"]) / MS
            if number == "second" and "post_first_before_graph_sync_duration_ns" in slow:
                data["second_late_rank_post_first_to_graph_submit_ms"] = (
                    slow["graph_submit_ns"] - slow["first_allreduce_done_ns"]) / MS
                data["second_late_rank_post_first_host_sync_ms"] = slow["post_first_before_graph_sync_duration_ns"] / MS
        for name, value in data.items():
            measurements[name].append(value)
        pairs.append({"steps": [{k.replace("_ns", "_ms"): v / MS if k.endswith("_duration_ns") else (v - anchor) / MS
                                 for k, v in row.items() if k.endswith("_ns")} for row in step]})
    for current, following in zip(rows, rows[1:]):
        if all(a["allgather_end_ns"] == b["previous_allgather_end_ns"] for a, b in zip(current, following)):
            for name, field in (("next_latest_graph_start_interval_ms", "graph_start_ns"),
                                ("next_latest_first_arrival_interval_ms", "first_allreduce_enter_ns")):
                measurements[name].append((max(r[field] for r in following) - max(r[field] for r in current)) / MS)
    return {"aligned_successor_steps": len(rows),
            "first_allreduce_in_graph_rank_steps": sum(row["first_allreduce_in_graph"] for step in rows for row in step),
            "eager_first_allreduce_rank_steps": sum("eager_enqueue_ns" in row for step in rows for row in step),
            "latest_arrival_ranks": latest_ranks, "costs_ms": {k: distribution(v) for k, v in measurements.items()},
            "per_step": pairs}


def report(folders):
    ranks, provenance = zip(*(rank_steps(folder) for folder in folders))
    rows = list(aligned_steps(ranks))
    if len(rows) < 3:
        raise ValueError("Too few complete matched graph steps")
    return {"sources": provenance, **summarize(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank0_to_rank3", type=Path, nargs=4, help="ASCEND_PROFILER_OUTPUT for each rank")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, report(args.rank0_to_rank3))


if __name__ == "__main__":
    main()
