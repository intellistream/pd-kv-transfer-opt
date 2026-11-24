"""Account for a Decode step at four-rank collective boundaries."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path

from pd_kv_transfer_opt.common import sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_decode_collective_alignment import cycles
from pd_kv_transfer_opt.profiling.analyze_decode_graph_input import aligned_steps
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution, interval_union
from pd_kv_transfer_opt.profiling.export_cann import time_ns


MS = 1_000_000
GRAPH = "AscendCL@aclmdlRIExecuteAsync"
DEVICE_GRAPH = "MODEL_EXECUTE"
REDUCE = "AscendCL@hcom_allReduce_"
STREAM_SYNC = "AscendCL@aclrtSynchronizeStreamWithTimeout"
EVENT_SYNC = "AscendCL@aclrtSynchronizeEvent"
HOST_DEVICE_SYNC = "AscendCL@aclrtSynchronizeDeviceWithTimeout"
GRAPH_SCOPE = "issue23::decode_graph_"
FORWARD_SCOPE = "issue23::decode_forward_"
FIELDS = ("compute", "copy", "device_event_wait", "uncovered")


class Coverage:
    def __init__(self, intervals):
        self.segments = []
        for left, right in sorted(intervals):
            if self.segments and left <= self.segments[-1][1]:
                start, end = self.segments[-1]
                self.segments[-1] = start, max(end, right)
            elif left < right:
                self.segments.append((left, right))
        self.ends = [right for _, right in self.segments]

    def clip(self, left, right):
        if right <= left:
            return []
        index = bisect_right(self.ends, left)
        selected = []
        for i in range(index, len(self.segments)):
            start, end = self.segments[i]
            if start >= right:
                break
            selected.append((max(start, left), min(end, right)))
        return selected


def load_kernels(path):
    data = defaultdict(list)
    cores = Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Start Time(us)"] in ("", "N/A") or row["Duration(us)"] in ("", "N/A"):
                continue
            start = time_ns(row["Start Time(us)"])
            stop = start + time_ns(row["Duration(us)"])
            core, op = row["Accelerator Core"], row["Type"]
            cores[core] += 1
            if core == "COMMUNICATION":
                if "allgather" in op.lower():
                    data["gather"].append((start, stop))
                continue
            data["compute"].append((start, stop))
            if "argmax" in op.lower():
                data["argmax"].append((start, stop))
    return data, dict(cores)


def load_runtime(path, *, with_batch_tags=False):
    data = defaultdict(list)
    payload = json.loads(path.read_bytes())
    events = payload["traceEvents"] if isinstance(payload, dict) else payload
    hardware = {row["pid"] for row in events if row["ph"] == "M" and row["name"] == "process_name"
                and row.get("args", {}).get("name") == "Ascend Hardware"}
    watched = {GRAPH, DEVICE_GRAPH, REDUCE, EVENT_SYNC, STREAM_SYNC, "EVENT_WAIT", "MEMCPY_ASYNC"}
    for row in events:
        if row["ph"] != "X":
            continue
        name = row["name"]
        if with_batch_tags and name.startswith((GRAPH_SCOPE, FORWARD_SCOPE)):
            prefix = GRAPH_SCOPE if name.startswith(GRAPH_SCOPE) else FORWARD_SCOPE
            begin = time_ns(row["ts"])
            data[prefix].append((begin, begin + time_ns(row["dur"]), row["tid"], name[len(prefix):]))
            continue
        if name not in watched and not (with_batch_tags and name == HOST_DEVICE_SYNC):
            continue
        if name in ("EVENT_WAIT", "MEMCPY_ASYNC", DEVICE_GRAPH) and row["pid"] not in hardware:
            continue
        start = time_ns(row["ts"])
        data[name].append((start, start + time_ns(row["dur"]), row["tid"], row.get("args", {}).get("connection_id")))
    return {name: sorted(rows) for name, rows in data.items()}


def tagged_submit(events, previous_end, first, second):
    """Prefer the matching device graph when host submission runs ahead."""
    device_ids = {connection for start, _, _, connection in events.get(DEVICE_GRAPH, ())
                  if previous_end < start <= second and connection is not None}
    matched = [row for row in events.get(GRAPH, ()) if row[3] in device_ids and row[0] <= second]
    return matched or [row for row in events.get(GRAPH, ()) if first < row[0] <= second]


def load(path, *, with_batch_tags=False):
    names = ("kernel_details.csv", "communication.json", "trace_view.json")
    kernel_path, comm_path, trace_path = (path / k for k in names)
    kernels, cores = load_kernels(kernel_path)
    communication, widths = cycles(comm_path)
    events = load_runtime(trace_path, with_batch_tags=with_batch_tags)
    gather = sorted(kernels["gather"], key=lambda row: row[1])
    if with_batch_tags and (not gather or len(communication) < 2 or len(communication[0][1]) < 2):
        raise ValueError(f"No complete Decode collective sequence in {path}")
    ends = [row[1] for row in gather]
    steps = []
    for (previous_end, _), (end, ops) in zip(communication, communication[1:]):
        pos = bisect_left(ends, end)
        near = min(gather[max(0, pos-1):pos+1], key=lambda row: abs(row[1]-end))
        if abs(near[1]-end) > 1500:
            raise ValueError(f"Unmatched logits AllGather in {path.name}")
        ops = [*ops, (near[0], end-near[0], 0.0)]
        first, second = ops[0][0], ops[1][0]
        graph = (tagged_submit(events, previous_end, first, second) if with_batch_tags
                 else [row for row in events.get(GRAPH, []) if first < row[0] <= second])
        eager = [row for row in events.get(REDUCE, []) if previous_end < row[0] <= first]
        submit = graph[0] if len(graph) == 1 else None
        sampler = [row for row in kernels["argmax"] if previous_end <= row[1] <= first]
        step = {"allgather_end_ns": end, "previous_allgather_end_ns": previous_end,
                "ops": ops, "sample_end_ns": sampler[-1][1] if sampler else None,
                "eager_submit_ns": eager[0][0] if len(eager) == 1 else None,
                "graph_submit_ns": submit[0] if submit else None, "graph_tid": submit[2] if submit else None}
        if with_batch_tags and submit is not None:
            for prefix in (GRAPH_SCOPE, FORWARD_SCOPE):
                matches = {tag for begin, finish, _, tag in events.get(prefix, ())
                           if begin <= submit[0] <= finish}
                if len(matches) == 1:
                    step["batch"], = matches
                    break
                if matches:
                    break
        steps.append(step)
    graph_tids = {row[2] for row in events.get(DEVICE_GRAPH, [])}
    timelines = {"compute": Coverage(kernels["compute"])}
    timelines["copy"] = Coverage((a, b) for a, b, *_ in events.get("MEMCPY_ASYNC", []))
    timelines["device_event_wait"] = Coverage((a, b) for a, b, tid, _ in events.get("EVENT_WAIT", []) if tid in graph_tids)
    return steps, timelines, events, {"sha256": {name: sha256(path / name) for name in names},
                                      "collective_widths": widths, "profile_cores": cores}


def parts(timeline, left, right):
    compute = timeline["compute"].clip(left, right)
    copies = timeline["copy"].clip(left, right)
    event = timeline["device_event_wait"].clip(left, right)
    c = interval_union(compute)
    cd = interval_union(compute + copies)
    all_time = interval_union(compute + copies + event)
    return dict(zip(FIELDS, (c, cd-c, all_time-cd, right-left-all_time), strict=True))


def projection(previous, phases, key, fraction):
    zero = max(previous)
    offsets = [zero-v for v in previous]
    completed = zero
    for prep, post_join, new_offsets, work in phases:
        values = [value-offset - fraction*classified.get(key, 0)
                  for value, offset, classified in zip(prep, offsets, work, strict=True)]
        completed += max(values) + post_join*(1-fraction if key == "collective_after_last" else 1)
        offsets = new_offsets
    return (completed-zero)/MS


def compute_rows(joined, timelines, events, *, include_boundaries=False, resource_costs=None):
    for batch in joined:
        window_start = max(s["previous_allgather_end_ns"] for s in batch)
        period = max(s["allgather_end_ns"] for s in batch)-window_start
        prev = [s["previous_allgather_end_ns"] for s in batch]
        origin = list(prev)
        totals = Counter()
        phases = []
        boundaries = []
        extra_components = set()
        marked = defaultdict(list)
        for i, group in enumerate(zip(*(s["ops"] for s in batch), strict=True)):
            arrivals = [entry[0] for entry in group]
            finishes = [a+entry[1] for a, entry in zip(arrivals, group, strict=True)]
            latest = max(range(len(group)), key=lambda r: arrivals[r])
            gate, done = max(prev), max(finishes)
            arrive = max(arrivals)
            if any(a<b for a,b in zip(arrivals,prev,strict=True)) or arrive < gate:
                raise ValueError("Unexpected overlap across TP reduction boundaries")
            costs = [parts(timeline,a,b) for timeline,a,b in zip(timelines,prev,arrivals,strict=True)]
            if resource_costs is not None:
                for rank, (a, b) in enumerate(zip(prev, arrivals, strict=True)):
                    extras = resource_costs(rank, batch[rank], i, a, b)
                    costs[rank].update(extras)
                    extra_components.update(extras)
            serial = parts(timelines[latest],gate,arrive)
            totals.update(serial)
            totals["collective_after_last"] += done-arrive
            if include_boundaries:
                boundaries.append((gate, arrive, done, latest))
            if i <= 1:
                marker = "eager_submit_ns" if i == 0 else "graph_submit_ns"
                label = "first" if i == 0 else "second"
                s = batch[latest]
                cutoff = s[marker]
                totals[label+"_prejoin_duration"] += arrive-gate
                totals[label+"_prejoin_uncovered"] += serial["uncovered"]
                marked[label+"_arrival_spread_ms"].append((arrive-min(arrivals))/MS)
                marked[label+"_addressable_join_completion_ms"].append((done-arrive)/MS)
                if cutoff is not None and gate <= cutoff <= arrive:
                    totals[label+"_uncovered_before_host_submit"] += parts(timelines[latest],gate,cutoff)["uncovered"]
                    marked[label+"_gate_to_host_submit_ms"].append((cutoff-gate)/MS)
                    marked[label+"_host_submit_to_device_arrival_ms"].append((arrive-cutoff)/MS)
                    foreground = [e for name in (EVENT_SYNC,STREAM_SYNC)
                                  for e in events[latest].get(name, []) if e[2]==s["graph_tid"] and e[0]<cutoff and e[1]>gate]
                    marked[label+"_explicit_host_sync_ms"].append(interval_union(
                        (max(a,gate),min(b,cutoff)) for a,b,*_ in foreground)/MS)
                if i==0 and s["sample_end_ns"] is not None:
                    marked["prior_logits_to_latest_rank_sampler_finish_ms"].append((s["sample_end_ns"]-gate)/MS)
                    marked["latest_rank_sampler_finish_to_eager_collective_ms"].append((arrive-s["sample_end_ns"])/MS)
                for rank,(p,arrival) in enumerate(zip(prev,arrivals,strict=True)):
                    cutoff = batch[rank][marker]
                    if cutoff is not None and p <= cutoff <= arrival:
                        costs[rank]["submission_gap"] = parts(timelines[rank],p,cutoff)["uncovered"]
            offsets = [done-end for end in finishes]
            phases.append(([a-b for a,b in zip(arrivals,prev,strict=True)],done-arrive,offsets,costs))
            prev = finishes

        counted = sum(totals[key] for key in (*FIELDS,"collective_after_last"))
        if counted != period or abs(projection(origin,phases,"compute",0)-period/MS)>1e-8:
            raise ValueError("The Decode step does not reconcile with collective timestamps")
        result = {key+"_ms": value/MS for key,value in totals.items()}
        result["period_ms"] = period/MS
        result["last_collectives_and_body_ms"] = (period-totals["first_prejoin_duration"]-totals["second_prejoin_duration"])/MS
        result["allgather_anchor_spread_ms"] = (max(origin)-min(origin))/MS
        for key, values in marked.items():
            result[key] = values[0]
        for key in ("compute", "collective_after_last", "submission_gap", *sorted(extra_components)):
            for fraction in (0.1, 0.2, 0.3):
                result[f"save_{key}_{fraction:g}_ms"] = period/MS-projection(origin,phases,key,fraction)
        if include_boundaries:
            result["collective_boundaries_ns"] = boundaries
        yield result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ranks", type=Path, nargs=4)
    p.add_argument("output", type=Path)
    args=p.parse_args()
    with ThreadPoolExecutor(4) as pool:
        steps, timelines, events, sources = zip(*pool.map(load,args.ranks),strict=True)
    aligned = list(aligned_steps(steps))
    summary = list(compute_rows(aligned,timelines,events))
    if len(summary)<10:
        raise ValueError("Too few aligned steps")
    values = defaultdict(list)
    for row in summary:
        for key, value in row.items():
            values[key].append(value)
    write_json(args.output,{"sources":sources,"aligned_steps":len(summary),
                            "coverage":{key:distribution(rows) for key,rows in values.items()},"steps":summary})


if __name__ == "__main__":
    main()
