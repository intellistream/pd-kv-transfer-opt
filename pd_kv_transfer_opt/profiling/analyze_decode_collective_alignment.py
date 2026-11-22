"""Align the first two AllReduces across NPU ranks in a stable Decode capture."""

import argparse
from bisect import bisect_left
from collections import Counter
import csv
from decimal import Decimal
import json
from pathlib import Path
from statistics import correlation

from pd_kv_transfer_opt.common import sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution, interval_union
from pd_kv_transfer_opt.profiling.export_cann import time_ns


MS = 1_000_000


def cycles(path):
    data = json.loads(path.read_bytes(), parse_float=Decimal)
    events = []
    for phase in data.values():
        for name, row in phase.get("collective", {}).items():
            name = name.lower()
            operation = "allreduce" if "allreduce" in name else "allgather" if "allgather" in name else None
            if operation is None:
                continue
            time = row["Communication Time Info"]
            start = time_ns(time["Start Timestamp(us)"])
            duration = int(time["Elapse Time(ms)"] * MS)
            events.append((start, operation, duration, float(time["Synchronization Time(ms)"])))

    steps, pending = [], []
    for start, operation, duration, sync in sorted(events):
        if operation == "allreduce":
            pending.append((start, duration, sync))
        else:
            steps.append((start + duration, pending))
            pending = []
    widths = Counter(len(group) for _, group in steps)
    width = widths.most_common(1)[0][0]
    return [(anchor, group) for anchor, group in steps if len(group) == width], dict(widths)


def graph_prefix(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [(row["Type"], time_ns(row["Start Time(us)"]), time_ns(row["Duration(us)"]))
                for row in csv.DictReader(stream) if row["Start Time(us)"] not in ("", "N/A")]
    anchors = [i for i, (kind, _, _) in enumerate(rows) if "allgather" in kind.lower()]
    widths = Counter(b - a for a, b in zip(anchors, anchors[1:]))
    width = widths.most_common(1)[0][0]
    if len(rows) != len(anchors) * width or len(widths) != 1:
        raise ValueError(f"Nonuniform graph trace {path.name}")
    result = {}
    for anchor in anchors:
        base = anchor - anchors[0]
        group = rows[base:base + width]
        first, second, *_ = [i for i, (kind, _, _) in enumerate(group) if "allreduce" in kind.lower()]
        _, t0, _ = group[0]
        _, a, service = group[first]
        _, b, _ = group[second]
        _, gather, gather_cost = rows[anchor]
        preparation = interval_union((t, t + length) for _, t, length in group[:first])
        between = interval_union((t, t + length) for _, t, length in group[first + 1:second])
        result[gather + gather_cost] = {
            "first_device_task_ns": t0, "pre_first_allreduce_ms": (a - t0) / MS,
            "pre_first_allreduce_tasks_ms": preparation / MS,
            "pre_first_allreduce_uncovered_ms": ((a - t0) - preparation) / MS,
            "between_first_two_allreduce_ms": (b - a - service) / MS,
            "between_first_two_allreduce_tasks_ms": between / MS,
            "between_first_two_allreduce_uncovered_ms": ((b - a - service) - between) / MS,
        }
    return result


def report(paths):
    rank_rows = [cycles(path) for path in paths]
    steps = [rows for rows, _ in rank_rows]
    anchor_times = [[anchor for anchor, _ in rows] for rows in steps]
    paired = []
    used = set()
    for first in steps[0]:
        t = first[0]
        matches = [first]
        for anchors, rows in zip(anchor_times[1:], steps[1:]):
            insertion = bisect_left(anchors, t)
            nearest = min(anchors[max(0, insertion - 1):insertion + 1], key=lambda a: abs(a - t))
            if abs(nearest - t) > 3 * MS:
                break
            matches.append(rows[bisect_left(anchors, nearest)])
        if len(matches) == len(steps):
            key = tuple(row[0] for row in matches[1:])
            if key in used:
                raise ValueError("Ambiguous device step matching")
            used.add(key)
            paired.append(matches)
    if len(paired) < 2:
        raise ValueError("Insufficient aligned steps")
    detail = {}
    for position in (0, 1):
        observations = []
        for step in paired:
            events = [group[position] for _, group in step]
            starts = [row[0] for row in events]
            finishes = [start + duration for start, duration, _ in events]
            origin = min(starts)
            rows = [{"rank": rank, "entry_ms": (start - origin) / MS,
                     "duration_ms": duration / MS, "synchronization_ms": sync}
                    for rank, (start, duration, sync) in enumerate(events)]
            observations.append({"arrival_spread_ms": (max(starts) - origin) / MS,
                                 "completion_spread_ms": (max(finishes) - min(finishes)) / MS,
                                 "last_arrival_to_last_completion_ms": (max(finishes) - max(starts)) / MS,
                                 "ranks": rows})
        lead = [row["arrival_spread_ms"] - row["ranks"][0]["entry_ms"] for row in observations]
        rank0_sync = [row["ranks"][0]["synchronization_ms"] for row in observations]
        detail[str(position + 1)] = {
            "arrival_spread_ms": distribution(row["arrival_spread_ms"] for row in observations),
            "completion_spread_ms": distribution(row["completion_spread_ms"] for row in observations),
            "last_arrival_to_last_completion_ms": distribution(
                row["last_arrival_to_last_completion_ms"] for row in observations),
            "rank0_lead_vs_sync_correlation": correlation(lead, rank0_sync)
            if len(set(lead)) > 1 and len(set(rank0_sync)) > 1 else None,
            "steps": observations,
        }
    summary = {"aligned_steps": len(paired), "allgather_end_spread_ms": distribution(
            (max(row[0] for row in step) - min(row[0] for row in step)) / MS for step in paired),
            "source_sha256_by_rank": [sha256(path) for path in paths],
            "allreduce_counts_per_step_by_rank": [widths for _, widths in rank_rows],
            "first_allreduces": detail}
    kernel_paths = [path.with_name("kernel_details.csv") for path in paths]
    if all(path.is_file() for path in kernel_paths):
        prefix = [graph_prefix(path) for path in kernel_paths]
        clocks = []
        for rank, entries in enumerate(prefix):
            starts = sorted(entries)
            ranks = []
            for step in paired:
                t = step[rank][0]
                index = bisect_left(starts, t)
                nearest = min(starts[max(0, index - 1):index + 1], key=lambda x: abs(x - t))
                if abs(nearest - t) > 1000:
                    raise ValueError("Kernel and collective exports disagree on AllGather end")
                ranks.append(entries[nearest])
            clocks.append(ranks)
        summary["device_prefix"] = {"source_sha256_by_rank": [sha256(path) for path in kernel_paths],
            "first_task_spread_ms": distribution((max(first) - min(first)) / MS
                for first in zip(*([row["first_device_task_ns"] for row in rank] for rank in clocks))),
            "cost_by_rank": [{name: distribution(row[name] for row in rank)
                             for name in rank[0] if name != "first_device_task_ns"} for rank in clocks]}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank0_to_rank3", type=Path, nargs=4)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, report(args.rank0_to_rank3))


if __name__ == "__main__":
    main()
