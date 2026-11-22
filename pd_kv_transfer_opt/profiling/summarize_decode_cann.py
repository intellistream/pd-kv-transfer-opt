"""Summarize native Decode kernel captures, with a compact rank-0 timeline."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import re
from statistics import median

from pd_kv_transfer_opt.common import read_jsonl, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution, interval_union
from pd_kv_transfer_opt.profiling.export_cann import TIME_KEYS, time_ns


WORKER_RANK = re.compile(r"_rank(\d+)_\d+_.*_ascend_pt$")
MS = 1_000_000


def kernel_family(name, kind, core):
    words = re.sub(r"[^a-z0-9]", "", (name + kind).lower())
    if re.search(r"batchget|batchput|rdma", words):
        return "kv_transfer"
    if core == "COMMUNICATION" or re.search(r"hccl|allreduce|allgather|alltoall|reducescatter", words):
        return "collective"
    if re.search(r"memcpy|memset|h2d|d2h|d2d", words):
        return "copy"
    if "attention" in words or "pagedattn" in words:
        return "attention"
    if re.search(r"gateddelta|recurrent|causalconv", words):
        return "recurrent_or_convolution"
    if re.search(r"matmul|gemm|mmv2", words):
        return "matrix_multiply"
    if re.search(r"moe|topk|routing", words):
        return "moe_routing_or_selection"
    return "other"


def collect(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if not all(row.get(key) and row[key] != "N/A" for key in TIME_KEYS):
                continue
            start, duration = (time_ns(row[key]) for key in TIME_KEYS)
            if duration <= 0:
                continue
            name = row["Name"]
            kind = row.get("Type") or re.sub(r"_+\d.*$", "", name)
            core = row["Accelerator Core"]
            yield start, duration, kind, core, kernel_family(name, kind, core)


def summarize(kernels):
    families = defaultdict(list)
    cores = defaultdict(list)
    names = Counter()
    for start, duration, name, core, family in kernels:
        interval = (start, start + duration)
        families[family].append(interval)
        cores[core].append(interval)
        names[name] += duration
    first = min(row[0] for row in kernels)
    last = max(start + duration for start, duration, *_ in kernels)

    def costs(groups):
        return {key: {"tasks": len(spans),
                      "sum_ms": sum(right - left for left, right in spans) / MS,
                      "union_ms": interval_union(spans) / MS}
                for key, spans in sorted(groups.items())}

    return {"tasks": len(kernels), "observed_span_ms": (last - first) / MS,
            "device_task_union_ms": interval_union((row[0], row[0] + row[1]) for row in kernels) / MS,
            "families": costs(families), "cores": costs(cores),
            "types_by_summed_ms": [{"type": name, "sum_ms": duration / MS} for name, duration in names.most_common(24)]}


def capture_name(kernels, windows, clock_offset):
    first = min(row[0] for row in kernels) - clock_offset
    last = max(start + duration for start, duration, *_ in kernels) - clock_offset
    matching = [row for row in windows if row["request_ns"] - 200 * MS <= first
                and last <= row["stopped_ns"] + 200 * MS]
    if len(matching) != 1:
        return None, first, last
    return matching[0]["window"], first, last


def analyze_collectives(path):
    if not path.is_file():
        return {}
    captures = json.loads(path.read_bytes())
    by_op = defaultdict(list)
    for phase in captures.values():
        for name, row in phase.get("collective", {}).items():
            name = name.lower()
            operation = "allreduce" if "allreduce" in name else "allgather" if "allgather" in name else None
            if operation is not None:
                by_op[operation].append(row)
    records = sorted(((row["Communication Time Info"]["Start Timestamp(us)"], kind, row["Communication Time Info"])
                      for kind, rows in by_op.items() for row in rows), key=lambda row: row[0])

    def summarize_details(timings):
        return {key: {"sum": sum(row[key] for row in timings),
                      **distribution(row[key] for row in timings)}
                for key in ("Elapse Time(ms)", "Synchronization Time(ms)", "Transit Time(ms)")}

    groups, current = [], []
    for _, kind, timing in records:
        if kind == "allreduce":
            current.append(timing)
        else:
            groups.append(current)
            current = []
    frequencies = Counter(map(len, groups))
    depth = frequencies.most_common(1)[0][0] if frequencies else 0
    complete = [group for group in groups if len(group) == depth]
    by_position = [summarize_details(position) for position in zip(*complete)]
    result = {"source_sha256": sha256(path), "allreduces_per_allgather_window": dict(frequencies),
              "allreduce_by_position_from_first_in_step": by_position}
    for kind, rows in by_op.items():
        sizes = defaultdict(list)
        for row in rows:
            for key, values in row["Communication Bandwidth Info"].items():
                if isinstance(values, dict) and values.get("Transit Size(MB)", 0) > 0:
                    sizes[key].append(values["Transit Size(MB)"])
        result[kind] = {"calls": len(rows),
                        "cost_ms": summarize_details([row["Communication Time Info"] for row in rows]),
                        "reported_positive_transit_size_MB": {
                            key: {"sum": sum(values), **distribution(values)} for key, values in sizes.items()}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    telemetry = read_jsonl(args.case_dir / "telemetry.jsonl")
    offset = int(median(row["unix_ns"] - row["monotonic_ns"] for row in telemetry))
    root = args.run_dir / "native-process-profile"
    archive = args.output_dir / "r34-cann-rank0.jsonl.gz"
    args.output_dir.mkdir(exist_ok=True, parents=True)
    report = {}
    with gzip.open(archive, "wt", encoding="utf-8", compresslevel=5) as stream:
        for role in ("D0", "D1"):
            window_file = args.case_dir / f"profile-windows-{role}.jsonl"
            windows = list(read_jsonl(window_file)) if window_file.is_file() else []
            inventory = []
            native_ranks = Counter()
            for capture in sorted((root / role).glob("*_ascend_pt")):
                rank = WORKER_RANK.search(capture.name)
                if rank is None:
                    continue
                rank = int(rank[1])
                native_ranks[rank] += 1
                if rank != 0:
                    continue
                path = capture / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
                if not path.is_file():
                    from torch_npu.profiler.profiler import analyse

                    analyse(str(capture), max_process_number=2)
                kernels = list(collect(path))
                if not kernels:
                    inventory.append({"rank": rank, "status": "no_timed_tasks"})
                    continue
                window, start, end = capture_name(kernels, windows, offset)
                record = {"rank": rank, "window": window, "first_task_monotonic_ns": start,
                          "last_task_monotonic_ns": end, "kernel_csv_sha256": sha256(path), **summarize(kernels)}
                record["collective_profile"] = analyze_collectives(path.parent / "communication.json")
                inventory.append(record)
                if window is not None:
                    t0 = min(row[0] for row in kernels)
                    for time, duration, name, core, family in kernels:
                        stream.write(json.dumps([role, window, time - t0, duration, name, core, family],
                                                separators=(",", ":")) + "\n")
            report[role] = {"trigger_windows": windows, "captures": inventory,
                            "native_capture_count_by_rank": dict(native_ranks)}
    write_json(args.output_dir / "cann-summary.json", {"roles": report, "rank0_timeline": {
        "fields": ["instance", "window", "relative_start_ns", "duration_ns", "type", "core", "family"],
        "sha256": sha256(archive), "bytes": archive.stat().st_size}})


if __name__ == "__main__":
    main()
