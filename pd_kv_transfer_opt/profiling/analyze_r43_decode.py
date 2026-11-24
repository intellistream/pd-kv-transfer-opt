"""Join native D batches to four-rank CANN captures and audit old profiling archives on CPU."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from math import isclose, isfinite
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_decode_critical_step import (
    EVENT_SYNC, HOST_DEVICE_SYNC, MS, STREAM_SYNC, Coverage, compute_rows, load,
)
from pd_kv_transfer_opt.profiling.analyze_decode_graph_input import aligned_steps
from pd_kv_transfer_opt.profiling.analyze_decode_steps import sampled_steps, summarize
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution, interval_union
from pd_kv_transfer_opt.profiling.export_cann import time_ns
from pd_kv_transfer_opt.profiling.profile_closed_sessions import read_trace
from pd_kv_transfer_opt.profiling.r43_decode_capture import CANN_FILES, CONFIG_DIR, RANK_IN_NAME, target_band
from pd_kv_transfer_opt.profiling.summarize_decode_cann import kernel_family
from pd_kv_transfer_opt.simulator.decode_stage_budget import historical_priors, verified_json


SYNC_CALLS = (EVENT_SYNC, STREAM_SYNC, HOST_DEVICE_SYNC)
PMU_CATEGORIES = ("arithmetic_candidate", "memory_candidate", "unclassified")
LOG_PREFIX = "ISSUE23_OBSERVATION "


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) and value >= 0 else None


def pmu_category(row):
    memory_bound = number(row.get("memory_bound"))
    if memory_bound is None or memory_bound == 0:
        tags = set()
        for prefix in ("aic_", "aiv_", ""):
            arithmetic = max(number(row.get(prefix + field)) or 0. for field in ("mac_ratio", "vec_ratio"))
            read = number(row.get(prefix + "mte2_ratio"))
            if arithmetic > 0 and read is not None:
                tags.add("memory_candidate" if read > arithmetic else "arithmetic_candidate"
                         if read < arithmetic else "unclassified")
        return tags.pop() if len(tags) == 1 else "unclassified"
    if memory_bound > 1:
        return "memory_candidate"
    return "arithmetic_candidate" if memory_bound < 1 else "unclassified"


def kernel_evidence(path):
    pmu, families = defaultdict(list), defaultdict(list)
    other_collectives = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {key.strip().lower(): value for key, value in raw.items()}
            a, duration = row["start time(us)"], row["duration(us)"]
            if not a or a == "N/A" or not duration or duration == "N/A":
                continue
            start, length = time_ns(a), time_ns(duration)
            if length <= 0:
                continue
            core, kind = row["accelerator core"], row["type"]
            window = (start, start + length)
            if core == "COMMUNICATION":
                if not any(kind_part in kind.lower() for kind_part in ("allreduce", "allgather")):
                    other_collectives.append(window)
                continue
            pmu[pmu_category(row)].append(window)
            families[kernel_family(row.get("name", kind), kind, core)].append(window)
    return ({name: Coverage(pmu[name]) for name in PMU_CATEGORIES},
            {name: Coverage(rows) for name, rows in families.items()}, Coverage(other_collectives))


def exclusive_kernels(classes, left, right):
    arithmetic, memory, unknown = (classes[key].clip(left, right) for key in PMU_CATEGORIES)
    device = interval_union(arithmetic + memory + unknown)
    a = device - interval_union(memory + unknown)
    m = device - interval_union(arithmetic + unknown)
    return {"arithmetic_candidate_ns": a, "memory_candidate_ns": m, "mixed_or_unknown_kernel_ns": device - a - m}


def exposed_host_wait(runtime, timeline, step, index, gate, arrival):
    """Count only visible blocking calls on the submitting thread before the join."""
    marker = step["eager_submit_ns"] if index == 0 else step["graph_submit_ns"]
    thread = step["graph_tid"]
    if index > 1 or thread is None or marker is None or not gate < marker <= arrival:
        return 0
    waits = [(max(gate, start), end) for kind in SYNC_CALLS for start, end, tid, _ in runtime.get(kind, ())
             if tid == thread and start < end <= marker and end > gate]
    covered = [span for name in ("compute", "copy", "device_event_wait")
               for span in timeline[name].clip(gate, marker)]
    return interval_union(covered + waits) - interval_union(covered)


def with_resources(row, ranks, timelines, events, kernels):
    boundaries = row.pop("collective_boundaries_ns")
    start, end = boundaries[0][0], boundaries[-1][2]
    if any(other.clip(start, end) for _, _, other in kernels):
        raise ValueError("An unsupported collective overlaps this step")
    resources, families = Counter(), Counter()
    for index, (gate, arrival, _, latest) in enumerate(boundaries):
        by_pmu, by_family, _ = kernels[latest]
        resources.update(exclusive_kernels(by_pmu, gate, arrival))
        resources["explicit_host_sync_ns"] += exposed_host_wait(events[latest], timelines[latest],
                                                                ranks[latest], index, gate, arrival)
        for name, coverage in by_family.items():
            families[name] += interval_union(coverage.clip(gate, arrival))
    residual = row["uncovered_ms"] - resources["explicit_host_sync_ns"] / MS
    if residual < -1e-9 or not isclose(sum(resources[key] for key in
                                         ("arithmetic_candidate_ns", "memory_candidate_ns", "mixed_or_unknown_kernel_ns")),
                                     row["compute_ms"] * MS, abs_tol=.01):
        raise ValueError("Disjoint PMU or host intervals do not reconcile with the four-rank step")
    row.update({name.removesuffix("_ns") + "_ms": duration / MS for name, duration in resources.items()})
    row["unknown_after_visible_host_ms"] = max(0., residual)
    row["kernel_family_union_ms"] = {name: length / MS for name, length in sorted(families.items()) if length}
    return row


def eligible(step, prompts):
    return (step["mode"] == "FULL" and step["replayed_all_ranks"]
            and all(key in prompts and offset is not None and offset >= prompts[key] and count == 1
                    for key, offset, count in step["rows"]))


def profile_groups(rows):
    output = {}
    for band in ("1-4", "9-16"):
        values = [row for row in rows if target_band(row["width"]) == band]
        numeric = defaultdict(list)
        for row in values:
            for name, value in row.items():
                if name.endswith("_ms") and isinstance(value, (int, float)):
                    numeric[name].append(value)
        output[band] = {"steps": len(values), "exact_widths": dict(Counter(row["width"] for row in values)),
                        "padded_widths": dict(Counter(row["padded_tokens"] for row in values)),
                        "spans": {key: distribution(items) for key, items in sorted(numeric.items())}}
    return output


def native_cadence(steps, windows, prompts):
    outside = [row for row in steps if eligible(row, prompts) and not any(
        window["role"] == row["instance"] and row["worker_enter_ns"] < window.get("stopped_ns", float("inf"))
        and row["update_end_ns"] > window.get("request_ns", 0) for window in windows)]
    return {band: summarize([row for row in outside if target_band(len(row["rows"])) == band])
            for band in ("1-4", "9-16")}


def analyze_window(folders, steps, prompts):
    with ThreadPoolExecutor(4) as pool:
        ranks, timelines, events, sources = zip(*pool.map(lambda path: load(path, with_batch_tags=True), folders),
                                                strict=True)
    kernels = [kernel_evidence(path / "kernel_details.csv") for path in folders]

    def resource_costs(rank, step, phase, start, end):
        pmu = exclusive_kernels(kernels[rank][0], start, end)
        return {"arithmetic_candidate": pmu["arithmetic_candidate_ns"],
                "memory_candidate": pmu["memory_candidate_ns"],
                "explicit_host_sync": exposed_host_wait(events[rank], timelines[rank], step, phase, start, end)}

    tagged, rows, rejected = set(), [], Counter()
    for aligned in aligned_steps(ranks):
        ids = {part.get("batch") for part in aligned}
        if len(ids) != 1 or None in ids:
            rejected["no_common_batch_marker"] += 1
            continue
        batch, = ids
        if batch in tagged:
            raise ValueError(f"Ambiguous four-rank mapping for batch {batch}")
        tagged.add(batch)
        step = steps.get(batch)
        if step is None or not eligible(step, prompts):
            rejected["missing_four_rank_worker_join_or_not_pure_FULL_decode"] += 1
            continue
        if any(len(part["ops"]) != 82 for part in aligned):
            rejected["unexpected_collective_signature"] += 1
            continue
        try:
            row, = compute_rows((aligned,), timelines, events, include_boundaries=True, resource_costs=resource_costs)
            row = with_resources(row, aligned, timelines, events, kernels)
        except ValueError as error:
            rejected[str(error)] += 1
            continue
        history = max(offset for _, offset, _ in step["rows"])
        rows.append({"batch": batch, "width": len(step["rows"]), "padded_tokens": step["padded_tokens"],
                     "max_context_tokens": history, "request_ids": [key for key, *_ in step["rows"]],
                     "sampled_worker_period_ms": step["cost"]["rank0_worker_gap_ms"], **row})
    return {"sources": sources, "rejected_steps": dict(rejected), "stats": profile_groups(rows), "steps": rows}


def read_step_logs(log_path, role):
    events = []
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if LOG_PREFIX in line:
                try:
                    event = json.loads(line.split(LOG_PREFIX, 1)[1])
                except json.JSONDecodeError:
                    continue
                events.append({**event, "instance": role})
    return sampled_steps(events)


def raw_folders(run_dir, ranks, parse):
    folders = []
    for rank in range(4):
        raw_path = (run_dir / ranks[str(rank)]).resolve()
        name = RANK_IN_NAME.search(raw_path.name)
        if not raw_path.is_relative_to(run_dir) or name is None or int(name[1]) != rank:
            raise ValueError("Rank profile path escaped the run directory or did not match its rank")
        path = raw_path / "ASCEND_PROFILER_OUTPUT"
        if parse and any(not (path / name).is_file() for name in CANN_FILES):
            from torch_npu.profiler.profiler import analyse
            analyse(str(raw_path), max_process_number=2)
        if any(not (path / name).is_file() for name in CANN_FILES):
            raise FileNotFoundError(f"Missing CANN outputs for rank {rank}: {path}")
        folders.append(path)
    return folders


def captured_report(run_dir, receipt_path, workload_path, parse):
    receipt, workload = read_json(receipt_path), read_json(workload_path)
    if (workload != read_json(CONFIG_DIR / "workload-r42-natural-sustained.json")
            or workload["workload_sha256"] != receipt["workload_sha256"]):
        raise ValueError("Capture receipt and r43 workload manifest disagree")
    if read_json(run_dir / "state.json")["run_id"] != receipt["run_id"]:
        raise ValueError("Capture receipt does not refer to this run directory")
    prompts = {row["id"]: row["prompt_tokens"] for row in workload["request_sources"]}
    by_role = {role: read_step_logs(run_dir / "logs" / f"{role}.log", role)
               for role in sorted({item["role"] for item in receipt["windows"]})}
    windows = []
    for window in receipt["windows"]:
        role = window["role"]
        result = {"role": role, "target_width": window["target_width"]}
        if len(window.get("ranks", {})) != 4:
            windows.append({**result, "error": "missing_rank_capture"})
            continue
        steps = {row["batch"]: row for row in by_role[role][0]}
        try:
            folders = raw_folders(run_dir, window["ranks"], parse)
            windows.append({**result, **analyze_window(folders, steps, prompts)})
        except (OSError, ValueError) as error:
            windows.append({**result, "error": str(error)})
    ready = [[window["role"], window["target_width"]] for window in windows
             if window.get("stats", {}).get(window["target_width"], {}).get("steps")]
    native_pace = {role: {"join_counts": coverage, "outside_CANN": native_cadence(steps, receipt["windows"], prompts)}
                   for role, (steps, coverage) in by_role.items()}
    return {"receipt_sha256": sha256(receipt_path), "run_id": receipt["run_id"], "windows": windows,
            "sampled_targets_with_confirmed_steps": ready, "native_phase_cadence": native_pace,
            "scope": "Four equal batch markers and native FULL one-token rows are required. PMU classes tag "
                     "disjoint kernel wall intervals, with mixed or missing counters kept unknown. HCCL starts "
                     "at last-rank arrival and visible host sync only counts outside covered device work. "
                     "Single-class local projections recompute each join's latest rank using the measured "
                     "rank intervals and frozen offsets. "
                     "The receipt is diagnostic. Formal throughput is measured in a separate unprofiled run."}


def archive_report(results):
    priors = historical_priors(results)
    r34 = results / "2026-11-20-r34-native-decode-path"
    manifest = read_json(r34 / "manifest.json")
    trace = r34 / "r34-events.jsonl.gz"
    digest = sha256(trace)
    if digest != manifest["files"][trace.name]["sha256"]:
        raise ValueError("r34 event digest changed")
    prompts = {row["id"]: row["prompt_tokens"] for row in manifest["workload"]["request_sources"]}
    arm, = read_trace(trace).values()
    steps, coverage = sampled_steps(arm["d_native"])
    eligible_steps = [step for step in steps if eligible(step, prompts)]
    cann_summary, cann_hash = verified_json(r34, "cann-summary.json", manifest)
    family_screen = []
    for role, report in cann_summary["roles"].items():
        for capture in report["captures"]:
            count = capture["collective_profile"]["allgather"]["calls"]
            family_screen.append({"role": role, "window": capture["window"], "allgathers": count,
                                  "rank0_family_union_ms_per_allgather": {
                                      name: data["union_ms"] / count for name, data in capture["families"].items()
                                  }})
    old = results / "2026-11-20-r37-decode-critical-step"
    old_manifest = read_json(old / "manifest.json")
    captures = {}
    for source, report in priors["captures"].items():
        payload, _ = verified_json(old, f"{source}-breakdown.json", old_manifest)
        captures[source] = {**report, "HCCL_first_two_after_last_ms": distribution(
            row["first_addressable_join_completion_ms"] + row["second_addressable_join_completion_ms"]
            for row in payload["steps"]),
            "explicit_host_sync_first_two_ms": distribution(
                row["first_explicit_host_sync_ms"] + row["second_explicit_host_sync_ms"] for row in payload["steps"]
                if "first_explicit_host_sync_ms" in row and "second_explicit_host_sync_ms" in row),
            "device_copy_ms": payload["coverage"]["copy_ms"], "PMU_kernel_classification": None}
    return {"r34_event_sha256": digest, "native_step_join": coverage, "r34_baseline": priors["r34_baseline"],
            "r34_pure_FULL_bands": {band: summarize(
                [row for row in eligible_steps if target_band(len(row["rows"])) == band]) for band in ("1-4", "9-16")},
            "r34_outside_CANN": native_cadence(steps, arm["cann_windows"], prompts),
            "r34_rank0_kernel_screen": {"cann_summary_sha256": cann_hash, "windows": family_screen},
            "r34_and_r37_critical_steps": captures,
            "scope": "Replay uses the r34 native event ledger and the two verified derived four-rank breakdowns. "
                     "The archived raw per-rank CANN output and device PMU columns are unavailable here."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    archived = modes.add_parser("archive")
    archived.add_argument("--results", type=Path, default=Path("results"))
    archived.add_argument("--output", type=Path, required=True)
    live = modes.add_parser("capture")
    live.add_argument("--run-dir", type=Path, required=True)
    live.add_argument("--receipt", type=Path, required=True)
    live.add_argument("--workload-manifest", type=Path, required=True)
    live.add_argument("--parse", action="store_true", help="Parse missing CANN exports in the existing Ascend runtime.")
    live.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = (archive_report(args.results) if args.mode == "archive"
              else captured_report(args.run_dir.resolve(), args.receipt, args.workload_manifest, args.parse))
    write_json(args.output, report)


if __name__ == "__main__":
    main()
