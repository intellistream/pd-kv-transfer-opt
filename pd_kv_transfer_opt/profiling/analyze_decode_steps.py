"""Join sampled D scheduler and device steps to native long-session drain."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_kv_lifetime import change_finish
from pd_kv_transfer_opt.profiling.analyze_native_prefill_path import decode_progress
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, measure_arms, read_trace, requests_for


MS = NS / 1000
RANKS = {0, 1, 2, 3}


def sampled_steps(events):
    dispatches, results = {}, {}
    workers = defaultdict(dict)
    for row in events:
        kind = row["type"]
        if kind not in ("d_step_dispatch", "d_step_result", "d_step_worker"):
            continue
        key = row["instance"], row["batch"]
        if kind == "d_step_worker":
            rank = row["rank"]
            if rank in workers[key]:
                raise ValueError(f"Duplicate timed worker {key}, rank {rank}")
            workers[key][rank] = row
        else:
            target = dispatches if kind == "d_step_dispatch" else results
            if key in target:
                raise ValueError(f"Duplicate sampled batch {key}")
            target[key] = row

    def milliseconds(start, end):
        if start is None:
            return None
        if end < start:
            raise ValueError("Same-host decode step clocks are out of order")
        return (end - start) / MS

    spans = []
    for key, dispatch in dispatches.items():
        if key not in results or set(workers[key]) != RANKS:
            continue
        rank0 = workers[key][0]
        device = [row.get("forward_device_ms") for row in workers[key].values()]
        if None in device or "sample_return_ns" not in rank0:
            continue
        result = results[key]
        rank_forward_entries = [row["forward_enter_ns"] for row in workers[key].values()]
        rank_worker_entries = [row["worker_enter_ns"] for row in workers[key].values()]
        costs = {
            "engine_dispatch_gap_ms": milliseconds(dispatch["previous_dispatch_ns"], dispatch["dispatch_ns"]),
            "scheduler_compute_ms": milliseconds(dispatch["schedule_begin_ns"], dispatch["dispatch_ns"]),
            "RPC_dispatch_to_rank0_enter_ms": milliseconds(dispatch["dispatch_ns"], rank0["worker_enter_ns"]),
            "rank0_worker_gap_ms": milliseconds(rank0["previous_worker_enter_ns"], rank0["worker_enter_ns"]),
            "rank0_input_prepare_ms": milliseconds(rank0["worker_enter_ns"], rank0["forward_enter_ns"]),
            "worker_entry_spread_across_ranks_ms": milliseconds(min(rank_worker_entries), max(rank_worker_entries)),
            "forward_entry_spread_across_ranks_ms": milliseconds(min(rank_forward_entries), max(rank_forward_entries)),
            "rank0_forward_host_ms": milliseconds(rank0["forward_enter_ns"], rank0["forward_return_ns"]),
            "rank0_forward_device_ms": rank0["forward_device_ms"],
            "slowest_rank_forward_device_ms": max(device),
            "rank0_after_forward_stream_ms": rank0["after_forward_stream_ms"],
            "rank0_sample_host_ms": milliseconds(rank0["sample_enter_ns"], rank0["sample_return_ns"]),
            "rank0_sample_return_to_engine_result_ms": milliseconds(rank0["sample_return_ns"], result["result_ns"]),
            "engine_update_ms": milliseconds(result["result_ns"], result["update_end_ns"]),
            "dispatch_to_update_ms": milliseconds(dispatch["dispatch_ns"], result["update_end_ns"]),
        }
        spans.append({"instance": key[0], "batch": key[1], "rows": dispatch["rows"],
                      "worker_enter_ns": rank0["worker_enter_ns"], "update_end_ns": result["update_end_ns"],
                      "mode": rank0["graph_mode"], "replayed_all_ranks": all(
                          row["graph_replays"] > 0 for row in workers[key].values()),
                      "padded_tokens": rank0["padded_tokens"], "cost": costs})
    coverage = {"sampled_dispatches": len(dispatches), "sampled_results": len(results),
                "worker_events": sum(map(len, workers.values())),
                "four_rank_joins": sum(
                    key in results and set(workers[key]) == RANKS for key in dispatches),
                "all_four_with_device_timing": len(spans),
                "unmatched_results": len(results.keys() - dispatches.keys()),
                "unmatched_worker_batches": len(workers.keys() - dispatches.keys()),
                "timing_unavailable_events": sum(row["type"] == "d_step_device_timing_unavailable" for row in events)}
    return spans, coverage


def summarize(steps):
    fields = defaultdict(list)
    for step in steps:
        for key, value in step["cost"].items():
            if value is not None:
                fields[key].append(value)
    return {"steps": len(steps), "modes": dict(Counter(row["mode"] for row in steps)),
            "padded_tokens": dict(Counter(str(row["padded_tokens"]) for row in steps)),
            "graph_replayed_all_ranks": sum(row["replayed_all_ranks"] for row in steps),
            "stage_cost_ms": {key: distribution(values) for key, values in sorted(fields.items())}}


def verify_terminal_cadence(steps):
    steps = sorted(steps, key=lambda row: row["worker_enter_ns"])
    if len(steps) < 8:
        return None
    boundary = len(steps) // 2
    training, held_out = steps[:boundary], steps[boundary:]
    cadence = median(row["cost"]["rank0_worker_gap_ms"] for row in training)
    start, end = held_out[0], held_out[-1]
    tok = end["rows"][0][1] - start["rows"][0][1]
    if tok <= 0:
        return None
    held_out_tpot = (end["worker_enter_ns"] - start["worker_enter_ns"]) / MS / tok
    return {"first_half_samples": len(training), "held_out_samples": len(held_out), "held_out_tokens": tok,
            "first_half_worker_gap_ms_p50": cadence, "held_out_elapsed_per_token_ms": held_out_tpot,
            "relative_error_percent": 100 * (cadence / held_out_tpot - 1)}


def stepped_report(arm, manifest):
    origin, = (row["monotonic_s"] for row in arm["client"] if row["type"] == "client_origin")
    workload = manifest["workload"]["request_sources"]
    profiles = requests_for(arm, workload, origin)
    prompts = {row["id"]: row["prompt_tokens"] for row in workload}
    d_stages, supply = decode_progress(arm, profiles, origin)
    baseline = measure_arms(arm, profiles, origin, 100, {"P": 16, "D": 16})
    steps, coverage = sampled_steps(arm["d_native"])
    if not steps:
        raise ValueError("No complete four-rank decode step measurements")

    groups = defaultdict(list)
    normal_steps = []
    capture_steps = defaultdict(list)
    for step in steps:
        start, end = step["worker_enter_ns"], step["update_end_ns"]
        overlapping = [window for window in arm["cann_windows"] if step["instance"] == window["role"]
                       and start < window["stopped_ns"] and end > window["request_ns"]]
        if overlapping:
            for window in overlapping:
                capture_steps[f"{window['role']}/{window['window']}"].append(step)
            continue
        if not all(profiles[key]["D"] == step["instance"] and offset is not None and offset >= prompts[key]
                   and d_stages[key]["D_KV_ready_s"] <= start / NS - origin for key, offset, _ in step["rows"]):
            continue
        normal_steps.append(step)
        history = max(offset for _, offset, _ in step["rows"])
        history_window = "0-32K" if history < 32768 else "32K-64K" if history < 65536 else "64K+"
        groups[f"{len(step['rows'])}/{history_window}"].append(step)

    terminal = {}
    adjusted = {}
    for role, timeline in supply.items():
        key = timeline["last_request"]
        req = profiles[key]
        solo = timeline["tail_alone_after_first_token_s"]
        effective_begin = req["end_s"] - solo
        attributed = [step for step in normal_steps if len(step["rows"]) == 1
                      and step["rows"][0][0] == key
                      and step["worker_enter_ns"] / NS - origin >= effective_begin]
        duration = summarize(attributed)
        costs = duration["stage_cost_ms"]
        forward = costs.get("slowest_rank_forward_device_ms", {})
        cadence = costs.get("rank0_worker_gap_ms", {}).get("p50")
        ordered = sorted(attributed, key=lambda row: row["worker_enter_ns"])
        fraction = typical_fraction = None
        if len(ordered) >= 2 and cadence:
            first, last = ordered[0], ordered[-1]
            tokens = last["rows"][0][1] - first["rows"][0][1]
            elapsed_ms = (last["worker_enter_ns"] - first["worker_enter_ns"]) / MS
            if tokens > 0 and elapsed_ms > 0:
                fraction = min(1.0, forward["mean"] * tokens / elapsed_ms)
            typical_fraction = min(1.0, forward["p50"] / cadence)
        if fraction is not None:
            adjusted[key] = solo * fraction * 0.1
        terminal[role] = {**timeline, "output_tokens": req["output_tokens"], "first_s": req["first_s"],
                          "end_s": req["end_s"], "effective_TPOT_ms": req["effective_TPOT_ms"],
                          "estimated_device_forward_share": fraction,
                          "median_forward_to_worker_gap_ratio": typical_fraction, "solo_after_first_samples": duration,
                          "solo_cadence_validation": verify_terminal_cadence(attributed)}

    method = {"coverage": coverage, "uncaptured_pure_decode_steps": len(normal_steps),
              "captures": {key: {**summarize(rows), "sampled_widths": dict(Counter(
                  str(len(row["rows"])) for row in rows))} for key, rows in sorted(capture_steps.items())}}
    return {"baseline": baseline, "method": method, "terminal": terminal,
            "step_cost_by_requests_and_history": {key: summarize(rows) for key, rows in sorted(groups.items())},
            "conditional_terminal_forward": {
                "shorten_ten_percent_single_request_body_both_Ds": change_finish(profiles, adjusted, "first_to_end"),
                "shorten_ten_percent_single_request_body_per_D": {
                    role: change_finish(profiles, {row["last_request"]: adjusted[row["last_request"]]}, "first_to_end")
                    for role, row in terminal.items() if row["last_request"] in adjusted},
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    digest = sha256(args.trace)
    if manifest["files"][args.trace.name]["sha256"] != digest:
        raise ValueError("Profile trace does not match the manifest")
    arm, = read_trace(args.trace).values()
    write_json(args.output, {"trace_sha256": digest, **stepped_report(arm, manifest)})


if __name__ == "__main__":
    main()
