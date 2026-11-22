"""Screen sustained RDMA throughput effects without changing a replay's arrival plan."""

from __future__ import annotations

import argparse
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, replace
import gzip
from hashlib import blake2b, sha256 as digest_sha256
from itertools import accumulate
import json
from pathlib import Path
from urllib.request import urlopen

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BYTES
from pd_kv_transfer_opt.simulator.agent_replay import GB, PAGE
from pd_kv_transfer_opt.simulator.agent_trace_gap import read_bytes
from pd_kv_transfer_opt.simulator.terminal_fixed_arrival import TARGET_LOSS


# These published traces are anonymous block metadata. Read the pinned bytes
# as a stream and keep only aggregate screening results.
SOURCES = {
    "bailian_coder": (
        "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/"
        "0b4273a1d8a83935b9d9356e78f95c31f2ac64cf/qwen_coder_blksz_16.jsonl", 16),
    "dynamo_agentic": (
        "https://media.githubusercontent.com/media/ai-dynamo/dynamo/66a95d5aa3b5d238fd94b6f8243c4e479cb7242c/"
        "recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl", 512),
}
AGENTX_URL = ("https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/"
              "resolve/8fecd2fc56694469f758f0afbbb6335ad3043740/traces.jsonl")
CONTEXT_LIMIT = 262_144
MICROSECONDS = 10**6


@dataclass(frozen=True, slots=True)
class Demand:
    input_tokens: int
    output_tokens: int
    at_us: int | None
    prefixes: tuple[bytes, ...]


def read_demand(row, source_block_size):
    prompt, output = row["input_length"], row["output_length"]
    timestamp = row.get("timestamp")
    at = round(timestamp * MICROSECONDS) if timestamp is not None else None
    ids = memoryview(array("q", row["hash_ids"])).cast("B")
    step = PAGE // source_block_size * array("q").itemsize
    sequence = blake2b(digest_size=16)
    complete = min((prompt - 1) // PAGE, len(ids) // step)
    prefixes = []
    for start in range(0, step * complete, step):
        sequence.update(ids[start:start + step])
        prefixes.append(sequence.digest())
    return Demand(prompt, output, at, tuple(prefixes))


def capacities(summary):
    services = [ref["service"] for ref in summary["models"].values()]
    return {
        "R100_GB_s": summary["design"]["effective_100_GB_s_by_reference"],
        "optimistic_decode_tokens_s": max(2 * service["slots"] / (
            service["decode_first_s"] + (service["slots"] - 1) * service["decode_add_s"]) for service in services),
        "optimistic_cold_tokens_s_per_P": max(service["cold_tokens_s"] for service in services),
        "optimistic_warm_tokens_s_per_P": max(service["warm_tokens_s"] for service in services),
        "chunk_tokens": max(service["chunk_tokens"] for service in services),
    }


def fixed_mix(feasibility):
    result = {}
    keep = 1 - TARGET_LOSS
    for name, case in feasibility["cases"].items():
        low, high = (case["arms"][cap] for cap in ("100", "200"))
        rate = low["effective_GB_s"]
        lo, hi = (arm["actual_regular_GB_per_planned_s"] for arm in (low, high))
        low_GB, high_GB = (arm["actual_regular_GET_GB"] for arm in (low, high))
        same_high_read_extra_s = high_GB / (2 * rate)
        saved_low_volume_s = (high_GB - low_GB) / rate
        result[name] = {
            "output_tokens": low["output_tokens"],
            "finish_s": [arm["finish_s"] for arm in (low, high)],
            "throughput_tokens_s": [arm["throughput_tokens_s"] for arm in (low, high)],
            "actual_100_loss": case["100_loss_relative_to_200"],
            "read_GB_by_cap_in_regular_window": [low_GB, high_GB],
            "minimum_arrival_multiplier_for_30pct_if_byte_mix_fixed": rate / (keep * lo),
            "maximum_arrival_multiplier_before_200_wire_fills_if_byte_mix_fixed": 2 * rate / hi,
            "maximum_steady_wire_100_loss_at_200_saturation_boundary_if_mix_fixed": max(0., 1 - hi / (2 * lo)),
            "steady_wire_100_loss_at_unchanged_offered_mix": max(0., 1 - rate / lo),
            "high_to_low_rate_cost_at_high_bytes_s": same_high_read_extra_s,
            "low_volume_reduction_equivalent_wire_s": saved_low_volume_s,
            "wire_time_cost_offset_fraction_by_volume": saved_low_volume_s / same_high_read_extra_s,
            "wire_time_gap_after_volume_feedback_s": same_high_read_extra_s - saved_low_volume_s,
        }
    return result


def ready_clock_bounds(path, summary):
    if sha256(path) != summary["trace_sha256"]:
        raise ValueError("Request trace digest does not match the frozen arrivals")
    per_arm_role = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["workload"] == "matched_945":
                continue
            key = row["workload"], row["order"], str(row["cap"])
            per_arm_role.setdefault(key, [[], []])[row["D"]].append((row["D_KV_ready_s"], row["output"]))
    bounds = {}
    for (name, order, cap), by_role in per_arm_role.items():
        service = summary["models"][order]["service"]
        slots = service["slots"]
        step = service["decode_first_s"]
        emit = slots / (step + (slots - 1) * service["decode_add_s"])
        fluid, single = 0., 0.
        for rows in by_role:
            outstanding = sum(q for _, q in rows)
            for ready, q in sorted(rows):
                fluid = max(fluid, ready + outstanding / emit)
                single = max(single, ready + q * step)
                outstanding -= q
        last = max(ready for rows in by_role for ready, _ in rows)
        arm = summary["preplanned_arrivals"][name][order]["arms"][cap]
        bound = max(fluid, single)
        if bound > arm["finish_s"] + 1e-7 or sum(q for rows in by_role for _, q in rows) != arm["output_tokens"]:
            raise ValueError("The fixed readiness bound must be consistent with the decode replay")
        bounds[f"{name}/{order}/{cap}"] = {
            "last_D_KV_ready_s": last, "earliest_finish_s_with_instant_generation_and_fixed_KV_ready": last,
            "optimistic_earliest_finish_s_at_calibrated_decode_rates_and_fixed_KV_ready": bound,
            "max_saving_from_any_faster_decode_if_KV_ready_unchanged_s": arm["finish_s"] - last,
            "max_saving_from_perfect_batching_at_calibrated_rate_if_KV_ready_unchanged_s": arm["finish_s"] - bound,
        }
    return bounds


def optimistic_demands(rows, cap):
    # At the timestamp boundary old prefixes are immediately available. For
    # sources without timestamps use any other row as a favorable prior owner.
    has_time = all(row.at_us is not None for row in rows)
    grouped = sorted(rows, key=lambda row: row.at_us) if has_time else rows
    available = Counter(key for row in grouped for key in row.prefixes) if not has_time else set()
    for row in grouped:
        shared = 0
        for key in row.prefixes:
            if (key not in available if has_time else available[key] < 2):
                break
            shared += 1
        if has_time:
            available.update(row.prefixes)
        suffix = row.input_tokens - PAGE * shared
        rate = cap["optimistic_warm_tokens_s_per_P"] if shared and suffix <= cap["chunk_tokens"] else (
            cap["optimistic_cold_tokens_s_per_P"])
        yield (row.at_us, row.output_tokens, read_bytes(shared), suffix / rate, suffix, shared * PAGE)


def resource_ceiling(output, read, p_seconds, window_s, cap):
    if not output:
        return {"optimistic_output_tokens_s": 0., "potential_GET_GB_s": 0.,
                "ideal_high_throughput_for_30pct_at_best_R100": None}
    limit = min(cap["optimistic_decode_tokens_s"], 2 * output / p_seconds)
    if window_s is not None:
        limit = min(limit, output / window_s)
    return {"optimistic_output_tokens_s": limit, "potential_GET_GB_s": limit * read / output / GB,
            "potential_GET_MB_per_output_token": read / output / 1e6,
            "ideal_high_throughput_for_30pct_at_best_R100":
                min(cap["R100_GB_s"]) * GB * output / ((1 - TARGET_LOSS) * read) if read else None}


def best_window(series, seconds, cap):
    times, output, payload, p_seconds, _, _ = zip(*series, strict=True)
    width = seconds * MICROSECONDS
    start, final_start = times[0], times[-1] - width
    if final_start < start:
        return None
    edges = sorted({t for edge in times for t in (edge, edge - width) if start <= t <= final_start})
    out, read, p = ([0, *accumulate(values)] for values in (output, payload, p_seconds))
    best = None
    for at in edges:
        for locate in (bisect_left, bisect_right):
            i, j = locate(times, at), locate(times, at + width)
            if out[j] == out[i]:
                continue
            q, b, work = out[j] - out[i], read[j] - read[i], p[j] - p[i]
            result = resource_ceiling(q, b, work, seconds, cap)
            if best is None or result["potential_GET_GB_s"] > best["potential_GET_GB_s"]:
                best = {**result, "source_start_s": at / MICROSECONDS, "output_tokens": q,
                        "potential_GET_GB": b / GB, "minimum_total_P_compute_s": work, "rows": j - i}
    return best


def profile_trace(url, block_size, cap):
    raw_hash = digest_sha256()
    requests, invalid = [], 0
    with urlopen(url, timeout=35) as response:
        for line in response:
            raw_hash.update(line)
            row = json.loads(line)
            prompt, output = row["input_length"], row["output_length"]
            if prompt <= 0 or output < 0 or prompt + output > CONTEXT_LIMIT:
                invalid += 1
            else:
                requests.append(read_demand(row, block_size))
    series = list(optimistic_demands(requests, cap))
    _, output, payload, prefill_times, suffix, shared = zip(*series, strict=True)
    q, b, prefill_s = sum(output), sum(payload), sum(prefill_times)
    has_time = all(row.at_us is not None for row in requests)
    full_prompt_upper = sum(read_bytes((row.input_tokens - 1) // PAGE) for row in requests)
    windows = {str(seconds): best_window(series, seconds, cap) for seconds in (60, 120)} if has_time else None
    return {"url": url, "source_sha256": raw_hash.hexdigest(), "source_block_tokens": block_size,
            "compatible": len(requests), "excluded": invalid, "has_original_timestamps": has_time,
            "prompt_tokens_p50_p90_p99": [percentile([row.input_tokens for row in requests], v) for v in (.5, .9, .99)],
            "output_tokens_p50_p90_p99": [percentile([row.output_tokens for row in requests], v) for v in (.5, .9, .99)],
            "whole_trace_reference_output": q, "whole_trace_matched_input_tokens": sum(shared),
            "whole_trace_minimum_prefill_tokens": sum(suffix), "whole_trace_Qwen_natural_EOS_available": False,
            "whole_trace_all_prompt_GET_upper_MB_per_output": full_prompt_upper / q / 1e6,
            "whole_trace_optimistic_balanced_mix": resource_ceiling(q, b, prefill_s, None, cap),
            "best_original_clock_windows": windows}


def agentx_all_input_upper(cap):
    # Published aggregate totals for the 256k trace, including all branches.
    prompt, output, requests = 6_891_228_864, 58_728_807, 68_266
    checkpoint_per_request = read_bytes(1) - ATTENTION_BYTES
    upper = prompt * ATTENTION_BYTES / PAGE + requests * checkpoint_per_request
    mb_per_output = upper / output / 1e6
    return {"source": "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/blob/main/README.md",
            "requests": requests, "prompt_tokens": prompt, "output_tokens": output,
            "optimistic_all_input_GET_MB_per_output_token": mb_per_output,
            "optimistic_all_input_GET_GB_s_at_D_ceiling": mb_per_output * cap["optimistic_decode_tokens_s"] / 1000}


def agentx_requests(entries, parents=()):
    for entry in entries:
        if entry["type"] == "subagent":
            yield from agentx_requests(entry["requests"], (*parents, entry["t"]))
        else:
            yield entry, parents


def peak_arrival_payload(series, seconds):
    times = [row[0] for row in series]
    total = [0, *accumulate(row[2] for row in series)]
    duration = seconds * MICROSECONDS
    return max(total[bisect_left(times, start + duration)] - total[bisect_left(times, start)]
               for start in set(times)) / GB


def agentx_session_screen(cap):
    raw_hash = digest_sha256()
    sessions, observed_requests, observed_output = [], 0, 0
    strongest = {}
    relative_children = False
    with urlopen(AGENTX_URL, timeout=35) as response:
        for line in response:
            raw_hash.update(line)
            session = json.loads(line)
            if session["hash_id_scope"] != "local":
                raise ValueError("AgentX prefix namespaces should be local to each session")
            requests, timing = [], []
            for entry, parents in agentx_requests(session["requests"]):
                prompt, output = entry["in"], entry["out"]
                observed_requests += 1
                observed_output += output
                if parents and entry["t"] + 1e-6 < parents[-1]:
                    relative_children = True
                if 0 < prompt and 0 <= output and prompt + output <= CONTEXT_LIMIT:
                    requests.append(read_demand({"input_length": prompt, "output_length": output,
                                                 "hash_ids": entry.get("hash_ids") or []}, session["block_size"]))
                    timing.append((entry["t"], sum(parents)))
            if not requests or not sum(row.output_tokens for row in requests):
                continue
            _, output, payload, prefill_times, _, _ = zip(*optimistic_demands(requests, cap), strict=True)
            q, b, prefill = sum(output), sum(payload), sum(prefill_times)
            score = {**resource_ceiling(q, b, prefill, None, cap), "session": session["id"],
                     "requests": len(requests), "output_tokens": q, "potential_GET_GB": b / GB}
            sessions.append(score)
            strongest[session["id"]] = (score["potential_GET_GB_s"], requests, timing)
            if len(strongest) > 5:
                del strongest[min(strongest, key=lambda key: strongest[key][0])]
    best_rate = min(cap["R100_GB_s"]) / (1 - TARGET_LOSS)
    worst_rate = max(cap["R100_GB_s"]) / (1 - TARGET_LOSS)
    ranked = sorted(sessions, key=lambda row: row["potential_GET_GB_s"], reverse=True)
    leaders = ranked[:5]
    for row in leaders:
        _, requests, timing = strongest[row["session"]]
        timed = [replace(request, at_us=round((at + (offset if relative_children else 0)) * MICROSECONDS))
                 for request, (at, offset) in zip(requests, timing, strict=True)]
        observed_order = list(optimistic_demands(timed, cap))
        row["recorded_span_s"] = (observed_order[-1][0] - observed_order[0][0]) / MICROSECONDS
        row["prior_observed_prefix_potential_GET_GB"] = sum(entry[2] for entry in observed_order) / GB
        row["peak_potential_GET_GB_at_observed_offsets"] = {
            str(seconds): peak_arrival_payload(observed_order, seconds) for seconds in (60, 120)}
    peak_sums = {str(seconds): sum(row["peak_potential_GET_GB_at_observed_offsets"][str(seconds)] for row in leaders)
                 for seconds in (60, 120)}
    return {"url": AGENTX_URL, "source_sha256": raw_hash.hexdigest(), "observed_requests": observed_requests,
            "observed_source_output": observed_output, "sessions_with_compatible_work": len(sessions),
            "optimistic_sessions_above_best_R100_threshold": sum(row["potential_GET_GB_s"] >= best_rate for row in ranked),
            "optimistic_sessions_above_worst_R100_threshold": sum(row["potential_GET_GB_s"] >= worst_rate for row in ranked),
            "potential_GET_GB_s_p50_p90_p99": [percentile([row["potential_GET_GB_s"] for row in sessions], q)
                                               for q in (.5, .9, .99)],
            "best_five_whole_sessions": leaders,
            "nested_times_interpreted_as_relative": relative_children,
            "sum_of_five_per_session_peaks_GB_upper_bound_for_any_root_offsets": peak_sums,
            "screen_scope": "Published 256k variant including branches. Matching peers may publish in any order. "
                            "Peak calibrated P/D budget and source output counts, no routing or physical residency assumed. "
                            "Timed peak sums permit perfect alignment of five independent roots. They retain within-session "
                            "observed offsets and require prior reported prefix evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = read_json(args.fixed / "feasibility.json")
    summary = read_json(args.fixed / "summary.json")
    cap = capacities(summary)
    benchmarks = {}
    for name, (url, block_size) in SOURCES.items():
        benchmarks[name] = profile_trace(url, block_size, cap)
        print(name, benchmarks[name]["compatible"], benchmarks[name]["whole_trace_optimistic_balanced_mix"], flush=True)
    benchmarks["agentx_256k_aggregate_only"] = agentx_all_input_upper(cap)
    benchmarks["agentx_256k_session_screen"] = agentx_session_screen(cap)
    write_json(args.output, {"fixed_arrival_feasibility_sha256": sha256(args.fixed / "feasibility.json"),
                             "target_100_loss_relative_to_200": TARGET_LOSS, "calibrated_resource_speeds": cap,
                             "fixed_mix": fixed_mix(prior), "fixed_KV_ready_decode_bounds": ready_clock_bounds(args.trace, summary),
                             "benchmark_prefilter": benchmarks})


if __name__ == "__main__":
    main()
