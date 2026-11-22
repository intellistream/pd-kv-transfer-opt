"""Check a bandwidth-gap target and screen a public agent trace for read pressure."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from heapq import heapify, heapreplace
import json
from pathlib import Path

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS, ATTENTION_BYTES, STATE_BYTES
from pd_kv_transfer_opt.simulator.terminal_magnitude import source_options
from pd_kv_transfer_opt.simulator.terminal_supply import load_source


TRACE_BLOCK_TOKENS = 16
NATIVE_PORTS = 8
MAX_CONTEXT = 262_144
NORMAL_FINISH = {"stop", "tool_calls"}


def completed_rate(tokens, seconds):
    return {"output_tokens": tokens, "seconds": seconds, "output_tokens_s": tokens / seconds}


def target_from_candidate(path, target):
    report = read_json(path / "summary.json")["candidate"]
    manifest = path / "candidate-workload.json"
    if sha256(manifest) != report["workload_sha256"]:
        raise ValueError("The target calculation needs the verified frozen candidate")
    workload = read_json(manifest)
    observed = {}
    for model in report["models"]:
        width = model["service"]["GET_batch_width"]
        efficiency = model["service"]["rail_efficiency"]
        wave_start = model["wave_start_s"]
        for order, case in model["main_ABBA"].items():
            low, high = (case["arms"][cap] for cap in ("100", "200"))
            if low["output_tokens"] != high["output_tokens"]:
                raise ValueError("A shared output count is necessary for a time-only target calculation")
            wire = low["D_TX_GB_by_port"]
            if not all(abs(a - b) < 1e-7 for a, b in zip(wire, high["D_TX_GB_by_port"], strict=True)):
                raise ValueError("Expected identical wire work at both caps")
            gb = sum(wire)
            seconds_100, seconds_200 = low["duration_s"], high["duration_s"]
            delta = seconds_100 - seconds_200
            wire_seconds_100 = 8 * max(wire) / (efficiency * 100 / NATIVE_PORTS)
            wire_rate = gb / wire_seconds_100
            fraction = delta / (wire_seconds_100 / 2)
            overhead_100 = seconds_100 - wire_seconds_100
            gap_factor = target / (1 - target)
            required_ratio = target / (.5 - target)
            exposure_denominator = fraction - target * (1 + fraction)
            child_tokens = sum(row["observed_EOS_reference_tokens"][order] for row in workload["request_sources"]
                               if row["predecessor"] is not None)
            child_seconds_100 = seconds_100 - wave_start
            child_seconds_200 = seconds_200 - wave_start
            scale_margin = delta - target * child_seconds_100
            observed[f"width_{width}/{order}"] = {
                "full_round": {"100": completed_rate(low["output_tokens"], seconds_100),
                               "200": completed_rate(high["output_tokens"], seconds_200)},
                "full_round_100_loss": 1 - seconds_200 / seconds_100,
                "ready_wave_and_drain": {"100": completed_rate(child_tokens, child_seconds_100),
                                         "200": completed_rate(child_tokens, child_seconds_200)},
                "ready_wave_100_loss": 1 - child_seconds_200 / child_seconds_100,
                "ready_wave_loss_after_r78_error": 1 - child_seconds_200 / max(
                    child_seconds_200, child_seconds_100 - model["heldout_error"]["r78_positive_finish_gap_error_s"]),
                "shared_cold_preparation_s": wave_start, "equal_D_egress_GB": gb,
                "fitted_effective_100_GB_s": wire_rate, "effective_100_link_service_s": wire_seconds_100,
                "net_observed_finish_gap_s": delta, "required_finish_gap_at_current_200_s": gap_factor * seconds_200,
                "GB_if_only_cold_prep_remains_and_all_service_exposed": required_ratio * wave_start * wire_rate,
                "GB_if_current_common_time_and_all_service_exposed": required_ratio * overhead_100 * wire_rate,
                "GB_if_current_200_overhead_and_net_exposure_persist": (
                    2 * target * (seconds_200 - wire_seconds_100 / 2) * wire_rate / exposure_denominator
                    if exposure_denominator > 0 else None),
                "largest_preparation_s_at_current_wave_and_target": delta / target - child_seconds_100,
                "serial_waves_needed_if_cold_paid_only_once": wave_start * target / scale_margin if scale_margin > 0 else None,
            }
    return workload, observed


def corpus_read_envelope(ledger, budgets, outcomes, sizes, limit):
    # Each single-edge range is available from source_options. The path matching
    # relaxes port, route, cache residency and repeated-checkpoint constraints.
    options = source_options(ledger, budgets, outcomes, sizes, limit)
    by_task = {task: {ledger[row["root"]]["step_id"]: row["potential_GET_bytes"]
                      for variant in variants for row in variant.pairs} for task, variants in options.items()}
    selected = []
    for edges in by_task.values():
        last = None
        previous = best = (0, 0)
        for step, demand in sorted(edges.items()):
            disjoint = previous if last is not None and step == last + 1 else best
            previous, best = best, max(best, (disjoint[0] + demand, disjoint[1] + 1))
            last = step
        selected.append(best)
    return {"observed_EOS_limit": limit, "tasks": len(by_task), "adjacent_edges": sum(map(len, by_task.values())),
            "disjoint_pairs_in_maximum_read_cohort": sum(count for _, count in selected),
            "full_cross_P_and_store_hit_ceiling_GB": sum(volume for volume, _ in selected) / 1e9}


def shared_native_blocks(a, b, token_ceiling):
    blocks_per_page = ATTENTION_BLOCK_TOKENS // TRACE_BLOCK_TOKENS
    matching = 0
    possible = min(len(a), len(b), token_ceiling // TRACE_BLOCK_TOKENS) // blocks_per_page
    while matching < possible:
        middle = (matching + possible + 1) // 2
        unchecked = slice(matching * blocks_per_page, middle * blocks_per_page)
        if a[unchecked] == b[unchecked]:
            matching = middle
        else:
            possible = middle - 1
    return matching


def read_bytes(blocks):
    return blocks * ATTENTION_BYTES + STATE_BYTES if blocks else 0


def fits(row):
    tokens = row["provider_tokens"]
    prompt, output = tokens["prompt"], tokens["completion"]
    return (type(prompt) is int and type(output) is int and prompt > 0 and output > 0
            and max(prompt, len(row["block_ids"] or []) * TRACE_BLOCK_TOKENS) + output <= MAX_CONTEXT
            and row["finish_reason"] in NORMAL_FINISH and row["end_ms"] >= row["start_ms"])


def same_source(a, b):
    return a["model"] == b["model"] and a["provider"] == b["provider"]


@dataclass(frozen=True)
class Turn:
    start_ms: int
    output: int
    prefix_blocks: int
    full_read_bytes: int
    return_read_bytes: int


def session_pressure(requests):
    turns = []
    good = [fits(row) for row in requests]
    for i, row in enumerate(requests):
        if not good[i]:
            continue
        shared = local = 0
        if i:
            before = requests[i - 1]
            if good[i - 1] and same_source(before, row) and before["end_ms"] <= row["start_ms"]:
                shared = shared_native_blocks(before["block_ids"] or [], row["block_ids"] or [],
                                              min(before["provider_tokens"]["prompt"], row["provider_tokens"]["prompt"]))
                if i >= 2:
                    earlier = requests[i - 2]
                    if good[i - 2] and same_source(earlier, row) and earlier["end_ms"] <= before["start_ms"]:
                        local = min(shared, shared_native_blocks(earlier["block_ids"] or [], row["block_ids"] or [],
                                    min(earlier["provider_tokens"]["prompt"], row["provider_tokens"]["prompt"])))
        turns.append(Turn(row["start_ms"], row["provider_tokens"]["completion"], shared, read_bytes(shared),
                          shared * ATTENTION_BYTES + STATE_BYTES - local * ATTENTION_BYTES if shared else 0))
    return turns


def densest_window(turns, seconds, *, returning=False):
    q = deque()
    current = tokens = highest = 0
    selected = {"window_s": seconds, "GB": 0.0, "Gbps": 0.0}
    for turn in sorted(turns, key=lambda item: item.start_ms):
        amount = turn.return_read_bytes if returning else turn.full_read_bytes
        q.append((turn.start_ms, amount, turn.output))
        current += amount
        tokens += turn.output
        while turn.start_ms - q[0][0] >= seconds * 1000:
            _, old, output = q.popleft()
            current -= old
            tokens -= output
        if current > highest:
            highest = current
            selected = {"window_s": seconds, "GB": current / 1e9, "Gbps": 8 * current / 1e9 / seconds,
                        "ending_timestamp_ms": turn.start_ms, "arriving_requests": len(q),
                        "recorded_output_tokens_for_arrivals": tokens}
    return selected


def least_required_read_fraction(turns, seconds, rate_100_GB_s, decode_tokens_s, target):
    """Find the most favorable simultaneous read and generation work in a fixed window."""
    arrivals = deque()
    tokens = volume = retained_volume = 0
    best = None
    for turn in sorted(turns, key=lambda row: row.start_ms):
        arrivals.append(turn)
        tokens += turn.output
        volume += turn.full_read_bytes
        retained_volume += turn.return_read_bytes
        while turn.start_ms - arrivals[0].start_ms >= seconds * 1000:
            oldest = arrivals.popleft()
            tokens -= oldest.output
            volume -= oldest.full_read_bytes
            retained_volume -= oldest.return_read_bytes
        if not volume:
            continue
        high_without_link = min(tokens / seconds, decode_tokens_s)
        required_GB = rate_100_GB_s * tokens / ((1 - target) * high_without_link)
        share = required_GB / (volume / 1e9)
        if best is None or share < best["required_fraction_of_every_read"]:
            best = {"window_s": seconds, "ending_timestamp_ms": turn.start_ms, "requests": len(arrivals),
                    "source_output_tokens": tokens, "full_read_GB": volume / 1e9, "required_GB": required_GB,
                    "read_GB_with_alternation_and_previous_visit_retained": retained_volume / 1e9,
                    "potential_read_MB_per_output": volume / tokens / 1e6, "required_fraction_of_every_read": share}
    return best


def projected_concurrency(sessions, concurrency):
    """Start each original session once, keeping its recorded within-session clock."""
    slots = [0] * concurrency
    heapify(slots)
    for original_start, duration, turns in sorted(sessions, key=lambda row: row[0]):
        start = slots[0]
        heapreplace(slots, start + duration)
        for r in turns:
            yield Turn(start + r.start_ms - original_start, r.output, r.prefix_blocks, r.full_read_bytes, r.return_read_bytes)


def trace_input(paths, rate_100_GB_s, decode_tokens_s, target):
    # Import only for the released .zst shards. The gzip/JSONL loader is
    # independent of whether the runtime includes the Zstandard module.
    import gzip

    all_turns, whole, sources = [], [], {}
    seen = set()
    for path in paths:
        if path.suffix == ".zst":
            try:
                from compression import zstd
            except ImportError:
                import zstandard as zstd
            stream = zstd.open(path, "rt", encoding="utf-8")
        elif path.suffix == ".gz":
            stream = gzip.open(path, "rt", encoding="utf-8")
        else:
            stream = path.open(encoding="utf-8")
        turns, eligible_sessions = [], []
        counts = Counter()
        with stream:
            for line in stream:
                session = json.loads(line)
                key = (session["week"], session["session_id"])
                if key in seen:
                    raise ValueError("The same source session appeared twice")
                seen.add(key)
                counts["sessions"] += 1
                requests = session["requests"]
                counts["requests"] += len(requests)
                current = session_pressure(requests)
                turns.extend(current)
                counts["usable_requests"] += len(current)
                if current and len(current) == len(requests) and all(a["end_ms"] <= b["start_ms"] for a, b in zip(requests, requests[1:])):
                    t = requests[0]["start_ms"]
                    eligible_sessions.append((t, requests[-1]["end_ms"] - t, current))
        pressure = [densest_window(turns, s) for s in (1, 5, 20, 60, 120, 300)]
        long = [r for r in turns if r.prefix_blocks * ATTENTION_BLOCK_TOKENS >= 65_536]
        sources[path.name] = {"sha256": sha256(path), "counts": dict(counts), "whole_compatible_sessions": len(eligible_sessions),
                              "long_predecessor_pairs": len(long), "long_pair_median_reference_output": percentile(
                                  [r.output for r in long], .5) if long else None, "recorded_clock": pressure}
        all_turns.extend(turns)
        whole.extend(eligible_sessions)
    projection = {}
    for n in (16, 32, 64, 128, 256):
        if n <= len(whole):
            ready = list(projected_concurrency(whole, n))
            projection[str(n)] = {"every_adjacent_prefix_read": [densest_window(ready, s) for s in (20, 60, 120)],
                                  "alternating_P_with_local_previous_visit_retained": [
                                      densest_window(ready, s, returning=True) for s in (20, 60, 120)],
                                  "steady_state_minimum_full_read_fraction": least_required_read_fraction(
                                      ready, 120, rate_100_GB_s, decode_tokens_s, target)}
    min_bytes_per_output = rate_100_GB_s * 1e9 / ((1 - target) * decode_tokens_s)
    high_read_sessions = [row for row in whole if sum(t.full_read_bytes for t in row[2])
                          >= min_bytes_per_output * sum(t.output for t in row[2])]
    short_output_pressure = {}
    for n in sorted({16, 32, 64, 128, 256, min(len(high_read_sessions), 256)} - {0}):
        if n <= len(high_read_sessions):
            ready = list(projected_concurrency(high_read_sessions, n))
            short_output_pressure[str(n)] = {"every_adjacent_prefix_read": densest_window(ready, 120),
                "alternating_P_with_local_previous_visit_retained": densest_window(ready, 120, returning=True),
                "steady_state_minimum_full_read_fraction": least_required_read_fraction(
                    ready, 120, rate_100_GB_s, decode_tokens_s, target)}
    return {"sources": sources, "complete_compatible_sessions": len(whole),
            "combined_recorded_clock": [densest_window(all_turns, s) for s in (1, 5, 20, 60, 120, 300)],
            "concurrency_demand_screen": projection,
            "steady_state_gate": {"effective_100_GB_s": rate_100_GB_s, "decode_peak_tokens_s": decode_tokens_s,
                                  "minimum_remote_MB_per_output_at_decode_peak": min_bytes_per_output / 1e6,
                                  "high_read_density_complete_sessions": len(high_read_sessions),
                                  "screen_at_controlled_concurrency": short_output_pressure},
            "scope": "Top-level source sessions. Keep original predecessor timing. For the concurrency screen, start each "
                     "eligible complete session once while preserving its original internal clock. Prefix identities "
                     "are source 16-token hashes grouped into target-size attention blocks. The read envelopes assume "
                     "every eligible adjacent read reaches Store, or alternate P with the earlier local prefix retained. "
                     "Source provider output counts are replay work, not measured Qwen EOS. This is not an end-to-end A/B."}


def codex_schema(path):
    traces = read_json(path)
    rows = [row for trace in traces for row in trace["conversations"]]
    return {"sha256": sha256(path), "trajectories": len(traces), "messages": len(rows),
            "trace_fields": sorted({key for row in traces for key in row}),
            "message_fields": sorted({key for row in rows for key in row}),
            "role_counts": dict(Counter(row["from"] for row in rows))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("candidate", "ledger", "eos-receipt", "source-completions", "output"):
        parser.add_argument(f"--{k}", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--freeinference", type=Path, action="append", required=True)
    parser.add_argument("--codex", type=Path, required=True)
    parser.add_argument("--target", type=float, default=.30)
    args = parser.parse_args()
    if not 0 < args.target < .5:
        parser.error("the serialized-link target must be between zero and the 50% asymptote")
    workload, target = target_from_candidate(args.candidate, args.target)
    rate_100 = max(row["fitted_effective_100_GB_s"] for row in target.values())
    decode = read_json(args.results / "2026-11-22-cpu-terminal-supply" / "summary.json")["decode_fit"]
    total_decode_rate = 32 / (decode["first_sequence_period_s"] + 15 * decode["additional_sequence_period_s"])
    ledger, budgets, eos, sizes = load_source(args.ledger, args.eos_receipt, args.source_completions,
                                            workload["source_revision"], args.results)
    write_json(args.output, {"100_loss_target": args.target, "Terminal_Bench_68_window_target": target,
               "Terminal_Bench_source_envelope": [corpus_read_envelope(ledger, budgets, eos, sizes, limit)
                                                  for limit in (675, 1400, 8192)],
               "FreeInference": trace_input(args.freeinference, rate_100, total_decode_rate, args.target),
               "vLLM_Codex_SWE_bench_Pro": codex_schema(args.codex)})


if __name__ == "__main__":
    main()
