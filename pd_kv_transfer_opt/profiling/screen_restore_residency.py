"""Screen natural session windows for cache-size pressure and sustained D egress demand."""

import argparse
from collections import defaultdict
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, read_jsonl, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import ALIGN_STATE_OBJECT_BYTES, ATTENTION_OBJECT_BYTES
from pd_kv_transfer_opt.profiling.screen_bidirectional_workload import D_RATE


ATTENTION_BLOCK_TOKENS = 1024
ATTENTION_BYTES = 4 * ATTENTION_OBJECT_BYTES
STATE_BYTES = 12 * ALIGN_STATE_OBJECT_BYTES
P_CAPACITY_TOKENS = 3_093_504


def long_prompt_fits(row, count, output_limit):
    return (not row["truncated"] and type(count) is int and 0 < count <= output_limit
            and 24_000 <= row["prompt_tokens"] <= 262_144 - max(8192, count))


def session_windows(rows, outputs, min_turns, max_turns, output_limit):
    for first in range(len(rows)):
        peak_context, get_bytes, decoded = 0, 0, 0
        previous = None
        for turns, row in enumerate(rows[first:first + max_turns], 1):
            count = outputs.get(row["id"])
            prompt = row["prompt_tokens"]
            if not long_prompt_fits(row, count, output_limit):
                break
            if previous is not None:
                lcp = row["source_prompt_lcp_to_previous"]
                if row["predecessor"] != previous["id"] or lcp < 0.9 * previous["prompt_tokens"]:
                    break
                get_bytes += lcp // ATTENTION_BLOCK_TOKENS * ATTENTION_BYTES + STATE_BYTES
                decoded += count
            peak_context = max(peak_context, prompt)
            if turns >= min_turns:
                # Optimistically ignore first-turn work and count a full preceding prefix GET on every later turn.
                yield peak_context, get_bytes * 8e-9 * D_RATE, decoded
            previous = row


def resident_prefix_windows(rows, outputs, output_limit, max_turns):
    """Bound remote bytes when each P retains the preceding prefixes it handled."""
    for start, root in enumerate(rows):
        if not long_prompt_fits(root, outputs.get(root["id"]), output_limit):
            continue
        previous = root
        common_blocks = []
        # The state is the last turn seen by the P without the immediate predecessor.
        by_other_turn = {-1: 0}
        prefill_tokens = root["prompt_tokens"]
        for turn, row in enumerate(rows[start + 1:start + max_turns], 1):
            if row["predecessor"] != previous["id"] or not long_prompt_fits(row, outputs.get(row["id"]), output_limit):
                break
            lcp = row["source_prompt_lcp_to_previous"]
            if lcp < 0.9 * previous["prompt_tokens"]:
                break
            prefix = lcp // ATTENTION_BLOCK_TOKENS
            common_blocks = [min(older, prefix) for older in common_blocks] + [prefix]
            if_other = max(cost + (prefix - (common_blocks[t] if t >= 0 else 0)) * ATTENTION_BYTES
                           for t, cost in by_other_turn.items())
            by_other_turn[turn - 1] = if_other
            prefill_tokens += row["prompt_tokens"] - prefix * ATTENTION_BLOCK_TOKENS
            previous = row
            if turn >= 2:
                # Credit the aligned Hybrid state on every successor even for a local turn.
                volume = max(by_other_turn.values()) + turn * STATE_BYTES
                yield (volume, prefill_tokens, root["id"], row["id"], turn + 1)


def resident_prefix_screen(sessions, outputs):
    result = []
    for limit in (512, 1024, 2048, 8192):
        largest = [choice for session in sessions
                   if (choice := max(resident_prefix_windows(session, outputs, limit, 10), default=None))]
        result.append({"source_completion_cutoff": limit, "eligible_tasks": len(largest),
                       "sum_upper_GET_bytes": sum(row[0] for row in largest),
                       "prefill_tokens_at_chosen_windows": sum(row[1] for row in largest),
                       "largest_windows": sorted(largest, reverse=True)[:6]})
    return result


def upper_certificate(groups, minimum_context, rate):
    """Bound 8 * D_RATE * sum(GET bytes in GB) - rate * sum(output tokens)."""
    linear = [[(size, weighted_bytes - rate * tokens) for size, weighted_bytes, tokens in choices] for choices in groups]

    def bound(price):
        return sum(max(0, max(intercept + price * size for size, intercept in choices))
                   for choices in linear) - price * minimum_context

    # Any nonnegative price gives an upper bound. Search for a tighter certificate.
    low, high = 0.0, 8.0
    for _ in range(28):
        left, right = (2 * low + high) / 3, (low + 2 * high) / 3
        if bound(left) > bound(right):
            low = left
        else:
            high = right
    return min(bound(0), bound((low + high) / 2))


def analyze(ledger, source_outputs, capacity):
    meta = read_json(source_outputs)
    outputs = meta["source_completion_tokens"]
    rows = list(read_jsonl(ledger))
    if {r["id"] for r in rows} != outputs.keys():
        raise ValueError("The length ledger and output metadata cover different requests")
    sessions = defaultdict(list)
    for row in rows:
        sessions[row["task"]].append(row)
    for session in sessions.values():
        session.sort(key=lambda row: row["step_id"])
    results = []
    for min_turns, max_turns in ((4, 12), (6, 12), (8, 20), (12, 25)):
        for cutoff in (2048, 4096, 8192, 16384):
            groups = [windows for session in sessions.values()
                      if (windows := tuple(session_windows(session, outputs, min_turns, max_turns, cutoff)))]
            working_set_ceiling = sum(max(row[0] for row in choices) for choices in groups)
            bandwidth_upper, certificate = None, None
            if working_set_ceiling >= capacity:
                certificate = upper_certificate(groups, capacity, 100)
                low, high = 0.0, max(weighted / out for group in groups for _, weighted, out in group)
                for _ in range(18):
                    trial = (low + high) / 2
                    if upper_certificate(groups, capacity, trial) > 0:
                        low = trial
                    else:
                        high = trial
                bandwidth_upper = high
            results.append({"turn_range": [min_turns, max_turns], "source_completion_cutoff": cutoff,
                            "eligible_tasks": len(groups), "eligible_windows": sum(map(len, groups)),
                            "largest_total_live_prefix_tokens": working_set_ceiling,
                            "D_egress_upper_Gbps_if_working_set_reaches_P_capacity": bandwidth_upper,
                            "100_Gbps_upper_certificate": certificate})
    return {"source_revision": meta["source_revision"], "ledger_sha256": sha256(ledger),
            "source_outputs_sha256": sha256(source_outputs), "source_tasks": len(sessions),
            "source_turns": len(rows), "single_P_capacity_tokens": capacity,
            "D_service_reference_output_tokens_s": D_RATE,
            "assumptions": ["One continuous window per source task and one live prefix per session",
                            "Keep consecutive untruncated prompts of at least 24000 tokens at 90 percent or greater adjacent prefix overlap",
                            "Credit every successor with a full shared-prefix GET and generous four-rank state bytes",
                            "Exclude first-turn cost, assume no Store evictions and use the source model's outputs",
                            "Bound aggregate demand at assumed D service, not the instantaneous load or hardware capacity of one hot port"],
            "cases": results,
            "resident_prefix_upper": {
                "assumptions": ["Three to ten contiguous turns in each source window",
                                "Neither P evicts a prefix during the window",
                                "Keep the immediate predecessor in Store and credit all missing full attention groups",
                                "Minimum adjacent LCP is a lower bound on any earlier prefix still resident",
                                "Choose P routes optimally for read volume and count Hybrid state on each successor"],
                "cases": resident_prefix_screen(list(sessions.values()), outputs)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("source_outputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--P-capacity-tokens", type=int, default=P_CAPACITY_TOKENS)
    args = parser.parse_args()
    write_json(args.output, analyze(args.ledger, args.source_outputs, args.P_capacity_tokens))


if __name__ == "__main__":
    main()
