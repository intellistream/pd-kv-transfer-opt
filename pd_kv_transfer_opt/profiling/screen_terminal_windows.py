"""Screen original Terminal-Bench windows for native P-to-D demand."""

import argparse
from collections import defaultdict
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, read_jsonl, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import ALIGN_STATE_OBJECT_BYTES, ATTENTION_OBJECT_BYTES
from pd_kv_transfer_opt.profiling.screen_bidirectional_workload import D_RATE, P_RATE


BLOCK = 1024
ATTENTION_BYTES = 4 * ATTENTION_OBJECT_BYTES
STATE_BYTES = 12 * ALIGN_STATE_OBJECT_BYTES


def advance_d(states, blocks, retained_lcp, turn):
    def payload(hit):
        return (blocks - hit) * ATTENTION_BYTES + STATE_BYTES

    most_recent_payload = payload(retained_lcp[turn - 1])
    result = {older: past + most_recent_payload for older, past in states.items()}
    result[turn - 1] = max(past + payload(retained_lcp[older] if older >= 0 else 0) for older, past in states.items())
    return result


def windows(rows, output):
    for start, root in enumerate(rows):
        n = output.get(root["id"])
        if not valid(root, n):
            continue
        computed = root["prompt_tokens"]
        decoded = n
        written_bytes = computed // BLOCK * ATTENTION_BYTES + STATE_BYTES
        states = {-1: written_bytes}
        retained_lcp = []
        long_turns = int(computed >= 24000)
        previous = root
        for turn, row in enumerate(rows[start + 1:], 1):
            n = output.get(row["id"])
            if not valid(row, n) or row["predecessor"] != previous["id"]:
                break
            size = row["prompt_tokens"]
            prev_lcp = min(size // BLOCK, row["source_prompt_lcp_to_previous"] // BLOCK)
            retained_lcp = [min(old, prev_lcp) for old in retained_lcp] + [prev_lcp]
            states = advance_d(states, size // BLOCK, retained_lcp, turn)
            computed += size - prev_lcp * BLOCK
            decoded += n
            written_bytes += (size // BLOCK - prev_lcp) * ATTENTION_BYTES + STATE_BYTES
            long_turns += size >= 24000
            previous = row
            if long_turns < 2:
                continue
            delivered = max(states.values())
            p_bytes = delivered + written_bytes
            capacity = min(P_RATE / computed, D_RATE / decoded)
            yield {"task": root["task"], "first": root["id"], "last": row["id"], "turns": turn + 1,
                   "long_turns": long_turns, "root_context_tokens": root["prompt_tokens"],
                   "ideal_P_compute_tokens": computed, "source_generated_tokens": decoded,
                   "max_D_delivery_GB": delivered / 1e9, "modeled_P_Store_write_GB": written_bytes / 1e9,
                   "D_delivery_Gbps_at_reference": delivered * capacity * 8e-9,
                   "P_TX_Gbps_at_P_reference": p_bytes * P_RATE / computed * 8e-9,
                   "P_TX_Gbps_at_reference": p_bytes * capacity * 8e-9}


def valid(row, output):
    return (not row["truncated"] and type(output) is int and output > 0
            and 0 < row["prompt_tokens"] <= 262144 - max(8192, output))


def analyze(ledger, output, comparison_plan):
    output_meta = read_json(output)
    r48 = read_json(comparison_plan)
    if output_meta["source_revision"] != r48["source_revision"]:
        raise ValueError("Different source revisions")
    records = list(read_jsonl(ledger))
    output_lengths = output_meta["source_completion_tokens"]
    if {row["id"] for row in records} != output_lengths.keys():
        raise ValueError("The length ledger and completion mapping cover different steps")
    sessions = defaultdict(list)
    for row in records:
        sessions[row["task"]].append(row)
    p_rates = []
    highest_P_reference = 0
    best = {}
    for task, rows in sessions.items():
        rows.sort(key=lambda row: row["step_id"])
        candidates = list(windows(rows, output_lengths))
        if candidates:
            best[task] = max(candidates, key=lambda row: row["P_TX_Gbps_at_reference"])
            p_rates.extend(row["P_TX_Gbps_at_reference"] for row in candidates)
            highest_P_reference = max(highest_P_reference, max(row["P_TX_Gbps_at_P_reference"] for row in candidates))
    ranked = sorted(best.values(), key=lambda row: -row["P_TX_Gbps_at_reference"])
    r48_tasks = {row["session_id"] for row in r48["request_sources"]}
    return {"source_revision": r48["source_revision"], "ledger_sha256": sha256(ledger),
              "source_output_sha256": sha256(output), "comparison_plan_sha256": sha256(comparison_plan),
              "source_tasks": len(sessions), "screened_windows": len(p_rates),
              "P_compute_reference_token_s": P_RATE, "D_reference_token_s": D_RATE,
              "best_P_TX_Gbps_including_only_P_supply": highest_P_reference,
              "tasks_with_at_least_two_long_turns": len(best), "windows_at_least_100_Gbps": sum(v >= 100 for v in p_rates),
              "tasks_at_least_100_Gbps": [row["task"] for row in ranked if row["P_TX_Gbps_at_reference"] >= 100],
              "best": ranked[:8], "best_not_in_r48": [row for row in ranked if row["task"] not in r48_tasks][:8],
              "assumptions": ["least possible new P prefix compute from prompt LCP including the cold first input",
                              "two resident D APC replicas; maximize missing complete prefix groups across D choices",
                              "one optimistic model of background Store writes by new prefix, with Hybrid state each turn",
                              "upstream published completion lengths and separate hardware service-rate references",
                              "sequential windows only; independent sessions, routing, contention and eviction are not simulated"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("source_outputs", type=Path)
    parser.add_argument("comparison_plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(args.ledger, args.source_outputs, args.comparison_plan))


if __name__ == "__main__":
    main()
