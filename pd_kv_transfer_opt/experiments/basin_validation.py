"""Run one frozen Basin-style warm-start cohort through the native proxy."""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from threading import Lock

from pd_kv_transfer_opt.common import new_directory, read_json, read_jsonl, sha256, write_json
from pd_kv_transfer_opt.experiments.analyze import summarize
from pd_kv_transfer_opt.experiments.loadgen import drive, validate_workload
from pd_kv_transfer_opt.workloads.materialize import token_digest


def split_arm(plan, prompts, arm):
    if plan["schema"] != "basin-small-replay/v1" or plan["ignore_eos"]:
        raise ValueError("Expected the frozen natural-EOS validation plan")
    schedule = plan["schedule"]
    warm, base, extras = (schedule[k] for k in ("initial_sessions", "base_offers_s", "extra_offers_s"))
    added = schedule["arms"][arm]
    if (set(warm) & set(base) or set(warm) & set(extras) or set(base) & set(extras)
            or len(added) != len(set(added)) or not set(added) <= set(extras)):
        raise ValueError("A source session appears in more than one arrival stream")
    if not all(schedule["surge_window_s"][0] <= t < schedule["surge_window_s"][1] for t in extras.values()):
        raise ValueError("Surge arrivals must stay within the pulse window")
    all_ids = {row["id"] for row in plan["requests"]}
    if len(all_ids) != len(plan["requests"]) or all_ids != prompts.keys():
        raise ValueError("Materialized inputs must cover the frozen cohort exactly once")
    sessions = defaultdict(list)
    for row in plan["requests"]:
        key, sid = row["id"], row["session_id"]
        prompt = prompts[key]
        if (len(prompt) != row["prompt_tokens"] or token_digest(prompt) != row["token_ids_uint32le_sha256"]
                or len(prompt) + row["max_tokens"] > plan["max_context_tokens"]):
            raise ValueError(f"Bad token IDs or context budget: {key}")
        prior = sessions[sid][-1]["id"] if sessions[sid] else None
        if prior != row["predecessor"]:
            raise ValueError(f"Broken source session order: {key}")
        sessions[sid].append({**row, "prompt": prompt})
    if sessions.keys() != (set(warm) | set(base) | set(extras)):
        raise ValueError("The arrival plan does not cover the frozen cohort")

    prime, measure = [], []

    def phase_rows(chain, offer, cohort):
        return [{**row, "offered_s": offer, "cohort": cohort,
                 "source_predecessor": row["predecessor"],
                 "predecessor": row["predecessor"] if i else None}
                for i, row in enumerate(chain)]

    for sid, item in warm.items():
        chain, cut = sessions[sid], item["prime_turns"]
        remaining = item["initial_remaining_tool_s"]
        if not 0 < cut < len(chain) or not 0 <= remaining <= chain[cut]["tool_gap_s"]:
            raise ValueError(f"Bad mid-trajectory seed: {sid}")
        prime.extend(phase_rows(chain[:cut], item["prime_start_s"], "priming"))
        measure.extend(phase_rows(chain[cut:], remaining, "initial"))
    for sid, start in base.items():
        measure.extend(phase_rows(sessions[sid], start, "base"))
    for sid in added:
        measure.extend(phase_rows(sessions[sid], extras[sid], "surge"))
    validate_workload(prime)
    validate_workload(measure)
    return prime, measure


async def drive_to_file(rows, args, path):
    lock = Lock()
    with path.open("x", encoding="utf-8", buffering=1) as stream:
        def emit(row):
            with lock:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

        await drive(rows, args.endpoint, args.model, args.workers, args.timeout, False, False, emit)


def summarize_phase(rows, path, args, seconds):
    manifest = {"run_id": args.run_id, "window_start": 0.0, "seconds": seconds,
                "selected_requests": len(rows), "tokenizer_fingerprint": args.tokenizer_fingerprint}
    events = read_jsonl(path)
    summary = summarize(events, manifest)
    completed = {row["id"]: row for row in events if row["type"] == "request_end" and row["status"] == "ok"}
    summary["finish_reasons"] = defaultdict(int)
    for event in completed.values():
        summary["finish_reasons"][event["finish_reason"]] += 1
    summary["cohorts"] = {}
    for cohort in dict.fromkeys(row["cohort"] for row in rows):
        source = [row for row in rows if row["cohort"] == cohort]
        matched = [completed[row["id"]] for row in source if row["id"] in completed]
        summary["cohorts"][cohort] = {"calls": len(source), "sessions": len({row["session_id"] for row in source}),
                                      "completed": len(matched),
                                      "output_tokens": sum(row.get("completion_tokens") or 0 for row in matched),
                                      "latest_finish_s": max((row["end_s"] for row in matched), default=None)}
    return summary


async def run_arm(prime, measured, args, end):
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=args.workers))
    prime_path = args.output / "warmup.jsonl"
    await drive_to_file(prime, args, prime_path)
    warming = summarize_phase(prime, prime_path, args, max(1., max(row["offered_s"] for row in prime) + 1.))
    write_json(args.output / "warmup-summary.json", warming)
    if not warming["quality"]["client_accounting_complete"]:
        raise RuntimeError("Warmup failed; measured phase is invalid")
    client = args.output / "client.jsonl"
    await drive_to_file(measured, args, client)
    summary = summarize_phase(measured, client, args, end)
    write_json(args.output / "summary.json", summary)
    if not summary["quality"]["client_accounting_complete"]:
        raise RuntimeError("The measured cohort did not complete correctly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True, help="Tokenized JSONL from workloads.materialize")
    parser.add_argument("--arm", choices=("steady", "plus33", "plus100"), required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--cap", type=int, choices=(100, 200), help="Label for the externally verified D egress cap")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not args.verify_only and not all((args.endpoint, args.model, args.output, args.run_id, args.cap)):
        parser.error("An actual arm requires endpoint, model, output, run ID and verified cap label")
    if args.workers < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Workers and timeout must be positive")
    plan = read_json(args.plan)
    inputs = {}
    for row in read_jsonl(args.workload):
        if row["id"] in inputs:
            raise ValueError("Duplicate materialized request body")
        inputs[row["id"]] = row["prompt"]
    prime, measured = split_arm(plan, inputs, args.arm)
    args.tokenizer_fingerprint = plan["tokenizer_sha256"]
    receipt = {"schema": "basin-small-client/v1", "arm": args.arm, "run_id": args.run_id, "D_cap_label_Gbps": args.cap,
               "workload_sha256": sha256(args.workload), "plan_sha256": sha256(args.plan),
               "tokenizer_fingerprint": args.tokenizer_fingerprint, "warmup_requests": len(prime),
               "initial_sessions": len(plan["schedule"]["initial_sessions"]), "measured_requests": len(measured),
               "measured_sessions": len({row["session_id"] for row in measured}),
               "surge_window_s": plan["schedule"]["surge_window_s"], "offer_stop_s": plan["schedule"]["offer_stop_s"]}
    if args.verify_only:
        print(json.dumps(receipt))
        return
    args.output = new_directory(args.output)
    write_json(args.output / "manifest.json", receipt)
    asyncio.run(run_arm(prime, measured, args, receipt["offer_stop_s"]))


if __name__ == "__main__":
    main()
