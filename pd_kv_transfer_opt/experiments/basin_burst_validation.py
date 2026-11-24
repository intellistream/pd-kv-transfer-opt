"""Replay frozen Basin surges, including an explicit burst-session stop control."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

from pd_kv_transfer_opt.common import clock_anchor, new_directory, read_json, sha256, write_json
from pd_kv_transfer_opt.experiments.analyze import summarize
from pd_kv_transfer_opt.experiments.basin_validation import split_arm


async def stream_call(row, args, origin, emit):
    """A cancellable HTTP stream; the response context closes on task cancellation."""
    result = {"type": "request_end", "id": row["id"], "session_id": row["session_id"],
              "cohort": row["cohort"], "predecessor": row["predecessor"],
              "offered_s": row["offered_s"], "ready_s": row["ready_s"],
              "dispatch_s": time.monotonic() - origin, "first_s": None,
              "completion_tokens": None, "streamed_token_ids": 0,
              "all_content_chunks_have_token_ids": True, "finish_reason": None,
              "done_signal": False, "status": "error"}
    payload = {"model": args.model, "prompt": row["prompt"], "max_tokens": row["max_tokens"],
               "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}

    def frame(data):
        if data == "[DONE]":
            result["done_signal"] = True
            return
        item = json.loads(data)
        if item.get("error"):
            raise RuntimeError("Server returned a streaming error")
        usage = item.get("usage")
        if usage is not None:
            result["completion_tokens"] = usage.get("completion_tokens")
            result["prompt_tokens"] = usage.get("prompt_tokens")
        for choice in item.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("Only a single completion is supported")
            ids = choice.get("token_ids")
            count = len(ids) if isinstance(ids, list) and ids else None
            if choice.get("text") or count:
                now = time.monotonic() - origin
                if result["first_s"] is None:
                    result["first_s"] = now
                if count is None:
                    result["all_content_chunks_have_token_ids"] = False
                else:
                    result["streamed_token_ids"] += count
                emit({"type": "stream_chunk", "id": row["id"], "at_s": now, "n_tokens": count})
            if choice.get("finish_reason") is not None:
                result["finish_reason"] = choice["finish_reason"]

    try:
        async with args.client.stream("POST", args.endpoint.rstrip("/") + "/v1/completions",
                                      json=payload, headers={"X-Request-Id": row["id"]}) as response:
            response.raise_for_status()
            result["http_status"] = response.status_code
            data = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif not line and data:
                    frame("\n".join(data))
                    data = []
                    if result["done_signal"]:
                        break
            if data and not result["done_signal"]:
                frame("\n".join(data))
        if not result["done_signal"] or result["first_s"] is None or result["finish_reason"] is None:
            raise RuntimeError("Incomplete HTTP-200 stream")
        if type(result["completion_tokens"]) is not int or result["completion_tokens"] < 0:
            raise RuntimeError("Response is missing exact output usage")
        result["status"] = "ok"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["end_s"] = time.monotonic() - origin
    return result


async def execute_rows(rows, args, emit, *, cancel_at=None, submit=stream_call):
    """External root times and causal successors; only burst sessions are cancelled."""
    origin = time.monotonic()
    emit({"type": "client_origin", "monotonic_s": origin, "anchor": clock_anchor()})
    chains = defaultdict(list)
    for row in rows:
        chains[row["session_id"]].append(row)
    slots = asyncio.Semaphore(args.workers)
    results = []

    def record(row, status, **extra):
        value = {"type": "request_end", "id": row["id"], "session_id": row["session_id"],
                 "cohort": row["cohort"], "predecessor": row["predecessor"],
                 "offered_s": row["offered_s"], "ready_s": None, "first_s": None,
                 "end_s": time.monotonic() - origin, "status": status, **extra}
        results.append(value)
        emit(value)

    async def session(chain):
        index, previous, dispatched = 0, None, False
        try:
            for index, source in enumerate(chain):
                dispatched = False
                if previous is not None and previous["status"] != "ok":
                    record(source, "dependency_failed")
                    continue
                ready = max(source["offered_s"], previous["end_s"] + source.get("tool_gap_s", 0)) if previous else source["offered_s"]
                await asyncio.sleep(max(0., origin + ready - time.monotonic()))
                async with slots:
                    dispatched = True
                    emit({"type": "request_start", "id": source["id"], "session_id": source["session_id"],
                          "cohort": source["cohort"], "at_s": time.monotonic() - origin})
                    previous = await submit({**source, "ready_s": ready}, args, origin, emit)
                results.append(previous)
                emit(previous)
        except asyncio.CancelledError:
            record(chain[index], "cancelled_inflight" if dispatched else "cancelled_before_dispatch")
            for source in chain[index + 1:]:
                record(source, "session_terminated")

    tasks = {sid: asyncio.create_task(session(chain)) for sid, chain in chains.items()}

    async def stop_burst():
        await asyncio.sleep(max(0., origin + cancel_at - time.monotonic()))
        pending = [sid for sid, chain in chains.items() if chain[0]["cohort"] == "surge" and not tasks[sid].done()]
        emit({"type": "burst_termination", "scheduled_s": cancel_at,
              "actual_s": time.monotonic() - origin, "sessions": pending,
              "server_abort_verified": False})
        for sid in pending:
            tasks[sid].cancel()

    control = asyncio.create_task(stop_burst()) if cancel_at is not None else None
    try:
        await asyncio.gather(*tasks.values())
        if control:
            await control
    finally:
        for task in [*tasks.values(), *([control] if control else [])]:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), *([control] if control else []), return_exceptions=True)
    return results


def load_arm(plan, workload, arm):
    wanted = {q["id"] for q in plan["requests"]}
    inputs, seen = {}, set()
    opener = gzip.open if workload.suffix == ".gz" else open
    with opener(workload, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in seen:
                raise ValueError("Duplicate materialized request")
            seen.add(row["id"])
            if row["id"] in wanted:
                inputs[row["id"]] = row["prompt"]
    return split_arm(plan, inputs, arm)


def verified_shaping(path, cap):
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if not isinstance(receipt, dict):
        raise ValueError("The D shaping receipt must be a JSON object")
    actual = receipt.get("readback", {})
    expected = cap * 125
    if (receipt.get("aggregate_Gbps", receipt.get("requested_aggregate_Gbps")) != cap
            or not isinstance(actual, dict)
            or set(actual) != set(map(str, range(8)))
            or any(not isinstance(row, dict) or row.get("configured_Mbps") != expected
                   or row.get("current_Mbps") != expected
                   for row in actual.values())):
        raise ValueError("Expected a matching hardware readback for all eight D egress ports")
    return raw


def phase_summary(rows, events, args, seconds, cancellation_expected=False):
    manifest = {"run_id": args.run_id, "window_start": 0., "seconds": seconds,
                "selected_requests": len(rows), "tokenizer_fingerprint": args.tokenizer_fingerprint}
    result = summarize(events, manifest)
    ends = [q for q in events if q["type"] == "request_end"]
    common = [q for q in rows if q["cohort"] != "surge"]
    common_ids = {q["id"] for q in common}
    if common:
        result["common_base"] = summarize([q for q in events if q.get("id") in common_ids],
                                          {**manifest, "selected_requests": len(common)})
    cancelled = {"cancelled_inflight", "cancelled_before_dispatch", "session_terminated"}
    allowed = {"ok"} | (cancelled if cancellation_expected else set())
    acceptable = all(q["status"] in allowed and (q["status"] == "ok" or q["cohort"] == "surge") for q in ends)
    identity = len(ends) == len(rows) and {q["id"] for q in ends} == {q["id"] for q in rows}
    ok_ids = {q["id"] for q in ends if q["status"] == "ok"}
    successful_quality = summarize([q for q in events if q.get("id") in ok_ids],
                                   {**manifest, "selected_requests": len(ok_ids)}) if ok_ids else None
    timing_ok = successful_quality is None or successful_quality["quality"]["client_accounting_complete"]
    result["control_accounting"] = {"complete": identity and acceptable and timing_ok,
        "intentional_termination_counts": dict(Counter(q["status"] for q in ends if q["status"] in cancelled)),
        "client_termination_events": [q for q in events if q["type"] == "burst_termination"],
        "server_abort_and_resource_release_verified": False if cancellation_expected else None,
        "partial_cancelled_output_tokens": None,
        "note": "Cancelled calls are not successful. Whole-cohort token/s stays null when output accounting is incomplete; compare the common base separately."}
    result["finish_reasons"] = dict(Counter(q.get("finish_reason") for q in ends if q["status"] == "ok"))
    return result


async def run_arm(plan, prime, measured, args):
    import httpx

    limits = httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits, trust_env=False) as client:
        args.client = client

        async def phase(rows, path, *, cancel_at=None):
            events = []
            with path.open("x", encoding="utf-8", buffering=1) as stream:
                def emit(value):
                    events.append(value)
                    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                await execute_rows(rows, args, emit, cancel_at=cancel_at)
            return events

        warm = await phase(prime, args.output / "warmup.jsonl")
        warming = phase_summary(prime, warm, args, 1.)
        write_json(args.output / "warmup-summary.json", warming)
        if not warming["quality"]["client_accounting_complete"]:
            raise RuntimeError("Warmup failed; do not start the timed arm")
        start = time.monotonic()
        deadline = plan["schedule"]["terminate_burst_at_s"].get(args.arm)
        events = await phase(measured, args.output / "client.jsonl", cancel_at=deadline)
        summary = phase_summary(measured, events, args, plan["schedule"]["offer_stop_s"], deadline is not None)
        write_json(args.output / "summary.json", summary)
        if not summary["control_accounting"]["complete"]:
            raise RuntimeError("Arm has an unexpected failure or incomplete request accounting")
        await asyncio.sleep(max(0., plan["schedule"]["observe_until_s"] - (time.monotonic() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True, help="Audited JSONL or JSONL.gz; a superset catalog is permitted")
    parser.add_argument("--arm", help="An arm from the plan; smoke plans can select a frozen default")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cap", type=int, choices=(100, 200))
    parser.add_argument("--shaping-readback", type=Path, help="Verified D egress readback for a timed validation arm")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600.)
    parser.add_argument("--short-reference", type=Path, help="Optional target healthy-run receipt; required before claiming paper-equivalent short duration")
    args = parser.parse_args()
    if args.workers < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Workers and timeout must be positive")
    if not args.verify_only and not all((args.endpoint, args.model, args.run_id, args.output, args.cap)):
        parser.error("Actual runs require endpoint/model/run-id/output and externally read-back cap label")
    plan = read_json(args.plan)
    window_validation = plan["experiment_profile"] in ("compact-final-light", "compact-final-fivefold", "compact-final-short")
    if (args.shaping_readback or (window_validation and not args.verify_only)) and not (args.cap and args.shaping_readback):
        parser.error("Basin window timing requires --cap and the matching --shaping-readback")
    shaping = verified_shaping(args.shaping_readback, args.cap) if args.shaping_readback else None
    if args.arm is None:
        args.arm = plan.get("smoke_execution", {}).get("default_arm")
    if args.arm is None:
        parser.error("Validation plans require --arm; only smoke plans have a representative default")
    if args.arm not in plan["schedule"]["arms"]:
        parser.error("This control is not part of the selected frozen experiment")
    prime, measured = load_arm(plan, args.workload, args.arm)
    args.tokenizer_fingerprint = plan["tokenizer_sha256"]
    reference = read_json(args.short_reference) if args.short_reference else None
    if reference is not None:
        if (reference.get("plan_sha256") != sha256(args.plan) or reference.get("model") != args.model
                or not reference.get("healthy_short_duration_qualified")
                or set(reference.get("duration_s_by_session", {})) != set(plan["schedule"]["extra_offers_s"])
                or not all(type(v) in (float, int) and 0 < v <= 300 for v in reference["duration_s_by_session"].values())):
            raise ValueError("The healthy-duration receipt does not qualify these exact burst sessions")
    receipt = {"schema": "basin-burst-client/v1", "experiment_profile": plan["experiment_profile"],
               "scope": "basin_windows_v1" if window_validation else ("smoke" if "smoke_execution" in plan else "validation"),
               "smoke_execution": plan.get("smoke_execution"),
               "arm": args.arm, "run_id": args.run_id, "D_cap_label_Gbps": args.cap,
               "D_shaping_readback_sha256": hashlib.sha256(shaping).hexdigest() if shaping else None,
               "plan_sha256": sha256(args.plan), "workload_sha256": sha256(args.workload),
               "warmup_requests": len(prime), "measured_requests": len(measured),
               "measured_sessions": len({q["session_id"] for q in measured}),
               "cohorts": dict(Counter(q["cohort"] for q in measured)),
               "surge_window_s": plan["schedule"]["surge_window_s"],
               "termination_s": plan["schedule"]["terminate_burst_at_s"].get(args.arm),
               "healthy_short_duration_qualified": reference is not None,
               "short_reference_sha256": sha256(args.short_reference) if args.short_reference else None,
               "server_abort_and_resource_release_verified": False,
               "deployment_invariants": plan["deployment_invariants"], "limits": plan["limits"]}
    if args.verify_only:
        print(json.dumps(receipt, ensure_ascii=False))
        return
    args.output = new_directory(args.output)
    write_json(args.output / "manifest.json", receipt)
    if shaping:
        (args.output / "D-shaping-readback.json").write_bytes(shaping)
    asyncio.run(run_arm(plan, prime, measured, args))


if __name__ == "__main__":
    main()
