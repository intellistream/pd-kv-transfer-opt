"""Open-loop streaming completions client with causal session dependencies.

Uses only the Python standard library. Client-side queuing is retained in TTFT.
SSE chunks are not treated as tokens. No retries or prompt content are logged.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
import urllib.request

from pd_kv_transfer_opt.common import clock_anchor, new_directory, read_jsonl, sha256, write_json


def sse_data(response):
    parts = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if parts:
                yield "\n".join(parts)
                parts = []
        elif line.startswith("data:"):
            parts.append(line[5:].lstrip(" "))
    if parts:
        yield "\n".join(parts)


def validate_workload(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        key = row["id"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Unique nonempty string request IDs required")
        if row.get("predecessor") is not None and row["predecessor"] not in seen:
            raise ValueError("Predecessor must occur earlier in the workload")
        seen.add(key)
        offset = row["offered_s"]
        if not math.isfinite(offset) or offset < 0 or not math.isfinite(row.get("tool_gap_s", 0)) or row.get("tool_gap_s", 0) < 0:
            raise ValueError("Finite nonnegative arrival and tool gap required")
        prompt = row["prompt"]
        if not isinstance(prompt, (str, list)) or not prompt:
            raise ValueError("Prompt must be text or a nonempty token-ID list")
        if isinstance(prompt, list) and any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in prompt):
            raise ValueError("Invalid prompt token ID")
        if not isinstance(row["max_tokens"], int) or row["max_tokens"] < 1:
            raise ValueError("Positive integer max_tokens required")


def request_one(row: dict, endpoint: str, model: str, origin: float, timeout: float,
                return_token_ids: bool, ignore_eos: bool, emit) -> dict:
    key = row["id"]
    record = {"type": "request_end", "id": key, "session_id": row.get("session_id"),
              "predecessor": row.get("predecessor"), "offered_s": row["offered_s"],
              "ready_s": row["ready_s"], "dispatch_s": time.monotonic() - origin,
              "first_s": None, "completion_tokens": None, "streamed_token_ids": 0,
              "all_content_chunks_have_token_ids": True, "finish_reason": None,
              "done_signal": False, "status": "error"}
    payload = {"model": model, "prompt": row["prompt"], "max_tokens": row["max_tokens"],
               "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if return_token_ids:
        payload["return_token_ids"] = True
    if ignore_eos:
        payload["ignore_eos"] = True
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/completions",
                  data=json.dumps(payload).encode(), method="POST",
                  headers={"Content-Type": "application/json", "X-Request-Id": key})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            record["http_status"] = response.status
            record["server_request_id"] = response.headers.get("X-Request-Id")
            for data in sse_data(response):
                at = time.monotonic() - origin
                if data == "[DONE]":
                    record["done_signal"] = True
                    break
                frame = json.loads(data)
                if frame.get("error"):
                    raise RuntimeError("Server returned a streaming error frame")
                usage = frame.get("usage")
                if usage is not None:
                    record["completion_tokens"] = usage.get("completion_tokens")
                    record["prompt_tokens"] = usage.get("prompt_tokens")
                for choice in frame.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise ValueError("Only n=1 is supported")
                    token_ids = choice.get("token_ids")
                    has_ids = isinstance(token_ids, list) and len(token_ids) > 0
                    content = bool(choice.get("text")) or has_ids
                    if content:
                        if record["first_s"] is None:
                            record["first_s"] = at
                        count = len(token_ids) if has_ids else None
                        if count is None:
                            record["all_content_chunks_have_token_ids"] = False
                        else:
                            record["streamed_token_ids"] += count
                        emit({"type": "stream_chunk", "id": key, "at_s": at, "n_tokens": count})
                    if choice.get("finish_reason") is not None:
                        record["finish_reason"] = choice["finish_reason"]
        if record["first_s"] is None or record["finish_reason"] is None or not record["done_signal"]:
            raise RuntimeError("Incomplete or empty HTTP-200 stream")
        record["status"] = "ok"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["end_s"] = time.monotonic() - origin
    return record


async def drive(rows, endpoint, model, workers, timeout, return_token_ids, ignore_eos, emit):
    origin = time.monotonic()
    emit({"type": "client_origin", "monotonic_s": origin, "anchor": clock_anchor()})
    semaphore = asyncio.Semaphore(workers)
    tasks = {}

    async def one(row):
        await asyncio.sleep(max(0, origin + row["offered_s"] - time.monotonic()))
        predecessor = row.get("predecessor")
        ready = row["offered_s"]
        if predecessor is not None:
            previous = await tasks[predecessor]
            if previous["status"] != "ok":
                record = {"type": "request_end", "id": row["id"], "status": "dependency_failed",
                          "offered_s": row["offered_s"], "ready_s": None,
                          "end_s": time.monotonic() - origin, "first_s": None}
                emit(record)
                return record
            ready = max(ready, previous["end_s"] + row.get("tool_gap_s", 0))
            await asyncio.sleep(max(0, origin + ready - time.monotonic()))
        async with semaphore:
            record = await asyncio.to_thread(request_one, {**row, "ready_s": ready}, endpoint,
                          model, origin, timeout, return_token_ids, ignore_eos, emit)
        emit(record)
        return record

    for row in rows:
        tasks[row["id"]] = asyncio.create_task(one(row))
    await asyncio.gather(*tasks.values())
    return time.monotonic() - origin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--endpoint", required=True, help="Proxy base URL, without /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--window-start", type=float, default=60)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--return-token-ids", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true", help="Fixed-length diagnostic workloads only")
    parser.add_argument("--tokenizer-fingerprint", default=None)
    args = parser.parse_args()
    if (args.workers < 1 or not all(math.isfinite(x) for x in (args.timeout, args.seconds, args.window_start))
            or min(args.timeout, args.seconds) <= 0 or args.window_start < 0):
        parser.error("Invalid load parameters")
    all_rows = read_jsonl(args.workload)
    validate_workload(all_rows)
    rows = [row for row in all_rows if row["offered_s"] < args.window_start + args.seconds]
    validate_workload(rows)
    output = new_directory(args.output)
    write_json(output / "manifest.json", {"schema": "rdma-lab-client/v1", **vars(args),
               "anchor": clock_anchor(), "workload_sha256": sha256(args.workload),
               "selected_requests": len(rows), "excluded_after_window": len(all_rows) - len(rows),
               "arrival_semantics": "external offered time; successors wait for full predecessor completion"})
    # Serialize concurrent worker writes. The lock never surrounds network calls.
    import threading
    lock = threading.Lock()
    with (output / "client.jsonl").open("x", encoding="utf-8") as stream:
        def emit(row):
            with lock:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()

        async def run():
            loop = asyncio.get_running_loop()
            loop.set_default_executor(ThreadPoolExecutor(max_workers=args.workers))
            return await drive(rows, args.endpoint, args.model, args.workers, args.timeout,
                               args.return_token_ids, args.ignore_eos, emit)
        duration = asyncio.run(run())
    write_json(output / "client-end.json", {"anchor": clock_anchor(), "drain_included_s": duration})


if __name__ == "__main__":
    main()
