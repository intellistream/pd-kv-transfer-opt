"""Summarize fixed windows and finite cohorts without counting SSE as tokens."""

from __future__ import annotations

import argparse
from collections import Counter
import math

from pd_kv_transfer_opt.common import percentile, read_json, read_jsonl, write_json, output_path


def summarize(rows: list[dict], manifest: dict, ttft_slo_s=None, e2e_slo_s=None) -> dict:
    start = manifest["window_start"]
    stop = start + manifest["seconds"]
    ends = [row for row in rows if row.get("type") == "request_end"]
    chunks = [row for row in rows if row.get("type") == "stream_chunk"]
    successful = [row for row in ends if row["status"] == "ok"]
    errors = []
    if not ends:
        errors.append("No requests selected")
    if len({row["id"] for row in ends}) != len(ends):
        errors.append("Duplicate request end records")
    if len(ends) != manifest["selected_requests"]:
        errors.append("Missing request end records")
    if len(successful) != len(ends):
        errors.append("Failed or dependency-failed requests")
    timings = []
    e2e = []
    client_queue = []
    for row in successful:
        if not row["offered_s"] <= row["ready_s"] <= row["dispatch_s"] <= row["first_s"] <= row["end_s"]:
            errors.append(f"Causal timestamp violation: {row['id']}")
        timings.append(row["first_s"] - row["ready_s"])
        e2e.append(row["end_s"] - row["ready_s"])
        client_queue.append(row["dispatch_s"] - row["ready_s"])
        usage = row.get("completion_tokens")
        if not isinstance(usage, int) or usage < 0:
            errors.append(f"No exact completion usage: {row['id']}")
        if row.get("all_content_chunks_have_token_ids") and usage != row.get("streamed_token_ids"):
            errors.append(f"Usage/token-ID count mismatch: {row['id']}")
    window_chunks = [row for row in chunks if start <= row["at_s"] < stop]
    all_ids = bool(successful) and all(row.get("all_content_chunks_have_token_ids") for row in ends)
    all_ids = all_ids and all(isinstance(row.get("n_tokens"), int) for row in chunks)
    window_completed = [row for row in successful if start <= row["end_s"] < stop]
    usage_total = sum(row["completion_tokens"] for row in window_completed
                      if isinstance(row.get("completion_tokens"), int))
    duration = max([row["end_s"] for row in ends] + [0.0])
    offered_cohort = [row for row in ends if start <= row["offered_s"] < stop]
    ready_cohort = [row for row in successful if start <= row["ready_s"] < stop]
    passed = completed_passed = None
    if ttft_slo_s is not None and e2e_slo_s is not None:
        def meets_slo(row):
            return (row["status"] == "ok" and row["first_s"] - row["ready_s"] <= ttft_slo_s
                    and row["end_s"] - row["ready_s"] <= e2e_slo_s)
        passed = sum(meets_slo(row) for row in offered_cohort)
        completed_passed = sum(meets_slo(row) for row in window_completed)
    def distribution(values):
        return {"n": len(values), **{f"p{int(p*100)}_s": percentile(values, p) for p in (.5, .9, .95, .99)}}
    return {"schema": "rdma-lab-summary/v1", "run_id": manifest["run_id"],
            "counts": dict(Counter(row["status"] for row in ends)),
            "quality": {"errors": errors, "client_accounting_complete": not errors,
                        "exact_chunk_token_counts": all_ids,
                        "tokenizer_fingerprint_present": bool(manifest.get("tokenizer_fingerprint")),
                        "host_device_network_instrumentation": "requires separate trace audit"},
            "window": {"start_s": start, "end_s": stop,
                       "client_delivered_token_ids": sum(row["n_tokens"] for row in window_chunks) if all_ids else None,
                       "client_delivered_tokens_s": sum(row["n_tokens"] for row in window_chunks) / manifest["seconds"] if all_ids else None,
                       "completed_requests_s": len(window_completed) / manifest["seconds"],
                       "completed_request_tokens_s_boundary_biased": usage_total / manifest["seconds"],
                       "slo_goodput_requests_s": completed_passed / manifest["seconds"] if completed_passed is not None else None,
                       "offered_cohort_slo_attainment": passed / len(offered_cohort) if passed is not None and offered_cohort else None,
                       "ready_cohort_successful_ttft": distribution([row["first_s"] - row["ready_s"] for row in ready_cohort]),
                       "offered_cohort_requests": len(offered_cohort),
                       "all_requests_finished_before_window_end": duration < stop,
                       "slo_definition": {"ttft_s": ttft_slo_s, "e2e_s": e2e_slo_s}},
            "finite_cohort": {"ttft_from_ready": distribution(timings), "e2e_from_ready": distribution(e2e),
                              "first_from_offered": distribution([row["first_s"] - row["offered_s"] for row in successful]),
                              "client_queue": distribution(client_queue), "drain_included_s": duration,
                              "completed_requests_per_drain_s": len(successful) / duration if duration else None},
            "note": "Client TTFT includes client queue. Unknown token counts stay null. p99 is nearest-rank."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("--ttft-slo", type=float)
    parser.add_argument("--e2e-slo", type=float)
    args = parser.parse_args()
    if any(value is not None and (not math.isfinite(value) or value <= 0) for value in (args.ttft_slo, args.e2e_slo)):
        parser.error("SLO values must be positive")
    from pathlib import Path
    root = Path(args.run_directory)
    result = summarize(read_jsonl(root / "client.jsonl"), read_json(root / "manifest.json"), args.ttft_slo, args.e2e_slo)
    target = output_path(root / "summary.json")
    if target.exists():
        raise FileExistsError(target)
    write_json(target, result)


if __name__ == "__main__":
    main()
