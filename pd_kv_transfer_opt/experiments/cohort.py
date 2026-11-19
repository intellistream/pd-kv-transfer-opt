"""Submit one fixed long-prefix cohort and verify each response before reporting it."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from pd_kv_transfer_opt.common import read_jsonl, sha256, write_json
from pd_kv_transfer_opt.experiments import loadgen
from pd_kv_transfer_opt.experiments.analyze import summarize
from pd_kv_transfer_opt.experiments.rdma_full_protocol import CASES, OUTPUT_TOKENS, PREFIXES, PROMPT_LENGTHS, offered_rate, request_id
from pd_kv_transfer_opt.workloads.materialize import token_digest


TOKEN_HASHES = {
    PREFIXES[0]: "9ffa4e0cbe5004210049a210726c7b6bd1aa8a03976ac29fc29affa3577b2c83",
    PREFIXES[1]: "48bd456797b884a794dfa81c021d05bba5ef0138f893681aad378d9504809130",
}


def fixed_sources(path):
    by_id = {row["id"]: row for row in read_jsonl(path)}
    sources = [by_id[key] for key in PREFIXES]
    for row, length in zip(sources, PROMPT_LENGTHS, strict=True):
        if len(row["prompt"]) != length or row["prompt_tokens"] != length or token_digest(row["prompt"]) != TOKEN_HASHES[row["id"]]:
            raise ValueError(f"Wrong tokens for {row['id']}. Materialize the audited pair first.")
    return sources


def run_cohort(case, run_id, source_path, endpoint, model, directory):
    _, cap, count = next(row for row in CASES if row[0] == case)
    sources = fixed_sources(source_path)
    rate = offered_rate(case)
    import msgspec

    previous = loadgen.json
    loadgen.json = SimpleNamespace(dumps=lambda value: msgspec.json.encode(value).decode(), loads=msgspec.json.decode)
    directory.mkdir(exist_ok=False, parents=True)
    work = [
        {"id": request_id(case, i), "source_id": sources[i % 2]["id"],
         "prompt": sources[i % 2]["prompt"], "prompt_tokens": sources[i % 2]["prompt_tokens"],
         "offered_s": i / rate, "max_tokens": OUTPUT_TOKENS}
        for i in range(count)
    ]
    manifest = {"run_id": run_id, "case": case, "workload_sha256": sha256(source_path),
                "aggregate_D_egress_cap_Gbps": cap, "selected_requests": count, "window_start": 0,
                "prefix_ids": PREFIXES, "prompt_lengths": PROMPT_LENGTHS, "offered_rate": rate,
                "output_tokens": OUTPUT_TOKENS, "ignore_eos": True, "background_traffic": False,
                "profiler_enabled": case.startswith("profile-"),
                "request_sources": [{k: v for k, v in row.items() if k != "prompt"} for row in work]}
    write_json(directory / "manifest.json", manifest)
    rows = []
    lock = threading.Lock()
    try:
        with (directory / "client.jsonl").open("x", encoding="utf-8", buffering=1) as output:
            def emit(row):
                with lock:
                    rows.append(row)
                    output.write(json.dumps(row, separators=(",", ":")) + "\n")

            async def run():
                asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=count))
                return await loadgen.drive(work, endpoint, model, count, 360, True, True, emit)

            manifest["seconds"] = asyncio.run(run())
    finally:
        loadgen.json = previous
    write_json(directory / "manifest.json", manifest)
    summary = summarize(rows, manifest)
    summary["fixed_output_deviations"] = [
        r["id"] for r in rows if r["type"] == "request_end" and r.get("completion_tokens") != OUTPUT_TOKENS
    ]
    write_json(directory / "analysis.json", summary)
    if summary["counts"] != {"ok": count} or summary["quality"]["errors"] or summary["fixed_output_deviations"]:
        raise RuntimeError(f"Invalid cohort {case}. See {directory / 'analysis.json'}.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[name for name, _, _ in CASES], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_cohort(args.case, args.run_id, args.workload, args.endpoint, args.model, args.output)
    print(json.dumps({"case": args.case, "counts": summary["counts"], "finite_cohort": summary["finite_cohort"]}))


if __name__ == "__main__":
    main()
