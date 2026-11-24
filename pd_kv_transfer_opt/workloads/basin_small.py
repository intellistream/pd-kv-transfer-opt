"""Freeze a small hardware replay of Basin's Terminal-Bench arrival experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path, PurePosixPath
from random import Random
from statistics import median

from pd_kv_transfer_opt.common import percentile, read_json, read_jsonl, sha256, write_json
from pd_kv_transfer_opt.workloads.fetch_trajectories import REPO, REVISION
from pd_kv_transfer_opt.workloads.histories import recover_tool_intervals


TASKS = (
    "build-cython-ext", "build-pmars", "build-pov-ray", "compile-compcert", "crack-7z-hash",
    "custom-memory-heap-crash", "dna-insert", "financial-document-processor", "fix-ocaml-gc",
    "git-leak-recovery", "headless-terminal", "hf-model-inference", "kv-store-grpc", "mailman",
    "make-doom-for-mips", "mteb-leaderboard", "password-recovery", "polyglot-rust-c", "portfolio-optimization",
    "pypi-server", "query-optimize", "reshard-c4-data", "schemelike-metacircular-eval", "sqlite-with-gcov",
)
TOKENIZER = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
CONTEXT = 262_144
SCALE = 15
PAPER = {
    "paper": "https://arxiv.org/html/2611.02543v1#S3",
    "source_sessions": 216, "source_requests": 7108, "initially_active_midtrajectory": 42,
    "prompt_p50_tokens": 25_600, "prompt_p90_tokens": 77_000, "adjacent_prompt_reuse_fraction": .982,
    "output_p50_tokens": 160, "tool_gap_p50_s": .42, "base_sessions_per_minute": 3.3,
    "surge_starts_s": 900, "surge_ends_s": 1500, "observed_post_s": 3600,
    "added_arrival_fractions": [0, .33, 1.0],
    "engine": "SGLang with NIXL, Qwen3.6-35B-A3B FP8, H100 NVL 1P1D",
    "P_prefix_capacity_tokens": 1_090_000, "D_KV_capacity_tokens": 1_660_000,
}


def source_paths(task):
    directory = f"trajectories/terminus2/{task[:16].rstrip('-')}"
    return tuple(f"{directory}/{name}" for name in ("config.json", "agent/trajectory.json", "result.json"))


def fetch_spec():
    return {"source_repo": REPO, "source_revision": REVISION,
            "source_files": [{"path": p} for task in TASKS for p in source_paths(task)]}


def poisson(rate, start, end, seed):
    rng = Random(seed)
    while True:
        start += rng.expovariate(rate / 60)
        if start >= end:
            return
        yield round(start, 6)


def schedule(sessions):
    start, end = (PAPER[key] / SCALE for key in ("surge_starts_s", "surge_ends_s"))
    stop = end + PAPER["observed_post_s"] / SCALE
    rate = PAPER["base_sessions_per_minute"]
    base = list(poisson(rate, 0, stop, 216))
    extra = list(poisson(rate, start, end, 397))
    marker = Random(219)
    low_surge = [i for i in range(len(extra)) if marker.random() < .33]
    warm_count = round(len(sessions) * PAPER["initially_active_midtrajectory"] / PAPER["source_sessions"])
    if len(base) + len(extra) + warm_count != len(sessions) or not 0 < len(low_surge) < len(extra):
        raise ValueError("Unexpected realization of the frozen three arrival plans")
    order = list(sessions)
    rng = Random(216)
    rng.shuffle(order)
    warm = {}
    for position, name in enumerate(order[:warm_count]):
        chain = sessions[name]
        cut = len(chain) // 2
        warm[name] = {"prime_turns": cut, "prime_start_s": float(position),
                      "initial_remaining_tool_s": round(rng.random() * chain[cut]["tool_gap_s"], 3)}
    base_offers = dict(zip(order[warm_count:warm_count + len(base)], base, strict=True))
    extra_offers = dict(zip(order[warm_count + len(base):], extra, strict=True))
    return {"surge_window_s": [start, end], "offer_stop_s": stop,
            "base_sessions_per_minute": rate, "clock_seeds": {"base": 216, "extra": 397, "thinning": 219},
            "initial_sessions": warm, "base_offers_s": base_offers, "extra_offers_s": extra_offers,
            "arms": {"steady": [], "plus33": [list(extra_offers)[i] for i in low_surge], "plus100": list(extra_offers)}}


def freeze(ledger_path, completions_path, traces):
    source = read_json(completions_path)
    receipt = read_json(traces / "download-receipt.json")
    if source["source_revision"] != REVISION or (receipt["repo"], receipt["revision"]) != (REPO, REVISION):
        raise ValueError("A frozen validation case requires the original public source revision")
    indexed = {row["path"]: row for row in receipt["files"]}
    files = []
    records = defaultdict(list)
    for row in read_jsonl(ledger_path):
        if row["task"] in TASKS:
            records[row["task"]].append(row)
    if records.keys() != set(TASKS):
        raise ValueError("The source ledger is missing a selected full task")
    raw_gaps, sessions = {}, {}
    for task in TASKS:
        paths = source_paths(task)
        for rel in paths:
            expected = indexed[rel]
            if sha256(traces / rel) != expected["sha256"]:
                raise ValueError(f"Bad source file hash: {rel}")
            files.append({key: expected[key] for key in ("path", "sha256", "bytes")})
        task_config, trajectory, trial = (read_json(traces / path) for path in paths)
        if PurePosixPath(task_config["task"]["path"]).name.removeprefix("terminal-bench__") != task:
            raise ValueError(f"Dataset directory resolves to another task: {task}")
        chain = sorted(records[task], key=lambda row: row["step_id"])
        observed_steps = [step["step_id"] for step in trajectory["steps"] if step["source"] == "agent"]
        if [row["step_id"] for row in chain] != observed_steps:
            raise ValueError(f"Selected trajectory is not complete: {task}")
        clock = recover_tool_intervals({"task": task, "turns": chain}, trajectory, trial)
        raw_gaps.update({key: row["gap_ms"] for key, row in clock["intervals_by_child"].items()})
        calls = []
        for i, row in enumerate(chain):
            prior = chain[i - 1]["id"] if i else None
            length = int(source["source_completion_tokens"][row["id"]])
            budget = max(8192, length)
            if row["truncated"] or row["predecessor"] != prior or not 0 < row["prompt_tokens"] <= CONTEXT - budget:
                raise ValueError(f"Unusable original causal call: {row['id']}")
            calls.append({**{k: row[k] for k in ("id", "task", "step_id", "prompt_tokens", "token_ids_uint32le_sha256")},
                          "session_id": task, "predecessor": prior, "max_tokens": budget,
                          "source_completion_tokens": length, "source_lcp_tokens": row["source_prompt_lcp_to_previous"],
                          "offered_s": 0.0})
        sessions[task] = calls

    scale = PAPER["tool_gap_p50_s"] * 1000 / median(raw_gaps.values())
    requests = [row for rows in sessions.values() for row in rows]
    for row in requests:
        raw = raw_gaps.get(row["id"], 0)
        row["source_tool_gap_ms"] = raw
        row["tool_gap_s"] = round(raw * scale) / 1000
    lengths = [row["prompt_tokens"] for row in requests]
    followers = [row for row in requests if row["predecessor"]]
    return {
        "schema": "basin-small-replay/v1", "paper_reference": PAPER, "phase_scale": SCALE,
        "source_repo": REPO, "source_revision": REVISION, "max_context_tokens": CONTEXT, "tokenizer_sha256": TOKENIZER,
        "source_files": files, "selection_ledger_sha256": sha256(ledger_path),
        "source_completions_sha256": sha256(completions_path), "ignore_eos": False,
        "tool_clock": {"origin": "Harbor step timestamps minus per-call API time", "scale_to_Basin_p50": scale,
                       "source_p50_s": median(raw_gaps.values()) / 1000,
                       "scaled_p50_s": median(row["tool_gap_s"] for row in followers),
                       "scaled_p95_s": percentile([row["tool_gap_s"] for row in followers], .95)},
        "profile": {"sessions": len(sessions), "requests_before_warm_split": len(requests),
                    "mean_calls": len(requests) / len(sessions), "prompt_p50_tokens": percentile(lengths, .5),
                    "prompt_p90_tokens": percentile(lengths, .9),
                    "adjacent_prompt_reuse_fraction": sum(row["source_lcp_tokens"] for row in followers) /
                                                      sum(row["prompt_tokens"] for row in followers),
                    "reference_source_output_p50_tokens": percentile([row["source_completion_tokens"] for row in requests], .5)},
        "schedule": schedule(sessions), "requests": requests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--source-completions", type=Path)
    parser.add_argument("--traces", type=Path)
    args = parser.parse_args()
    paths = (args.ledger, args.source_completions, args.traces)
    if any(paths) and not all(paths):
        parser.error("Freeze requires the ledger, source completions and downloaded trajectories")
    write_json(args.output, freeze(*paths) if all(paths) else fetch_spec())


if __name__ == "__main__":
    main()
