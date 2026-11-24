"""Rebuild the frozen r43 and r69 inputs from the checked Terminal-Bench archive."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.experiments.loadgen import validate_workload
from pd_kv_transfer_opt.workloads.basin_burst_inputs import prompt_encoder
from pd_kv_transfer_opt.workloads.basin_small import source_paths
from pd_kv_transfer_opt.workloads.histories import request_histories
from pd_kv_transfer_opt.workloads.materialize import token_digest


SPECS = {
    "r43": "configs/workload-r42-natural-sustained.json",
    "r69": "results/2026-11-22-terminal-source-restore-short-guard/workload.json",
}
FIELDS = ("id", "session_id", "predecessor", "offered_s", "tool_gap_s", "prompt_tokens", "max_tokens")


def materialize(spec, archive, audited_members, encode):
    by_task = defaultdict(dict)
    for row in spec["request_sources"]:
        name, step = row["id"].rsplit(":", 1)
        task = name.removeprefix("natural-").removeprefix("sustained-")
        by_task[task][int(step)] = row

    records = {}
    for task, planned in by_task.items():
        _, trajectory_path, _ = source_paths(task)
        source = archive.read(trajectory_path)
        if hashlib.sha256(source).hexdigest() != audited_members.get(trajectory_path):
            raise ValueError(f"Unexpected trajectory hash for {task}")
        for step, messages in request_histories(json.loads(source)["steps"], "visible"):
            if step not in planned:
                continue
            source_row = planned[step]
            tokens = encode(messages)
            if (len(tokens) != source_row["prompt_tokens"] or token_digest(tokens) != source_row["prompt_sha256"]
                    or len(tokens) + source_row["max_tokens"] > 262144):
                raise ValueError(f"Frozen prompt or context differs for {source_row['id']}")
            records[source_row["id"]] = {**{key: source_row[key] for key in FIELDS}, "prompt": tokens}

    ordered = [records[row["id"]] for row in spec["request_sources"]]
    if len(ordered) != spec["requests"] or len({row["session_id"] for row in ordered}) != spec["sessions"]:
        raise ValueError("Request or session count differs from the frozen plan")
    validate_workload(ordered)
    return ordered


def prepare(source, tokenizer, output):
    catalog = read_json("configs/basin-windows-v1/catalog.json")
    audited = {row["path"]: row["sha256"] for row in catalog["source_files"]}
    encode = prompt_encoder(tokenizer, catalog["tokenizer_sha256"])
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory, ZipFile(source) as archive:
        staging = Path(directory) / "inputs"
        staging.mkdir()
        evidence = {}
        for name, spec_path in SPECS.items():
            spec = read_json(spec_path)
            if (spec["source_repo"], spec["source_revision"], spec["tokenizer_fingerprint"]) != (
                    catalog["source_repo"], catalog["source_revision"], catalog["tokenizer_sha256"]):
                raise ValueError(f"Source identity differs for {name}")
            records = materialize(spec, archive, audited, encode)
            path = staging / f"{name}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in records:
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            del records
            digest = sha256(path)
            if spec.get("workload_sha256", digest) != digest:
                raise ValueError(f"Materialized file differs from the published replay for {name}")
            evidence[name] = {"file": path.name, "sha256": digest, "spec_sha256": sha256(spec_path),
                              "requests": spec["requests"], "sessions": spec["sessions"]}
        receipt = {"schema": "native-replay-inputs/v1", "source_archive_sha256": sha256(source),
                   "tokenizer_sha256": catalog["tokenizer_sha256"], "workloads": evidence}
        write_json(staging / "manifest.json", receipt)
        staging.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/basin-windows-v1/source.zip"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/basin-windows-v1/tokenizer"))
    parser.add_argument("--output", type=Path, default=Path("data/native-npu-inputs"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.tokenizer, args.output)))


if __name__ == "__main__":
    main()
