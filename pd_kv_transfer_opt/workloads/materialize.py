"""Render audited Terminal-Bench request histories from locally fetched artifacts."""

import argparse
from array import array
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

from pd_kv_transfer_opt.common import read_json, sha256
from pd_kv_transfer_opt.workloads.histories import request_histories


def token_digest(tokens):
    if not isinstance(tokens, list) or any(type(t) is not int or not 0 <= t < 2**32 for t in tokens):
        raise ValueError("Expected flat nonnegative token IDs.")
    packed = array("I", tokens)
    if packed.itemsize != 4:
        raise RuntimeError("Token digest needs a uint32 representation.")
    if sys.byteorder != "little":
        packed.byteswap()
    return hashlib.sha256(packed.tobytes()).hexdigest()


def materialize(spec, traces, model):
    from transformers import AutoTokenizer

    if sha256(model / "tokenizer.json") != spec["tokenizer_sha256"]:
        raise ValueError("Tokenizer does not match the audited workload.")
    receipt = read_json(traces / "download-receipt.json")
    if (receipt["repo"], receipt["revision"]) != (spec["source_repo"], spec["source_revision"]):
        raise ValueError("Trajectory revision does not match the workload.")
    audited = {r["path"]: r["sha256"] for r in receipt["files"]}
    for expected in spec.get("source_files", []):
        if audited.get(expected["path"]) != expected["sha256"]:
            raise ValueError(f"Unexpected source hash for {expected['path']}")
    by_task = defaultdict(dict)
    for row in spec["requests"]:
        by_task[row["task"]][row["step_id"]] = row
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    rendered = {}

    def checked(path):
        rel = path.relative_to(traces).as_posix()
        if audited.get(rel) != sha256(path):
            raise ValueError(f"Trajectory file failed verification: {rel}")
        return read_json(path)

    for config_path in sorted((traces / "trajectories/terminus2").glob("*/config.json")):
        raw_config = read_json(config_path)
        task = Path(raw_config["task"]["path"]).name.removeprefix("terminal-bench__")
        if task not in by_task:
            continue
        checked(config_path)
        trajectory = checked(config_path.parent / "agent/trajectory.json")
        for step, messages in request_histories(trajectory["steps"], "visible"):
            row = by_task[task].get(step)
            if row is None:
                continue
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                                   add_generation_prompt=True, enable_thinking=True, truncation=False)
            if len(tokens) != row["prompt_tokens"] or token_digest(tokens) != row["token_ids_uint32le_sha256"]:
                raise ValueError(f"Rendered token IDs differ for {row['id']}.")
            if len(tokens) + row["max_tokens"] > spec["max_context_tokens"]:
                raise ValueError(f"Context limit exceeded by {row['id']}.")
            if row["id"] in rendered:
                raise ValueError(f"Duplicate source request: {row['id']}")
            rendered[row["id"]] = dict(row, prompt=tokens)
    missing = {row["id"] for row in spec["requests"]} - rendered.keys()
    if missing:
        raise ValueError(f"Missing requests: {sorted(missing)}")
    return [rendered[row["id"]] for row in spec["requests"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixed-output", type=Path, help="Use observed generation budgets from a paired replay fixture.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = read_json(args.spec)
    if args.fixed_output:
        requests = read_json(args.fixed_output)["requests"]
        if requests.keys() != {row["id"] for row in spec["requests"]}:
            raise ValueError("The fixed-output fixture does not cover the same requests.")
        for row in spec["requests"]:
            limit = requests[row["id"]]["max_tokens"]
            if type(limit) is not int or limit < 1:
                raise ValueError(f"Invalid generation budget: {row['id']}")
            row["max_tokens"] = limit
    records = materialize(spec, args.traces, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in records:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({"requests": len(records), "sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
