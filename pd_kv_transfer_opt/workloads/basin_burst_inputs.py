"""Materialize the frozen public source catalog with exact per-prompt hash checks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.workloads.basin_small import source_paths
from pd_kv_transfer_opt.workloads.histories import request_histories
from pd_kv_transfer_opt.workloads.materialize import token_digest


def prompt_encoder(tokenizer_dir, expected_sha256):
    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    if sha256(tokenizer_dir / "tokenizer.json") != expected_sha256:
        raise ValueError("Wrong tokenizer file")
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def reject(message):
        raise TemplateError(message)

    environment.globals["raise_exception"] = reject
    environment.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    template_path = tokenizer_dir / "chat_template.jinja"
    template = environment.from_string(template_path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))

    def encode(messages):
        rendered = template.render(messages=messages, add_generation_prompt=True, enable_thinking=True)
        return tokenizer.encode(rendered, add_special_tokens=False).ids

    return encode


def materialize(catalog, source_archive, tokenizer_dir, output):
    if output.exists():
        raise FileExistsError(output)
    encode = prompt_encoder(tokenizer_dir, catalog["tokenizer_sha256"])
    expected = defaultdict(dict)
    for row in catalog["requests"]:
        expected[row["task"]][row["step_id"]] = row
    audited = {row["path"]: row["sha256"] for row in catalog["source_files"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    verified, followers = set(), 0
    try:
        with ZipFile(source_archive) as archive, gzip.open(output, "wt", encoding="utf-8", compresslevel=4) as target:
            def checked(name):
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != audited[name]:
                    raise ValueError(f"Bad source archive member: {name}")
                return json.loads(data)

            for task, calls in expected.items():
                config_path, trajectory_path, result_path = source_paths(task)
                checked(config_path)
                checked(result_path)
                trajectory = checked(trajectory_path)
                previous = None
                for step, messages in request_histories(trajectory["steps"], "visible"):
                    if step not in calls:
                        continue
                    row = calls[step]
                    tokens = encode(messages)
                    if len(tokens) != row["prompt_tokens"] or token_digest(tokens) != row["token_ids_uint32le_sha256"]:
                        raise ValueError(f"Tokenized prompt differs from canonical receipt: {row['id']}")
                    if previous is not None:
                        common = 0
                        for left, right in zip(previous, tokens):
                            if left != right:
                                break
                            common += 1
                        if common != row["source_lcp_tokens"]:
                            raise ValueError(f"Prefix identity differs: {row['id']}")
                        followers += 1
                    if row["id"] in verified or len(tokens) + row["max_tokens"] > catalog["max_context_tokens"]:
                        raise ValueError("Duplicate request or excessive context")
                    target.write(json.dumps(dict(row, prompt=tokens), separators=(",", ":")) + "\n")
                    verified.add(row["id"])
                    previous = tokens
        if verified != {row["id"] for row in catalog["requests"]}:
            raise ValueError("Missing source requests")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {"schema": "basin-burst-input-receipt/v1", "requests": len(verified), "adjacent_prefixes": followers,
            "output_sha256": sha256(output), "output_bytes": output.stat().st_size,
            "source_archive_sha256": sha256(source_archive), "tokenizer_sha256": catalog["tokenizer_sha256"],
            "chat_template_sha256": sha256(tokenizer_dir / "chat_template.jinja"),
            "method": "Local tokenizers + canonical Jinja template, checked against every previously audited prompt hash and LCP; no truncation or padding"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("configs/basin-small-2p2d.json"))
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    result = materialize(read_json(args.catalog), args.source_archive, args.tokenizer, args.output)
    result["catalog_sha256"] = sha256(args.catalog)
    write_json(args.receipt, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
