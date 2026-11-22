"""Materialize a native shared-prefix workload from genuine NarrativeQA questions."""

import argparse
from array import array
from collections import defaultdict
import gzip
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path
from random import Random
import sys

from pd_kv_transfer_opt.common import read_json, sha256 as file_sha256, write_json
from pd_kv_transfer_opt.experiments.loadgen import validate_workload
from pd_kv_transfer_opt.workloads.handoff_cohort import prompt_hash


TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
MAX_CONTEXT = 262144
MAX_OUTPUT = 8192
BLOCK = 1024
SEED = 20261121


def common_length(first, second, limit):
    return next((i for i, (a, b) in enumerate(islice(zip(first, second), limit)) if a != b), limit)


def token_bytes(ids):
    data = array("I", ids)
    if data.itemsize != 4:
        raise ValueError("Expected 32-bit token IDs")
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def tokenize_books(data, cohort, tokenizer_directory, questions, enable_thinking):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    model_path = tokenizer_directory / "tokenizer.json"
    if file_sha256(model_path) != TOKENIZER_SHA256:
        raise ValueError("The tokenizer differs from the audited Qwen3.5 deployment")
    model = Tokenizer.from_file(str(model_path))
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template_path = tokenizer_directory / "chat_template.jinja"
    template = environment.from_string(template_path.read_text(encoding="utf-8"))
    books, references, origins = {}, {}, []
    chosen = set(cohort["documents"])
    with gzip.open(data, "rt", encoding="utf-8") as stream:
        for line in stream:
            document = json.loads(line)
            key = document["document_id"]
            if key not in chosen:
                continue
            selected = document["questions"][:questions]
            if len({item["question"] for item in selected}) != questions:
                raise ValueError(f"Missing distinct original questions for {key}")
            context = document["context"]
            prompts = []
            for item in selected:
                content = ("Read the following story and answer the question. Give a brief answer based on the story.\n\n"
                           f"{context}\n\nQuestion: {item['question']}")
                rendered = template.render(messages=[{"role": "user", "content": content}],
                                           add_generation_prompt=True, enable_thinking=enable_thinking)
                ids = model.encode(rendered, add_special_tokens=False).ids
                if len(ids) + MAX_OUTPUT > MAX_CONTEXT:
                    raise ValueError(f"Book and output budget exceed the model context for {key}")
                prompts.append(ids)
            first = prompts[0]
            shared = len(first)
            for tokens in prompts[1:]:
                shared = common_length(first, tokens, min(shared, len(tokens)))
            aligned = shared // BLOCK * BLOCK
            if aligned // BLOCK != document["prefix_blocks"] or prompt_hash(first[:aligned]) != document["prompt_prefix_sha256"]:
                raise ValueError(f"The questioned document prefix changed for {key}")
            books[key] = {"prefix": first[:aligned], "suffixes": [tokens[aligned:] for tokens in prompts]}
            references[key] = [min(len(model.encode(item[answer], add_special_tokens=False).ids)
                                   for answer in ("answer1", "answer2")) for item in selected]
            origins.append({"document_id": key, "archive_member": document["archive_member"],
                            "raw_document_sha256": document["raw_document_sha256"],
                            "prefix_sha256": document["prompt_prefix_sha256"], "prefix_tokens": aligned})
            print(f"Verified document {len(origins)}/{len(chosen)} with {aligned} shared tokens", flush=True)
    if set(books) != chosen:
        raise ValueError("The documents do not cover the selected cohort")
    return books, references, origins, file_sha256(template_path)


def offer_plan(books, references, waves, questions_per_wave, pressure_start, period):
    order = Random(SEED)
    fingerprints = {key: sha256(token_bytes(book["prefix"])) for key, book in books.items()}
    offered = []
    for wave in range(waves + 1):
        documents = sorted(books)
        order.shuffle(documents)
        if wave:
            first_question = 1 + (wave - 1) * questions_per_wave
            wave_requests = [(doc, i) for i in range(first_question, first_question + questions_per_wave) for doc in documents]
            start, rate = pressure_start + (wave - 1) * period, len(wave_requests) / period
        else:
            wave_requests = [(doc, 0) for doc in documents]
            start, rate = 0, 8
        for index, (doc, question) in enumerate(wave_requests):
            key = f"nqa-{doc}-{question:02d}"
            book = books[doc]
            suffix = book["suffixes"][question]
            digest = fingerprints[doc].copy()
            digest.update(token_bytes(suffix))
            offered.append({"id": key, "session_id": key, "predecessor": None, "offered_s": start + index / rate,
                            "tool_gap_s": 0, "max_tokens": MAX_OUTPUT, "prompt_tokens": len(book["prefix"]) + len(suffix),
                            "prompt_sha256": digest.hexdigest(), "document_id": doc, "source_question_index": question,
                            "reference_completion_tokens": references[doc][question], "wave": wave})
    return offered


def build(source, tokenizer, output, blocks, waves, questions_per_wave, pressure_start, period, enable_thinking):
    metadata = read_json(source / "residency-cohorts.json")
    cohort, = (row for row in metadata["cohorts"] if row["prefix_blocks"] == blocks)
    if waves < 1 or questions_per_wave < 1 or period <= 0 or pressure_start <= len(cohort["documents"]) / 8:
        raise ValueError("Invalid request windows")
    data = source / "selected-documents.jsonl.gz"
    books, references, origins, template = tokenize_books(data, cohort, tokenizer, 1 + waves * questions_per_wave, enable_thinking)
    specs = offer_plan(books, references, waves, questions_per_wave, pressure_start, period)
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "document-prompts.json.gz"
    with payload.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(json.dumps(books, separators=(",", ":")).encode("utf-8"))
    manifest = {"workload_type": "narrativeqa_document_questions", "source_repo": "google-deepmind/narrativeqa",
                "source_revision": metadata["source_revision"], "source_workload_sha256": file_sha256(data),
                "document_archive_repository": metadata["archive_repository"], "document_archive_revision": metadata["archive_revision"],
                "document_sources": origins, "shared_prefix_blocks": blocks, "block_tokens": BLOCK,
                "tokenizer_fingerprint": TOKENIZER_SHA256, "chat_template_sha256": template,
                "payload_sha256": file_sha256(payload), "requests": len(specs), "sessions": len(specs),
                "client_inflight_cap": 96, "last_session_start_s": specs[-1]["offered_s"],
                "tool_gap_policy": "independent source questions on shared real documents", "max_tokens_policy": "8192 with native EOS",
                "temperature": 0, "ignore_eos": False, "enable_thinking": enable_thinking,
                "route_policy": "upstream native without route override",
                "arrival_policy": "one real question per document in the measured lead-in, then repeated questions on those documents",
                "pressure_window_s": [pressure_start, pressure_start + waves * period], "request_sources": specs}
    write_json(output / "workload.json", manifest)
    print(json.dumps({"requests": len(specs), "prefix_blocks": blocks, "pressure_window_s": manifest["pressure_window_s"],
                      "payload_bytes": payload.stat().st_size}), flush=True)


def load_questions(plan_path, plan, salt):
    payload = plan_path.with_name("document-prompts.json.gz")
    if file_sha256(payload) != plan["payload_sha256"]:
        raise ValueError("The prompt payload does not match the frozen manifest")
    if plan["ignore_eos"] or plan["temperature"] != 0 or plan["route_policy"] != "upstream native without route override":
        raise ValueError("The document workload must keep native sampling and routing")
    with gzip.open(payload, "rt", encoding="utf-8") as stream:
        books = json.load(stream)
    prefixes = {doc: sha256(token_bytes(book["prefix"])) for doc, book in books.items()}
    used_questions = defaultdict(set)
    rows = []
    for spec in plan["request_sources"]:
        doc, question = spec["document_id"], spec["source_question_index"]
        book = books[doc]
        suffix = book["suffixes"][question]
        digest = prefixes[doc].copy()
        digest.update(token_bytes(suffix))
        prompt = book["prefix"] + suffix
        if (question in used_questions[doc] or digest.hexdigest() != spec["prompt_sha256"] or
                len(prompt) != spec["prompt_tokens"] or len(prompt) + spec["max_tokens"] > MAX_CONTEXT):
            raise ValueError(f"Invalid document question {spec['id']}")
        used_questions[doc].add(question)
        rows.append({**spec, "prompt": prompt, "cache_salt": salt})
    validate_workload(rows)
    if len(rows) != plan["requests"] or sum(len(x["prefix"]) // BLOCK for x in books.values()) != plan["shared_prefix_blocks"]:
        raise ValueError("Incorrect workload size or shared working set")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prefix-blocks", type=int, default=3052)
    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--questions-per-wave", type=int, default=2)
    parser.add_argument("--pressure-start", type=float, default=125)
    parser.add_argument("--period", type=float, default=4)
    parser.add_argument("--direct-answer", action="store_true", help="Apply the model's official non-thinking chat template")
    args = parser.parse_args()
    build(args.source, args.tokenizer, args.output, args.prefix_blocks, args.waves, args.questions_per_wave,
          args.pressure_start, args.period, not args.direct_answer)


if __name__ == "__main__":
    main()
