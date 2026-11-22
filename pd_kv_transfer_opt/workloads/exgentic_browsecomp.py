"""Build a CPU replay ledger from complete BrowseComp-Plus sessions in Exgentic v2."""

from __future__ import annotations

import argparse
from collections import Counter
from compression import zstd
from datetime import datetime
from hashlib import sha256 as digest
import json
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen

from pd_kv_transfer_opt.common import percentile, read_json, sha256
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS as PAGE
from pd_kv_transfer_opt.simulator.agent_trace_gap import MAX_CONTEXT, read_bytes
from pd_kv_transfer_opt.workloads.document_questions import TOKENIZER_SHA256, common_length, token_bytes


REPO = "Exgentic/agent-llm-traces-v2"
REVISION = "ccd4c607cb708b4bb9e043bdc63a9013df35a9bc"
TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
FINISH = {"stop": "stop", "end_turn": "stop", "STOP": "stop",
          "tool_calls": "tool_calls", "tool_use": "tool_calls"}
HARNESSES = ("openai_solo", "smolagents_code", "tool_calling")


def fetch(directory):
    base = "https://huggingface.co"
    with urlopen(f"{base}/api/datasets/{REPO}/tree/{REVISION}/data/train", timeout=30) as stream:
        inventory = json.load(stream)
    if {row["path"] for row in inventory} != {f"data/train/{i:04d}.parquet" for i in range(9)}:
        raise ValueError("Unexpected files in the pinned dataset")
    directory.mkdir(parents=True, exist_ok=True)
    for row in inventory:
        path = directory / Path(row["path"]).name
        if path.exists() and path.stat().st_size == row["size"] and sha256(path) == row["lfs"]["oid"]:
            continue
        temporary = path.with_suffix(".part")
        url = f"{base}/datasets/{REPO}/resolve/{REVISION}/{row['path']}?download=true"
        with urlopen(url, timeout=60) as response, temporary.open("wb") as destination:
            copyfileobj(response, destination)
        if temporary.stat().st_size != row["size"] or sha256(temporary) != row["lfs"]["oid"]:
            raise ValueError(f"Dataset shard verification failed for {path.name}")
        temporary.replace(path)
    (directory / "source.json").write_text(
        json.dumps({"repo": REPO, "revision": REVISION, "files": inventory}, indent=2) + "\n", encoding="utf-8")


def optional_json(value, *, default=None):
    return json.loads(value) if value else default


def data_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def format_input(attributes):
    system = optional_json(attributes["gen_ai.system_instructions"], default=[])
    parts = [part["content"] for part in system if part["type"] == "text"]
    if len(parts) != len(system):
        raise ValueError("Non-text system instructions")
    messages = [{"role": "system", "content": "\n".join(parts)}] if parts else []
    for raw in optional_json(attributes["gen_ai.input.messages"], default=[]):
        texts, calls, output = [], [], []
        for part in raw["parts"]:
            match part["type"]:
                case "text":
                    texts.append(part["content"])
                case "tool_call":
                    args = part["arguments"]
                    if not isinstance(args, dict):
                        raise ValueError("Expected structured tool arguments")
                    calls.append({"id": part["id"], "type": "function",
                                  "function": {"name": part["name"], "arguments": args}})
                case "tool_call_response":
                    output.append(data_text(part["result"]))
                case _:
                    raise ValueError("Unsupported message part")
        role = raw["role"]
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported message role")
        message = {"role": role, "content": "\n".join(texts + output)}
        if calls:
            message["tool_calls"] = calls
        messages.append(message)
    tools = [{"type": "function", "function": {key: value for key, value in tool.items() if key != "type"}}
             for tool in optional_json(attributes["gen_ai.tool.definitions"], default=[])]
    return messages, tools


def observe(row):
    if row["status"] not in {"success", "unsuccessful"}:
        return None, "session_status"
    if row["harness"] not in HARNESSES:
        return None, "harness"
    if len(row["spans"]) < 3:
        return None, "short_session"
    starts_and_ends = sorted((datetime.fromisoformat(span["start_time"]), datetime.fromisoformat(span["end_time"]),
                              span["span_id"], span) for span in row["spans"])
    calls = []
    before = starts_and_ends[0][0]
    last = len(starts_and_ends) - 1
    for i, (start, end, _, span) in enumerate(starts_and_ends):
        a = span["attributes"]
        tags = a.get("gen_ai.response.finish_reasons") or []
        finish = FINISH.get(tags[0]) if len(tags) == 1 else None
        raw_prompt, output = a["gen_ai.usage.input_tokens"], a["gen_ai.usage.output_tokens"]
        cap = a["gen_ai.request.max_tokens"]
        if (span["status"]["code"] == 2 or start < before or end < start or not finish
                or type(raw_prompt) is not int or raw_prompt <= 0 or type(output) is not int or output <= 0
                or (cap and output >= cap) or not a["gen_ai.input.messages"]):
            return None, "span_status_dependency_or_EOS"
        if i == last and finish == "tool_calls":
            outputs = optional_json(a["gen_ai.output.messages"], default=[])
            submitted = any(part.get("name", "").rsplit("__", 1)[-1] == "submit"
                            for message in outputs for part in message.get("parts", []) if part["type"] == "tool_call")
            if not submitted:
                return None, "no_terminal_submit"
        calls.append((round((start - before).total_seconds() * 1000), raw_prompt, output, finish, a))
        before = end
    if not any(call[1] >= 65_536 for call in calls):
        return None, "no_long_source_prompt"
    if any(call[1] + call[2] > MAX_CONTEXT for call in calls):
        return None, "source_context_limit"
    return (starts_and_ends[0][0], before, calls), None


def encode_trace(row, observation, tokenizer, template, keys):
    start, end, calls = observation
    rendered, first_user = [], None
    for _, _, _, _, a in calls:
        messages, tools = format_input(a)
        if first_user is None:
            first_user = next(m["content"] for m in messages if m["role"] == "user")
        rendered.append(template.render(messages=messages, tools=tools,
                                        add_generation_prompt=True, enable_thinking=True))
    encoded = tokenizer.encode_batch_fast(rendered, add_special_tokens=False)
    tokens = [encoding.ids for encoding in encoded]
    if any(len(prompt) + c[2] > MAX_CONTEXT for prompt, c in zip(tokens, calls, strict=True)):
        return None, "rendered_context_limit"
    if not any(len(prompt) >= 65_536 for prompt in tokens):
        return None, "no_long_rendered_prompt"
    # A same tokenizer byte sequence and parent chain share a physical object.
    # Full prompts remain unchanged across two bandwidth runs.
    previous, turns, reads = [], [], 0
    for prompt, (gap, _, output, finish, _) in zip(tokens, calls, strict=True):
        shared = common_length(prompt, previous, min(len(prompt), len(previous))) // PAGE
        pages, parent = [], b""
        for end_of_page in range(PAGE, len(prompt) + 1, PAGE):
            child = digest(parent + token_bytes(prompt[end_of_page - PAGE:end_of_page])).digest()
            pages.append(keys.setdefault(child, len(keys) + 1))
            parent = child
        turns.append([gap, len(prompt), output, pages, shared, finish])
        reads += read_bytes(shared)
        previous = prompt
    label = json.dumps([row["session_id"], row["run_id"]], separators=(",", ":"))
    session = [digest(label.encode()).hexdigest()[:20], round(start.timestamp() * 1000),
               round((end - start).total_seconds() * 1000), reads, turns]
    details = {"id": session[0], "first_user_sha256": digest(first_user.encode()).hexdigest(),
               "harness": row["harness"], "source_model": row["models"][0],
               "success": row["status"] == "success",
               "reconstructed_to_source_prompt_ratio_median": percentile(
                   [len(t) / c[1] for t, c in zip(tokens, calls, strict=True)], .5),
               "source_prompt_tokens": sum(c[1] for c in calls)}
    return (session, details), None


def build(directory, tokenizer_directory, output, receipt):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from pyarrow.parquet import ParquetFile
    from tokenizers import Tokenizer

    metadata = read_json(directory / "source.json")
    if (metadata["repo"], metadata["revision"]) != (REPO, REVISION):
        raise ValueError("Dataset release differs from the pinned benchmark")
    tokenizer_path = tokenizer_directory / "tokenizer.json"
    template_path = tokenizer_directory / "chat_template.jinja"
    if sha256(tokenizer_path) != TOKENIZER_SHA256 or sha256(template_path) != TEMPLATE_SHA256:
        raise ValueError("Qwen tokenizer or chat template differs from the archived model")
    engine = Tokenizer.from_file(str(tokenizer_path))
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
        template_path.read_text(encoding="utf-8"))
    counts, keys, details_by_id, sources = Counter(), {}, {}, {}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part")
    with zstd.open(temporary, "wt", encoding="utf-8") as dest:
        print(json.dumps({"repo": REPO, "revision": REVISION, "benchmark": "BrowseComp-Plus",
                          "tokenizer_sha256": TOKENIZER_SHA256, "template_sha256": TEMPLATE_SHA256}), file=dest)
        for entry in metadata["files"]:
            path = directory / Path(entry["path"]).name
            if path.stat().st_size != entry["size"] or sha256(path) != entry["lfs"]["oid"]:
                raise ValueError(f"Unexpected release content for {path.name}")
            sources[path.name] = entry["lfs"]["oid"]
            parquet = ParquetFile(path)
            if "browsecompplus" not in parquet.read(columns=["benchmark"]).column("benchmark").to_pylist():
                continue
            for rows in parquet.iter_batches(batch_size=6, columns=[
                    "benchmark", "session_id", "run_id", "status", "models", "harness", "spans"]):
                for row in rows.to_pylist():
                    if row["benchmark"] != "browsecompplus":
                        continue
                    counts["source_sessions"] += 1
                    observation, reason = observe(row)
                    if reason:
                        counts[reason] += 1
                        continue
                    try:
                        encoded, reason = encode_trace(row, observation, engine, template, keys)
                    except (TypeError, ValueError, KeyError, StopIteration):
                        counts["unsupported_payload"] += 1
                        continue
                    if reason:
                        counts[reason] += 1
                        continue
                    session, details = encoded
                    if session[0] in details_by_id:
                        raise ValueError("Duplicate benchmark session identity")
                    details_by_id[session[0]] = details
                    counts["complete_rendered_long_sessions"] += 1
                    print(json.dumps(session, separators=(",", ":")), file=dest)
            print(path.name, counts["source_sessions"], counts["complete_rendered_long_sessions"], flush=True)
    temporary.replace(output)
    receipt.write_text(json.dumps({"counts": dict(counts), "sources": sources, "cohort_sha256": sha256(output),
                                   "sessions": details_by_id}, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "tokenizer", "output", "receipt"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="Download and verify the pinned public dataset")
    args = parser.parse_args()
    if args.fetch:
        fetch(args.data)
    build(args.data, args.tokenizer, args.output, args.receipt)


if __name__ == "__main__":
    main()
