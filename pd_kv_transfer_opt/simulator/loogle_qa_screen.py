"""Screen genuine same-document questions under native APC using CPU only."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256 as digest
import json
from math import ceil
from pathlib import Path
from urllib.request import urlopen

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.simulator.agent_cache import STATE_BYTES_PER_CHECKPOINT, STORE_ATTENTION_BYTES
from pd_kv_transfer_opt.simulator.agent_replay import GB, PAGE, Replay
from pd_kv_transfer_opt.simulator.agent_trace_gap import read_bytes
from pd_kv_transfer_opt.simulator.freeinference_ab import PUBLISHED_BASELINE, calibration, report
from pd_kv_transfer_opt.simulator.terminal_pressure import peak_window
from pd_kv_transfer_opt.workloads.document_questions import MAX_CONTEXT, MAX_OUTPUT, TOKENIZER_SHA256, common_length, token_bytes
from pd_kv_transfer_opt.workloads.exgentic_browsecomp import TEMPLATE_SHA256
from pd_kv_transfer_opt.workloads.freeinference import Session, Turn


REPO = "bigai-nlco/LooGLE"
REVISION = "4b50b4cb9333b48b3f7ddfd307c1d13369be5d1e"
ASSUMED_OUTPUT = 125


@dataclass(slots=True)
class Document:
    fingerprint: str
    context: str
    questions: dict
    pages: int = 0


def read_documents():
    documents, sources = {}, {}
    for task in ("longdep_qa", "shortdep_qa"):
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{task}.jsonl"
        hashed, count = digest(), 0
        with urlopen(url, timeout=35) as stream:
            for line in stream:
                count += 1
                hashed.update(line)
                row = json.loads(line)
                key = digest(row["context"].encode()).hexdigest()
                if key not in documents:
                    documents[key] = Document(key, row["context"], {})
                documents[key].questions.setdefault(row["question"], row["id"])
        sources[task] = {"rows": count, "sha256": hashed.hexdigest(), "url": url}
    return documents.values(), sources


def encodings(document, tokenizer, template, count):
    chosen = list(document.questions)[:count]
    messages = [template.render(messages=[{"role": "user", "content":
                                           f"Please answer the question based on the long texts below.\n\n"
                                           f"{document.context}\n\nQuestion: {question}"}],
                                add_generation_prompt=True, enable_thinking=True) for question in chosen]
    return [row.ids for row in tokenizer.encode_batch_fast(messages, add_special_tokens=False)]


def geometry(documents, local_pages, store_pages):
    # This cap permits a complete ring to replace the longest lookup even if
    # it skipped one other full document in the same cycle.
    largest = (store_pages - local_pages) // 2
    choices, counts = [], {}
    for waves in range(2, 13):
        required = 1 + 2 * waves
        pool = [doc for doc in documents if doc.pages <= largest and len(doc.questions) >= required]
        counts[str(required)] = {"docs": len(pool), "pages": sum(doc.pages for doc in pool)}
        if counts[str(required)]["pages"] < local_pages + largest:
            continue
        # Minimize the number of documents per wave along the feasible fill
        # range so that the 200 Gbps D stage can consume more natural output.
        states = {0: ()}
        for doc in sorted(pool, key=lambda item: (-item.pages, item.fingerprint)):
            for pages, selected in tuple(states.items()):
                total = pages + doc.pages
                if total <= store_pages and (total not in states or len(selected) + 1 < len(states[total])):
                    states[total] = (*selected, doc)
        eligible = [(pages, picked) for pages, picked in states.items()
                    if picked and pages - sum(d.pages for d in picked[:2]) >= local_pages]
        if eligible:
            pages, picked = min(eligible, key=lambda item: (len(item[1]), -item[0]))
            choices.append((waves, picked, pages))
    return max(choices, default=None, key=lambda item: item[0]), counts, largest


def freeze(books, waves, tokenizer, template, service_models):
    payload = {doc.fingerprint: encodings(doc, tokenizer, template, 1 + 2 * waves) for doc in books}
    source = {doc.fingerprint: list(doc.questions.values())[:1 + 2 * waves] for doc in books}
    for doc in books:
        prompts = payload[doc.fingerprint]
        ceiling = min(map(len, prompts))
        pages = min(common_length(prompts[0], other, ceiling) for other in prompts[1:]) // PAGE
        if pages != doc.pages or any(len(ids) + MAX_OUTPUT > MAX_CONTEXT for ids in prompts):
            raise ValueError("Selected questions changed the capacity geometry or context envelope")
    keys, templates, tasks = {}, {}, {}
    for key, prompts in payload.items():
        for question, ids in enumerate(prompts):
            parent, blocks = b"", []
            for i in range(PAGE, len(ids) + 1, PAGE):
                parent = digest(parent + token_bytes(ids[i - PAGE:i])).digest()
                blocks.append(keys.setdefault(parent, len(keys) + 1))
            name = f"{key[:20]}-{question:02d}"
            req = {"model": "Qwen3.5-35B-A3B", "prompt": ids, "max_tokens": MAX_OUTPUT, "temperature": 0, "stream": True,
                   "stream_options": {"include_usage": True}, "return_token_ids": True, "cache_salt": "loogle_qa_screen"}
            body = len(json.dumps(req).encode("utf-8"))
            turn = Turn(0, len(ids), ASSUMED_OUTPUT, tuple(blocks), 0, "diagnostic_reference")
            session = Session(name, len(templates), 0, 0, (turn,))
            tasks[name] = {"id": name, "document_sha256": key, "source_question_id": source[key][question],
                           "prompt_sha256": digest(token_bytes(ids)).hexdigest(), "prompt_tokens": len(ids), "body_bytes": body}
            templates[key, question] = session
    full_wave = 2 * sum(read_bytes(doc.pages) for doc in books) / GB
    slow_R = min(100 * service.rail_efficiency / (64 * max(service.shares)) for service in service_models)
    period = full_wave / (2 * slow_R * .85)
    ordered = sorted(books, key=lambda doc: (tasks[templates[doc.fingerprint, 0].name]["body_bytes"], doc.fingerprint))
    roots = [templates[doc.fingerprint, 0] for doc in ordered]
    root_offers = {session.name: i / 8 for i, session in enumerate(roots)}
    body_bytes = {f"{key}:0": task["body_bytes"] for key, task in tasks.items()}
    start = ceil(max(Replay(roots, root_offers, service, 100, observed_body_bytes=body_bytes).run().now
                     for service in service_models) + 2)
    sessions, offers = roots[:], root_offers.copy()
    for cycle in range(waves):
        for i, doc in enumerate(ordered):
            for question in (1 + 2 * cycle, 2 + 2 * cycle):
                session = templates[doc.fingerprint, question]
                sessions.append(session)
                offers[session.name] = start + cycle * period + .048 * i
    rows = [{**tasks[s.name], "offered_s": offers[s.name]} for s in sessions]
    decode_tokens = min(2 * s.slots / (s.decode_first_s + (s.slots - 1) * s.decode_add_s) for s in service_models)
    design = {"regular_start_s": start, "period_s": period, "waves": waves,
              "requests_per_wave": 2 * len(books),
              "max_mean_output_tokens_at_85pct_decode_capacity": .85 * decode_tokens * period / (2 * len(books)),
              "max_mean_output_tokens_at_decode_peak": decode_tokens * period / (2 * len(books)),
              "potential_full_GET_GB_per_wave": full_wave}
    return sessions, offers, body_bytes, rows, design


def arm_receipt(run, design):
    result = report(run)
    start, period = design["regular_start_s"], design["period_s"]
    wave = defaultdict(list)
    for job in run.requests.values():
        if job.offered >= start:
            wave[int((job.offered - start + 1e-8) // period)].append(job.row)
    work = [row for batch in wave.values() for row in batch]
    finish = max(row["D_KV_ready_s"] for row in work)
    epochs = [(max(a, start), min(b, finish), rates) for a, b, rates in run.tx_epochs if b > start and a < finish]
    result["pressure"] = {
        "actual_GET_GB": sum(row["GET_bytes"] for row in work) / GB,
        "actual_GET_GB_by_wave": [sum(row["GET_bytes"] for row in wave[i]) / GB for i in sorted(wave)],
        "200_or_100_wire_ceiling_Gbps": 8 * run.GB_s,
        "aggregate_tx_peak_30s_Gbps": peak_window(epochs, 30),
        "waves_with_D_KV_ready_before_next": sum(
            max(row["D_KV_ready_s"] for row in wave[i]) < start + (i + 1) * period
            for i in sorted(wave)[:-1]),
    }
    return result


def cadence_bound(design, read_GB, service_models):
    rates = [100 * s.rail_efficiency / (64 * max(s.shares)) for s in service_models]
    decode_peak = min(2 * s.slots / (s.decode_first_s + (s.slots - 1) * s.decode_add_s) for s in service_models)
    q_wave = ASSUMED_OUTPUT * design["requests_per_wave"]
    read_per_wave = read_GB / design["waves"]
    period = read_per_wave / max(rates)
    return {"100_actual_GET_GB_per_wave": read_per_wave,
            "longest_period_s_to_saturate_100_at_observed_mix_and_faster_100_rate": period,
            "shortest_period_s_to_keep_up_with_125_token_proxy_at_200_D_peak": q_wave / decode_peak,
            "shortest_period_s_at_85pct_D_peak_with_same_proxy": q_wave / (.85 * decode_peak),
            "longest_period_s_to_saturate_100_assuming_full_read_at_faster_100_rate":
                design["potential_full_GET_GB_per_wave"] / max(rates),
            "max_mean_output_tokens_with_observed_mix_and_no_D_slack_at_200": period * decode_peak / design["requests_per_wave"]}


def eos_provenance(results, requests):
    expected = {row["prompt_sha256"] for row in requests}
    found, checked = set(), 0
    for name in ("manifest.json", "workload.json"):
        for path in results.glob(f"*/{name}"):
            value = read_json(path)
            if not isinstance(value, dict):
                continue
            workload = value.get("workload", value)
            if isinstance(workload, dict) and isinstance(workload.get("request_sources"), list):
                checked += 1
                found.update(row.get("prompt_sha256") for row in workload["request_sources"] if isinstance(row, dict))
    return {"archived_workload_manifests_with_request_sources": checked,
            "exact_prompt_fingerprint_matches": len(expected & found), "dataset_fields_with_natural_EOS": 0,
            "scope": "Local results archive manifests and the public pinned LooGLE source schema."}


def main():
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    token_path, template_path = args.tokenizer / "tokenizer.json", args.tokenizer / "chat_template.jinja"
    if sha256(token_path) != TOKENIZER_SHA256 or sha256(template_path) != TEMPLATE_SHA256:
        raise ValueError("Use the exact deployment tokenizer and template")
    tokenizer = Tokenizer.from_file(str(token_path))
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(template_path.read_text(encoding="utf-8"))
    documents, sources = read_documents()
    compatible = []
    for doc in documents:
        if len(doc.questions) < 2:
            continue
        first, second = encodings(doc, tokenizer, template, 2)
        if max(len(first), len(second)) + MAX_OUTPUT <= MAX_CONTEXT:
            doc.pages = common_length(first, second, min(len(first), len(second))) // PAGE
            if doc.pages:
                compatible.append(doc)
    models = [replace(service, release_suffix_first=True) for service, _ in calibration(args.results).values()]
    service = models[0]
    touched = service.store_bytes // (STORE_ATTENTION_BYTES + STATE_BYTES_PER_CHECKPOINT)
    cohort, counts, max_robust = geometry(compatible, service.p_pages, touched)
    result = {"baseline_revision": PUBLISHED_BASELINE, "repo": REPO, "revision": REVISION, "source_files": sources,
              "tokenizer_sha256": TOKENIZER_SHA256, "template_sha256": TEMPLATE_SHA256,
              "distinct_document_content": len(documents), "multi_question_compatible_docs": len(compatible),
              "shared_prefix_tokens_p50_p90_p99": [percentile([d.pages * PAGE for d in compatible], p) for p in (.5, .9, .99)],
              "max_pages_per_doc_for_single_skip_full_ring": max_robust, "geometry_screen": counts,
              "P_capacity_pages": service.p_pages, "D_Store_capacity_pages_dense": touched,
              "exact_native_EOS_receipts": 0, "native_EOS_precondition_passed": False}
    if cohort is not None:
        waves, books, pages = cohort
        sessions, offers, body, rows, design = freeze(books, waves, tokenizer, template, models)
        arms = {cap: arm_receipt(Replay(sessions, offers, service, cap, observed_body_bytes=body).run(), design)
                for cap in (100, 200)}
        result.update({"documents": [{"sha256": doc.fingerprint, "shared_pages": doc.pages,
                                      "distinct_questions": len(doc.questions)} for doc in books],
                       "total_shared_pages": pages, "plan": design, "requests": rows,
                       "assumed_output_tokens_per_request_for_prefilter": ASSUMED_OUTPUT,
                       "local_EOS_provenance_screen": eos_provenance(args.results, rows),
                       "proxy_arms": arms, "cadence_bound": cadence_bound(design, arms[100]["pressure"]["actual_GET_GB"], models)})
    write_json(args.output, result)


if __name__ == "__main__":
    main()
