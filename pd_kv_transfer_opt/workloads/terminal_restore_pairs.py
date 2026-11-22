"""Freeze original Terminal-Bench long pairs with high restore bytes and short output evidence."""

from collections import defaultdict
from statistics import median

from pd_kv_transfer_opt.common import read_json, read_jsonl, sha256
from pd_kv_transfer_opt.profiling.profile_closed_sessions import read_trace
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS, ATTENTION_BYTES, STATE_BYTES
from pd_kv_transfer_opt.workloads.terminal_restore_order import choose_order, prefill_weights, routes


SOURCE_MAX_OUTPUT = 512
OBSERVED_MAX_OUTPUT = 1280
ROOT_GAP_S = 0.05
SUCCESSOR_GAP_S = 0.03


def historical_outputs(archives, ledger, revision):
    outcomes = defaultdict(list)
    evidence = {}
    for archive in archives:
        manifest = read_json(archive / "manifest.json")
        if manifest["workload"]["source_revision"] != revision or manifest["workload"]["ignore_eos"]:
            raise ValueError(f"Different source or EOS policy in {archive.name}")
        trace_path = archive / manifest["event_trace"]
        trace_hash = sha256(trace_path)
        if trace_hash != manifest["artifacts"][trace_path.name]["sha256"]:
            raise ValueError(f"Historical trace checksum mismatch in {archive.name}")
        aliases = {}
        for row in manifest["workload"]["request_sources"]:
            identifier = row["id"]
            key = identifier if identifier in ledger else identifier.split("-", 1)[1]
            if ledger[key]["token_ids_uint32le_sha256"] != row["prompt_sha256"]:
                raise ValueError(f"Historical request does not match original input {key}")
            aliases[row["id"]] = key
        trace = read_trace(trace_path)
        evidence[archive.name] = {"trace_sha256": trace_hash, "cases": list(trace)}
        for arm in trace.values():
            for row in arm["client"]:
                if row["type"] == "request_end" and row["status"] == "ok":
                    outcomes[aliases[row["id"]]].append((row["finish_reason"], row["completion_tokens"]))
    return outcomes, evidence


def select(ledger_file, source_outputs_file, template, archives, *, max_source_output=SOURCE_MAX_OUTPUT):
    rows = list(read_jsonl(ledger_file))
    ledger = {row["id"]: row for row in rows}
    meta = read_json(source_outputs_file)
    source_counts = meta["source_completion_tokens"]
    if meta["source_revision"] != template["source_revision"] or ledger.keys() != source_counts.keys():
        raise ValueError("Dataset does not match the audited source length ledger")
    outcomes, evidence = historical_outputs(archives, ledger, meta["source_revision"])

    def eligible(row):
        source = source_counts[row["id"]]
        return (not row["truncated"] and type(source) is int and 0 < source <= max_source_output
                and 24_000 <= row["prompt_tokens"] <= template["max_context_tokens"] - 8192
                and all(reason == "stop" and 0 < num <= OBSERVED_MAX_OUTPUT for reason, num in outcomes[row["id"]]))

    best = {}
    for row in rows:
        root = ledger.get(row["predecessor"])
        if root is None or not (eligible(root) and eligible(row)):
            continue
        lcp = row["source_prompt_lcp_to_previous"]
        if lcp < 0.9 * root["prompt_tokens"]:
            continue
        volume = lcp // ATTENTION_BLOCK_TOKENS * ATTENTION_BYTES + STATE_BYTES
        task = row["task"]
        if task not in best or volume > best[task]["potential_GET_bytes"]:
            best[task] = {"root": root["id"], "successor": row["id"], "potential_GET_bytes": volume}

    if not best:
        raise ValueError("The original source has no qualifying two-turn windows")
    requests = []
    for pair in best.values():
        root_id, child_id = pair["root"], pair["successor"]
        pair["historical_completions"] = {key: len(outcomes[key]) for key in (root_id, child_id)}
        pair["estimated_output_tokens"] = {
            key: median(count for _, count in outcomes[key]) if outcomes[key] else source_counts[key]
            for key in (root_id, child_id)}
        for key, predecessor in ((root_id, None), (child_id, root_id)):
            row = ledger[key]
            requests.append({"id": key, "task": row["task"], "step_id": row["step_id"], "session_id": row["task"],
                             "predecessor": predecessor, "offered_s": 0, "tool_gap_s": 0,
                             "prompt_tokens": row["prompt_tokens"], "max_tokens": 8192,
                             "token_ids_uint32le_sha256": row["token_ids_uint32le_sha256"]})
    return {**{key: template[key] for key in ("source_repo", "source_revision", "tokenizer_sha256", "max_context_tokens")},
            "selection": {"source_ledger_sha256": sha256(ledger_file), "source_output_sha256": sha256(source_outputs_file),
                          "max_source_output": max_source_output, "max_observed_output": OBSERVED_MAX_OUTPUT,
                          "output_history": evidence, "pairs": list(best.values())}, "requests": requests}


def plan(spec, body_receipt, release_s, *, root_first_gap_s=ROOT_GAP_S,
         successor_first_gap_s=SUCCESSOR_GAP_S, successor_gap_s=SUCCESSOR_GAP_S):
    rows = spec["requests"]
    lengths = body_receipt["wire_body_bytes"]
    if lengths.keys() != {row["id"] for row in rows}:
        raise ValueError("Native source data do not cover the selected requests")
    pairs = spec["selection"]["pairs"]
    weights = prefill_weights(lengths)
    roots = sorted((p["root"] for p in pairs), key=lambda key: (-weights[key], key))
    successors = sorted(p["successor"] for p in pairs)
    root_roles = routes(roots, weights)
    source_roles = {p["successor"]: root_roles[p["root"]] for p in pairs}
    potential = {p["successor"]: p["potential_GET_bytes"] for p in pairs}
    volume, order, selected = choose_order(successors, weights, source_roles, potential)
    offers = {roots[0]: 0.0, order[0]: release_s}
    offers.update({key: root_first_gap_s + index * ROOT_GAP_S for index, key in enumerate(roots[1:])})
    offers.update({key: release_s + successor_first_gap_s + index * successor_gap_s
                   for index, key in enumerate(order[1:])})
    return {"workload_type": "terminal_source_restore_pairs", "source_repo": spec["source_repo"],
            "source_revision": spec["source_revision"], "source_workload_sha256": spec["selection"]["source_ledger_sha256"],
            "tokenizer_fingerprint": spec["tokenizer_sha256"], "parent_requests_sha256": body_receipt["sha256"],
            "requests": len(rows), "sessions": len(roots), "last_session_start_s": max(offers[key] for key in roots),
            "pressure_window_s": [release_s, release_s + 10], "ignore_eos": False, "temperature": 0,
            "tool_gap_policy": "Use the prior Terminal-Bench replay policy of zero additional tool delay",
            "max_tokens_policy": "Original Qwen replay budget rule max(8192, source completion tokens), with natural EOS",
            "route_policy": "upstream native without route override",
            "arrival_policy": "Offer original roots in a short burst and original successors at the common late burst after their own real predecessors. Root and successor plans are shared across all bandwidth arms.",
            "selection": spec["selection"], "arrival_order_screen": {
                "full_restore_upper_bytes": sum(potential.values()), "new_order_estimated_GET_bytes": volume,
                "root_first_gap_s": root_first_gap_s, "successor_first_gap_s": successor_first_gap_s,
                "successor_gap_s": successor_gap_s,
                "projected_root_P": {key: f"P{value}" for key, value in root_roles.items()},
                "projected_successor_P": {key: f"P{value}" for key, value in selected.items()},
                "wire_body_bytes": lengths,
            },
            "request_sources": [{**{key: row[key] for key in ("id", "session_id", "predecessor", "tool_gap_s", "max_tokens", "prompt_tokens")},
                                 "original_offered_s": row["offered_s"], "offered_s": offers[row["id"]],
                                 "prompt_sha256": row["token_ids_uint32le_sha256"]} for row in rows]}
