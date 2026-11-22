"""Schedule original long-history snapshots around already-timed, shared-prefix producers."""

from collections import defaultdict
from itertools import permutations
from math import ceil
from statistics import median

from pd_kv_transfer_opt.common import read_jsonl, sha256
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS, ATTENTION_BYTES, STATE_BYTES
from pd_kv_transfer_opt.workloads.terminal_restore_expanded import arrange_successors
from pd_kv_transfer_opt.workloads.terminal_restore_order import prefill_weights, routes


FAMILIES = ("caffe-cifar-10", "financial-document-processor", "make-doom-for-mips", "regex-chess")


def select(ledger, template, outputs, observed, admission_fit, *, natural_output_limit=900, max_suffix=24_000):
    source = defaultdict(list)
    for row in read_jsonl(ledger):
        if row["task"] in FAMILIES:
            source[row["task"]].append(row)

    def eligible(row):
        key = row["id"]
        history = observed.get(key, ())
        budget = outputs[key]
        return (not row["truncated"] and 65_536 <= row["prompt_tokens"] <= template["max_context_tokens"] - 8192
                and type(budget) is int and 0 < budget <= 8192 and len(history) >= 4
                and all(reason == "stop" and 0 < count <= natural_output_limit for reason, count in history))

    def reference_output(key):
        return median(count for _, count in observed[key])

    groups, requests = {}, []
    for task, rows in sorted(source.items()):
        rows.sort(key=lambda row: row["step_id"])
        best = None
        for index, anchor in enumerate(rows[:-1]):
            if not eligible(anchor) or max(count for _, count in observed[anchor["id"]]) > 500:
                continue
            blocks = (anchor["prompt_tokens"] - 1) // ATTENTION_BLOCK_TOKENS
            prefix = blocks * ATTENTION_BLOCK_TOKENS
            common_bytes = blocks * ATTENTION_BYTES + STATE_BYTES
            later = []
            for other in rows[index + 1:]:
                if other["source_prompt_lcp_to_previous"] < prefix:
                    break
                if other["prompt_tokens"] - prefix <= max_suffix and eligible(other):
                    later.append(other)
            if len(later) < 8:
                continue
            # Count every overlapping demand in the ideal case. Real scheduler batches may share an earlier load.
            q = sum(reference_output(row["id"]) for row in later)
            volume = len(later) * common_bytes
            score = volume / (1.0 + q / 40_000)
            if best is None or score > best[0]:
                best = score, anchor, later, prefix, common_bytes
        if best is None:
            raise ValueError(f"No original producer with enough compatible later snapshots in {task}")
        _, anchor, later, prefix, common_bytes = best
        anchor_key = anchor["id"]
        groups[task] = {"timed_producer": anchor_key, "later_original_requests": [row["id"] for row in later],
                        "shared_complete_prefix_tokens": prefix, "bytes_if_one_full_read": common_bytes,
                        "producer_estimated_output_tokens": reference_output(anchor_key)}
        for row in (anchor, *later):
            key = row["id"]
            requests.append({**{name: row[name] for name in ("id", "task", "step_id", "prompt_tokens", "token_ids_uint32le_sha256")},
                             "source_predecessor_in_full_trace": row["predecessor"], "session_id": key, "predecessor": None,
                             "offered_s": 0.0, "tool_gap_s": 0, "max_tokens": max(8192, outputs[key])})
    return {**{key: template[key] for key in ("source_repo", "source_revision", "tokenizer_sha256", "max_context_tokens")},
            "selection": {"source_ledger_sha256": sha256(ledger), "shared_producer_groups": groups,
                          "admission_fit": admission_fit, "maximum_suffix_tokens": max_suffix,
                          "max_natural_observed_output": natural_output_limit}, "requests": requests}


def make_plan(spec, receipt):
    by_id = {row["id"]: row for row in spec["requests"]}
    sizes = receipt["wire_body_bytes"]
    if sizes.keys() != by_id.keys():
        raise ValueError("The original prompt bodies do not match the shared-prefix scope")
    weights = prefill_weights(sizes)
    groups = spec["selection"]["shared_producer_groups"]
    grouped = {task: [{"task": task, "root": group["timed_producer"], "successor": key,
                       "potential_GET_bytes": group["bytes_if_one_full_read"]}
                      for key in group["later_original_requests"]] for task, group in groups.items()}
    anchors = [group["timed_producer"] for group in groups.values()]
    best = None
    for order in permutations(anchors):
        roots = routes(order, weights)
        burst, recovered = arrange_successors(grouped, roots, weights)
        cold = {f"P{role}": sum(by_id[key]["prompt_tokens"] for key, assigned in roots.items() if assigned == role)
                for role in (0, 1)}
        score = recovered / max(cold.values())
        if best is None or score > best[0]:
            best = score, order, roots, burst, recovered, cold
    _, anchors, roots, burst, recovered, cold = best
    start = ceil((max(cold.values()) / 10_600 + 4) * 2) / 2
    slope = spec["selection"]["admission_fit"]["s_per_byte"]
    offers = {anchors[0]: 0.0, burst[0]: start}
    offers.update({key: .72 + .12 * index + slope * (sizes[anchors[0]] - sizes[key])
                   for index, key in enumerate(anchors[1:])})
    offers.update({key: start + .11 + .027 * index + slope * (sizes[burst[0]] - sizes[key])
                   for index, key in enumerate(burst[1:])})
    all_upper = sum(row["potential_GET_bytes"] for rows in grouped.values() for row in rows)
    rows = [{**{key: row[key] for key in ("id", "session_id", "predecessor", "tool_gap_s", "max_tokens", "prompt_tokens")},
             "offered_s": offers[row["id"]], "original_offered_s": row["offered_s"],
             "prompt_sha256": row["token_ids_uint32le_sha256"]} for row in spec["requests"]]
    if len(rows) != len(offers) or len(set(offers.values())) != len(rows):
        raise ValueError("Each original history must have a distinct frozen arrival")
    return {"workload_type": "terminal_source_restore_pairs", "source_repo": spec["source_repo"],
            "source_revision": spec["source_revision"], "source_workload_sha256": spec["selection"]["source_ledger_sha256"],
            "parent_requests_sha256": receipt["sha256"], "tokenizer_fingerprint": spec["tokenizer_sha256"],
            "requests": len(rows), "sessions": len(rows), "pressure_window_s": [start, start + 40.0],
            "last_session_start_s": max(offers.values()), "request_sources": rows, "selection": spec["selection"],
            "ignore_eos": False, "temperature": 0, "route_policy": "upstream native without route override",
            "tool_gap_policy": "Each timed request is an original single-request history snapshot from a replay window",
            "max_tokens_policy": "Original Qwen replay budget rule max(8192, source completion tokens), with natural EOS",
            "arrival_policy": "Time all original snapshots including the producers. Separate input history windows share an exact complete prefix without sharing a runtime session dependency.",
            "arrival_order_screen": {"wire_body_bytes": sizes, "planned_producers": list(anchors),
                                     "producer_P_if_empty": {key: f"P{role}" for key, role in roots.items()},
                                     "producer_cold_prefix_tokens_by_P": cold,
                                     "planned_later_snapshots": burst, "admission_fit": spec["selection"]["admission_fit"],
                                     "full_restore_upper_bytes": all_upper, "new_order_estimated_GET_bytes": recovered}}
