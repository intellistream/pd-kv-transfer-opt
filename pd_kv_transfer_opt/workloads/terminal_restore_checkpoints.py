"""Select original adjacent windows that share attention prefixes at different checkpoints."""

from collections import defaultdict, deque
from itertools import product
from random import Random
from statistics import median

from pd_kv_transfer_opt.common import read_json, read_jsonl, sha256
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS, ATTENTION_BYTES, STATE_BYTES
from pd_kv_transfer_opt.workloads.terminal_restore_order import prefill_weights, routes
from pd_kv_transfer_opt.workloads.terminal_restore_pairs import historical_outputs


SHARED_PREFIX_FLOOR = 81920
MAX_SOURCE_OUTPUT = 1024
MAX_RECORDED_OUTPUT = 1600


def select(ledger_path, source_outputs_path, template, archives, *,
           shared_prefix_floor=SHARED_PREFIX_FLOOR, max_source_output=MAX_SOURCE_OUTPUT,
           max_recorded_output=MAX_RECORDED_OUTPUT, exclude_requests=()):
    ledger = {row["id"]: row for row in read_jsonl(ledger_path)}
    meta = read_json(source_outputs_path)
    source_counts = meta["source_completion_tokens"]
    if meta["source_revision"] != template["source_revision"] or ledger.keys() != source_counts.keys():
        raise ValueError("The source ledger does not match the template")
    outcomes, evidence = historical_outputs(archives, ledger, meta["source_revision"])
    excluded = set(exclude_requests)
    if not excluded.issubset(ledger):
        raise ValueError("Replay exclusions must be original Terminal-Bench requests")
    tasks = defaultdict(list)
    for row in ledger.values():
        tasks[row["task"]].append(row)

    def eligible(row):
        key = row["id"]
        source = source_counts[key]
        return (key not in excluded and not row["truncated"] and type(source) is int and 0 < source <= max_source_output
                and shared_prefix_floor <= row["prompt_tokens"] <= template["max_context_tokens"] - 8192
                and all(reason == "stop" and 0 < tokens <= max_recorded_output for reason, tokens in outcomes[key]))

    pairs = []
    bounds = {}
    for task, source_rows in sorted(tasks.items()):
        source_rows.sort(key=lambda row: row["step_id"])
        components = defaultdict(list)
        component = 0
        for index, (parent, child) in enumerate(zip(source_rows, source_rows[1:]), 1):
            lcp = child["source_prompt_lcp_to_previous"]
            if child["predecessor"] != parent["id"] or lcp < shared_prefix_floor:
                component += 1
                continue
            if eligible(parent) and eligible(child) and lcp >= .9 * parent["prompt_tokens"]:
                components[component].append((index, lcp // ATTENTION_BLOCK_TOKENS))
        choices = []
        for component_pairs in components.values():
            chosen, checkpoints, previous = [], set(), len(source_rows) + 1
            for index, blocks in reversed(component_pairs):
                if index < previous - 1 and blocks not in checkpoints:
                    chosen.append((index, blocks))
                    checkpoints.add(blocks)
                    previous = index
            if len(chosen) >= 3:
                choices.append((sum(blocks for _, blocks in chosen), sorted(chosen)))
        if not choices:
            continue
        _, chosen = max(choices)
        # Min adjacent LCP is a conservative shared-prefix lower bound for selected roots.
        previous_root = None
        distinct_upper = 0
        for index, blocks in chosen:
            parent, child = source_rows[index - 1:index + 1]
            root_at = index - 1
            overlap = min(row["source_prompt_lcp_to_previous"]
                          for row in source_rows[previous_root + 1:root_at + 1]) if previous_root is not None else 0
            distinct_upper += parent["prompt_tokens"] - overlap // ATTENTION_BLOCK_TOKENS * ATTENTION_BLOCK_TOKENS
            previous_root = root_at
            keys = (parent["id"], child["id"])
            pairs.append({"task": task, "root": keys[0], "successor": keys[1], "checkpoint_blocks": blocks,
                          "potential_GET_bytes": blocks * ATTENTION_BYTES + STATE_BYTES,
                          "estimated_output_tokens": {key: median(n for _, n in outcomes[key]) if outcomes[key]
                                                      else source_counts[key] for key in keys}})
        bounds[task] = distinct_upper
    if len(bounds) < 2:
        raise ValueError("The source does not provide multiple shared-prefix groups")
    requests = []
    for pair in pairs:
        for key, predecessor in ((pair["root"], None), (pair["successor"], pair["root"])):
            original = ledger[key]
            # Each disjoint pair is an independent original two-turn replay window.
            requests.append({"id": key, "task": pair["task"], "step_id": original["step_id"],
                             "session_id": pair["root"], "predecessor": predecessor, "offered_s": 0, "tool_gap_s": 0,
                             "prompt_tokens": original["prompt_tokens"], "max_tokens": max(8192, source_counts[key]),
                             "token_ids_uint32le_sha256": original["token_ids_uint32le_sha256"]})
    return {**{key: template[key] for key in ("source_repo", "source_revision", "tokenizer_sha256", "max_context_tokens")},
            "requests": requests, "selection": {"pairs": pairs, "source_ledger_sha256": sha256(ledger_path),
                "source_output_sha256": sha256(source_outputs_path), "historical_outcomes": evidence,
                "shared_prefix_floor_tokens": shared_prefix_floor, "max_source_output": max_source_output,
                "max_observed_output": max_recorded_output, "excluded_requests": sorted(excluded),
                "shared_root_tokens_upper_by_task": bounds}}


def root_order(grouped, weights, task_roles=None):
    tasks = sorted(grouped)
    if task_roles is None:
        sizes = [sum(weights[row["root"]] for row in grouped[task]) for task in tasks]
        total = sum(sizes)
        task_roles = min(((0, *bits) for bits in product((0, 1), repeat=len(tasks) - 1)),
                         key=lambda roles: abs(total - 2 * sum(size for size, role in zip(sizes, roles, strict=True) if role)))
    queues = [deque(row["root"] for task, role in zip(tasks, task_roles, strict=True)
                    if role == rank for row in grouped[task]) for rank in (0, 1)]
    load, order = [0.0, 0.0], []
    while any(queues):
        rank = int(load[1] < load[0])
        key = (queues[rank] or queues[1 - rank]).popleft()
        order.append(key)
        load[rank] += weights[key]
    return order


def successor_order(grouped, roots, weights):
    rng = Random(20261124)
    tasks = sorted(grouped)
    by_key = {row["successor"]: row for rows in grouped.values() for row in rows}
    templates = [tuple(row["successor"] for row in reversed(grouped[task])) for task in tasks]
    labels = [task for task, queue in enumerate(templates) for _ in queue]
    best = None
    for _ in range(4000):
        # Interleave tasks while offering each task's deeper checkpoints first.
        rng.shuffle(labels)
        parts = [iter(queue) for queue in templates]
        order = [next(parts[task]) for task in labels]
        placements = routes(order, weights)
        transfers = [0, 0]
        for key, rank in placements.items():
            pair = by_key[key]
            transfers[int(rank == roots[pair["root"]])] += pair["potential_GET_bytes"]
        score = min(transfers)
        if best is None or score > best[0]:
            best = score, order, placements, transfers
    return best


def plan(spec, body_receipt, release_s, *, admission_fit=None, ordered_successors=None):
    rows = spec["requests"]
    sizes = body_receipt["wire_body_bytes"]
    if sizes.keys() != {row["id"] for row in rows}:
        raise ValueError("The materialized rows differ from the source workload")
    weights = prefill_weights(sizes)
    grouped = defaultdict(list)
    for pair in spec["selection"]["pairs"]:
        grouped[pair["task"]].append(pair)
    for pairs in grouped.values():
        pairs.sort(key=lambda pair: (pair["checkpoint_blocks"], pair["root"]))
    cold_order = root_order(grouped, weights)
    roots = routes(cold_order, weights)
    potential, order, placements, transfers = successor_order(grouped, roots, weights)
    if ordered_successors is not None:
        replacement = list(ordered_successors)
        if sorted(replacement) != sorted(order):
            raise ValueError("An arrival order must cover all selected successors exactly once")
        order = replacement
        placements = routes(order, weights)
        transfers = [0, 0]
        for pair in spec["selection"]["pairs"]:
            same = placements[pair["successor"]] == roots[pair["root"]]
            transfers[int(same)] += pair["potential_GET_bytes"]
        potential = min(transfers)
    latency = {key: 0.0 for key in sizes}
    if admission_fit is not None:
        latency = {key: max(0.0, admission_fit["offset_s"] + admission_fit["s_per_byte"] * size)
                   for key, size in sizes.items()}

    def offered(keys, start, first_gap, gap):
        first, *rest = keys
        times = {first: start}
        times.update({key: start + first_gap + index * gap + latency[first] - latency[key]
                      for index, key in enumerate(rest)})
        if min(times.values()) != start:
            raise ValueError("Admission calibration moves a planned arrival before its burst")
        return times

    successor_gap = .049 if admission_fit is not None else .038
    offers = {**offered(cold_order, 0.0, .36, .05), **offered(order, release_s, .14, successor_gap)}
    route = {key: f"P{role}" for key, role in placements.items()}
    planned_rows = [{**{key: row[key] for key in ("id", "session_id", "predecessor", "tool_gap_s", "max_tokens", "prompt_tokens")},
                     "prompt_sha256": row["token_ids_uint32le_sha256"], "original_offered_s": row["offered_s"],
                     "offered_s": offers[row["id"]]} for row in rows]
    return {"workload_type": "terminal_source_restore_pairs", "source_repo": spec["source_repo"],
            "source_revision": spec["source_revision"], "source_workload_sha256": spec["selection"]["source_ledger_sha256"],
            "tokenizer_fingerprint": spec["tokenizer_sha256"], "parent_requests_sha256": body_receipt["sha256"],
            "requests": len(rows), "sessions": len(cold_order), "last_session_start_s": max(offers[key] for key in cold_order),
            "pressure_window_s": [release_s, release_s + 40], "ignore_eos": False, "temperature": 0,
            "tool_gap_policy": "Use the prior Terminal-Bench replay policy of zero additional tool delay",
            "max_tokens_policy": "Original Qwen replay budget rule max(8192, source completion tokens), with natural EOS",
            "route_policy": "upstream native without route override",
            "arrival_policy": "Independent two-turn windows with original adjacent prompts. Interleave their planned arrivals, with deeper checkpoints of the same source task offered first in the successor burst.",
            "selection": spec["selection"], "arrival_order_screen": {
                "full_restore_upper_bytes": sum(transfers), "new_order_estimated_GET_bytes": potential,
                "full_GET_by_relative_successor_first_P": {"same_as_roots": transfers[0], "opposite_to_roots": transfers[1]},
                "projected_root_P": {key: f"P{role}" for key, role in roots.items()},
                "projected_successor_P": route, "wire_body_bytes": sizes,
                "planned_root_order": cold_order, "planned_successor_order": order,
                "admission_fit": admission_fit,
                "assumptions": "Potential totals assume complete remote reads after P changes. Existing hybrid checkpoints, early root releases and actual P routing can reduce demand.",
            }, "request_sources": planned_rows}
