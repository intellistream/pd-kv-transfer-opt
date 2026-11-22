"""Place timed original roots across two arrivals to expose restore before successors."""

import argparse
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from random import Random

from pd_kv_transfer_opt.common import read_jsonl, write_json
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS, ATTENTION_BYTES, STATE_BYTES
from pd_kv_transfer_opt.workloads.terminal_restore_checkpoints import root_order
from pd_kv_transfer_opt.workloads.terminal_restore_dense import cold_prefix_bound, make_plan as dense_plan
from pd_kv_transfer_opt.workloads.terminal_restore_order import prefill_weights, routes
from pd_kv_transfer_opt.workloads.terminal_restore_stable import guard_cold_arrivals


def compatible_roots(windows, source):
    indexes = {row["id"]: index for index, row in enumerate(source)}
    window_roots = sorted((row["root"] for row in windows), key=indexes.__getitem__)
    edges = {}
    for i, seed in enumerate(window_roots):
        first = indexes[seed]
        checkpoint = (source[first]["prompt_tokens"] - 1) // ATTENTION_BLOCK_TOKENS
        if checkpoint < 64:
            continue
        for j in range(i + 1, len(window_roots)):
            later = indexes[window_roots[j]]
            common = min(row["source_prompt_lcp_to_previous"] for row in source[first + 1:later + 1])
            if checkpoint * ATTENTION_BLOCK_TOKENS <= common:
                edges[i, j] = checkpoint

    @lru_cache(None)
    def best(mask):
        if not mask:
            return 0, ()
        i = (mask & -mask).bit_length() - 1
        remaining = mask & ~(1 << i)
        found = best(remaining)
        for j in range(i + 1, len(window_roots)):
            if (i, j) not in edges or not (remaining & (1 << j)):
                continue
            volume, matches = best(remaining & ~(1 << j))
            found = max(found, (volume + edges[i, j], ((i, j), *matches)))
        return found

    _, matches = best((1 << len(window_roots)) - 1)
    distinct = set()
    result = []
    for i, j in matches:
        checkpoint = edges[i, j]
        if checkpoint not in distinct:
            result.append((window_roots[i], window_roots[j], checkpoint))
        distinct.add(checkpoint)
    return result


def organize(groups, producer_roles, weights):
    rng = Random(20261128)
    queues = [sorted(rows, key=lambda pair: (-pair["checkpoint_blocks"], pair["successor"])) for _, rows in sorted(groups.items())]
    template = tuple(index for index, group in enumerate(queues) for _ in group)
    pairs = {row["successor"]: row for group in queues for row in group}
    choice = None
    for _ in range(4000):
        labels = list(template)
        rng.shuffle(labels)
        cursors = [iter(group) for group in queues]
        order = [next(cursors[index])["successor"] for index in labels]
        places = routes(order, weights)
        volumes = [sum(row["potential_GET_bytes"] for key, row in pairs.items()
                       if (places[key] ^ initial) != roles[row["root"]])
                   for roles in producer_roles for initial in (0, 1)]
        score = min(volumes), -max(volumes)
        if choice is None or score > choice[0]:
            choice = score, order, places, volumes
    return choice[1:]


def choose_first_roots(guessed, windows, weights, ledger):
    choices = [None] + [row["root"] for rows in guessed.values() if len(rows) > 1 for row in rows]
    roots = {row["root"]: row for rows in windows.values() for row in rows}
    best = None
    for delayed in choices:
        links = {task: [row for row in rows if row["root"] != delayed] for task, rows in guessed.items()}
        grouped = {task: sorted((roots[row["root"]] for row in rows), key=lambda row: row["checkpoint_blocks"])
                   for task, rows in links.items()}
        order = root_order(grouped, weights)
        roles = routes(order, weights)
        by_role = {f"P{rank}": cold_prefix_bound([roots[key] for key in order if roles[key] == rank], ledger)
                   for rank in (0, 1)}
        max_cold = max(sum(b.values()) for b in by_role.values())
        volume = sum(row["potential_GET_bytes"] for rows in links.values() for row in rows)
        score = volume / max_cold
        if best is None or score > best[0]:
            best = score, links, order, roles, by_role, delayed
    return best[1:]


def make_plan(parent, references, ledger, late_root_s, successor_s):
    if not 0 < late_root_s < successor_s:
        raise ValueError("The later original roots must be offered before the successor wave")
    workload = dense_plan(parent, references, ledger, late_root_s)
    pairs = workload["selection"]["pairs"]
    windows = defaultdict(list)
    source = defaultdict(list)
    for pair in pairs:
        windows[pair["task"]].append(pair)
    for row in read_jsonl(ledger):
        if row["task"] in windows:
            source[row["task"]].append(row)
    guessed = defaultdict(list)
    for task, task_pairs in windows.items():
        raw = sorted(source[task], key=lambda row: row["step_id"])
        for seed, later, checkpoint in compatible_roots(task_pairs, raw):
            guessed[task].append({"root": seed, "successor": later, "checkpoint_blocks": checkpoint,
                                  "potential_GET_bytes": checkpoint * ATTENTION_BYTES + STATE_BYTES})
    if len(guessed) < 2:
        raise ValueError("At least two original source families need compatible root checkpoints")
    screen = workload["arrival_order_screen"]
    sizes = screen["wire_body_bytes"]
    weights = prefill_weights(sizes)
    guessed, first, first_roles, projected_cold, moved_to_later = choose_first_roots(guessed, windows, weights, ledger)
    seed_ids = set(first)
    later, _, read_by_orientation = organize(guessed, [first_roles], weights)
    without_match = {row["root"] for row in pairs} - seed_ids - set(later)
    later += sorted(without_match)
    later_roles = routes(later, weights)
    parent_roles = [{**first_roles, **{key: role ^ initial for key, role in later_roles.items()}} for initial in (0, 1)]
    successors, _, successor_read = organize(windows, parent_roles, weights)
    screen["planned_root_order"] = first
    guard_cold_arrivals(workload)
    first_offers = {row["id"]: row["offered_s"] for row in workload["request_sources"] if row["id"] in seed_ids}
    lag = screen["admission_fit"]["s_per_byte"]

    def arrivals(order, start):
        result = {order[0]: start}
        for i, key in enumerate(order[1:]):
            result[key] = start + .14 + i * .05 + lag * (sizes[order[0]] - sizes[key])
        return result

    offers = {**first_offers, **arrivals(later, late_root_s), **arrivals(successors, successor_s)}
    if len(offers) != workload["requests"]:
        raise ValueError("Every original request must occur once in the timed replay")
    workload["request_sources"] = [{**row, "offered_s": offers[row["id"]]} for row in workload["request_sources"]]
    workload["last_session_start_s"] = max(offers[key] for key in (*first, *later))
    workload["pressure_window_s"] = [late_root_s, successor_s + 30]
    workload["arrival_policy"] = "Every original root is a timed request with its own original successor. Some roots are offered earlier than other sibling roots with a compatible original checkpoint. The later root and successor waves are fixed. Every successor still waits for its own original predecessor."
    screen["planned_root_order"] = [*first, *later]
    screen["planned_successor_order"] = successors
    screen["projected_root_P"] = {key: f"P{role}" for key, role in parent_roles[0].items()}
    screen["projected_successor_P"] = {key: f"P{role}" for key, role in routes(successors, weights).items()}
    screen["full_GET_by_relative_successor_first_P"] = dict(zip(("same_as_roots", "opposite_to_roots"), successor_read[:2], strict=True))
    screen["full_restore_upper_bytes"] += sum(row["potential_GET_bytes"] for rows in guessed.values() for row in rows)
    screen["new_order_estimated_GET_bytes"] = min(read_by_orientation) + min(successor_read)
    matched_checkpoints = {task: [{"earlier_root": row["root"], "later_root": row["successor"],
                                  "checkpoint_blocks": row["checkpoint_blocks"], "potential_GET_bytes": row["potential_GET_bytes"]}
                                 for row in rows] for task, rows in guessed.items()}
    workload["root_batch_calibration"] = {"first_roots": first, "later_roots": later, "checkpoint_matches": matched_checkpoints,
                                          "first_root_cold_token_upper_by_task": cold_prefix_bound([row for row in pairs if row["root"] in seed_ids], ledger),
                                          "projected_first_root_footprint_by_P": projected_cold,
                                          "moved_to_later_roots_for_prefill_balance": moved_to_later,
                                          "later_root_offer_start_s": late_root_s, "successor_offer_start_s": successor_s,
                                          "later_root_full_GET_by_first_role": read_by_orientation,
                                          "successor_full_GET_by_two_phase_choices": successor_read,
                                          "assumptions": "One potential full remote read per distinct original checkpoint. Earlier requests must publish this state before the later root needs it. Native service and APC may change the measured amount."}
    selection_receipt = workload["stable_checkpoint_calibration"]
    workload["stable_checkpoint_calibration"] = {
        "reference_input_trace_sha256": selection_receipt["reference_input_trace_sha256"],
        "max_natural_observed_tokens_by_role": selection_receipt["max_natural_observed_tokens_by_role"],
        "reference_projected_GET_bytes": {"sum_of_phase_minima": min(read_by_orientation) + min(successor_read),
                                          "sum_of_phase_maxima": max(read_by_orientation) + max(successor_read)},
    }
    return workload


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("ledger", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--later-root-s", type=float, required=True)
    cli.add_argument("--successor-s", type=float, required=True)
    cli.add_argument("--reference", type=Path, action="append", default=[])
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError("The selected replay plan already exists")
    result = make_plan(args.archive, args.reference, args.ledger, args.later_root_s, args.successor_s)
    write_json(args.output, result)
    print(result["sessions"], result["root_batch_calibration"])
