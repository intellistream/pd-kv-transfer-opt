"""Select source families with high repeated read volume and low cold-prefix work."""

import argparse
from collections import defaultdict
from pathlib import Path
from random import Random

from pd_kv_transfer_opt.common import read_jsonl, write_json
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS
from pd_kv_transfer_opt.workloads.terminal_restore_checkpoints import plan
from pd_kv_transfer_opt.workloads.terminal_restore_order import prefill_weights
from pd_kv_transfer_opt.workloads.terminal_restore_stable import (
    guard_cold_arrivals, predict, proposed_orders, read_reference,
)


FAMILIES = ("caffe-cifar-10", "extract-moves-from-video", "financial-document-processor", "make-mips-interpreter")


def cold_prefix_bound(pairs, ledger):
    return cold_prefix_bound_from_rows(pairs, read_jsonl(ledger))


def cold_prefix_bound_from_rows(pairs, source_rows):
    roots = defaultdict(set)
    for pair in pairs:
        roots[pair["task"]].add(pair["root"])
    selected = defaultdict(list)
    for row in source_rows:
        if row["task"] in roots:
            selected[row["task"]].append(row)
    result = {}
    for name, rows in selected.items():
        rows.sort(key=lambda row: row["step_id"])
        previous, total = None, 0
        for index, row in enumerate(rows):
            if row["id"] not in roots[name]:
                continue
            lcp = (min(r["source_prompt_lcp_to_previous"] for r in rows[previous + 1:index + 1])
                   if previous is not None else 0)
            total += row["prompt_tokens"] - lcp // ATTENTION_BLOCK_TOKENS * ATTENTION_BLOCK_TOKENS
            previous = index
        result[name] = total
    return result


def model_reference(rows, pairs, cold_offers, scale, restore_scale):
    view = {}
    for pair in pairs:
        root, child = pair["root"], pair["successor"]
        original = rows[root]
        ready = cold_offers[root] + (original["P_end_s"] - original["P_start_s"]) * scale
        view[root] = {**original, "P_credit_release_s": ready + original["P_credit_release_s"] - original["P_end_s"],
                      "end_s": ready + original["end_s"] - original["P_end_s"]}
        original = rows[child]
        reserved = original["P_credit_release_s"] - original["P_start_s"]
        view[child] = {**original, "P_credit_release_s": original["P_start_s"] + restore_scale * reserved}
    return view


def refine_order(order, volumes, by_child, forecast, score):
    """Improve the common arrival order while preserving each source family's order."""
    rng = Random(20261127)
    best = score(volumes)
    for _ in range(1400):
        i = rng.randrange(len(order) - 1)
        first = by_child[order[i]]["task"]
        j = rng.randrange(i + 1, min(i + 6, len(order)))
        second = by_child[order[j]]["task"]
        if first == second or any(by_child[key]["task"] in (first, second) for key in order[i + 1:j]):
            continue
        alternate = list(order)
        alternate[i], alternate[j] = alternate[j], alternate[i]
        candidate = forecast(alternate)
        ranked = score(candidate)
        if ranked > best:
            order, volumes, best = alternate, candidate, ranked
    return order, volumes


def make_plan(parent_archive, additional_references, ledger, release_s, *, limit=600, successor_limit=None):
    parent, prior, receipt_sha = read_reference(parent_archive)
    hashes = {parent_archive.name: receipt_sha}
    spec_by_id = {row["id"]: row for row in parent["request_sources"]}
    historical_outputs = defaultdict(list)
    for rows in prior.values():
        for key, row in rows.items():
            historical_outputs[key].append((row["finish_reason"], row["output_tokens"]))
    for archive in additional_references:
        reference, rows_by_case, digest = read_reference(archive)
        if reference["source_revision"] != parent["source_revision"] or reference["parent_requests_sha256"] != parent["parent_requests_sha256"]:
            raise ValueError("Dense cohort references must have identical parent input data")
        hashes[archive.name] = digest
        for row in reference["request_sources"]:
            original = spec_by_id[row["id"]]
            if (row["prompt_sha256"], row["predecessor"], row["max_tokens"]) != (original["prompt_sha256"], original["predecessor"], original["max_tokens"]):
                raise ValueError(f"A reference changed the original input {row['id']}")
        for case in rows_by_case.values():
            for key, view in case.items():
                historical_outputs[key].append((view["finish_reason"], view["output_tokens"]))

    max_tokens = {"root": limit, "successor": limit if successor_limit is None else successor_limit}
    pairs = [pair for pair in parent["selection"]["pairs"] if pair["task"] in FAMILIES
             and all(reason == "stop" and 0 < tokens <= max_tokens[edge] for edge in ("root", "successor")
                     for reason, tokens in historical_outputs[pair[edge]])]
    bounds = cold_prefix_bound(pairs, ledger)
    reference_bounds = cold_prefix_bound(parent["selection"]["pairs"], ledger)
    if set(bounds) != set(FAMILIES):
        raise ValueError("The natural output filter left a selected family empty")
    keys = {pair[role] for pair in pairs for role in ("root", "successor")}
    selection = {**parent["selection"], "pairs": pairs, "shared_root_tokens_upper_by_task": bounds}
    spec = {"source_repo": parent["source_repo"], "source_revision": parent["source_revision"],
            "tokenizer_sha256": parent["tokenizer_fingerprint"], "selection": selection,
            "requests": [{**row, "offered_s": row["original_offered_s"], "token_ids_uint32le_sha256": row["prompt_sha256"]}
                         for row in parent["request_sources"] if row["id"] in keys]}
    sizes = {key: size for key, size in parent["arrival_order_screen"]["wire_body_bytes"].items() if key in keys}
    weights = prefill_weights(sizes)
    fit = parent["arrival_order_screen"]["admission_fit"]
    receipt = {"wire_body_bytes": sizes, "sha256": parent["parent_requests_sha256"]}
    provisional = plan(spec, receipt, release_s, admission_fit=fit)
    guard_cold_arrivals(provisional)
    cold_offers = {row["id"]: row["offered_s"] for row in provisional["request_sources"] if row["predecessor"] is None}
    base = provisional["arrival_order_screen"]
    root_roles = {key: int(role[1]) for key, role in base["projected_root_P"].items()}
    by_child = {pair["successor"]: pair for pair in pairs}
    scale = sum(bounds.values()) / sum(reference_bounds.values())
    scenarios = {f"{case}/{sensitivity}": model_reference(rows, pairs, cold_offers, scale * factor, restore_scale)
                 for case, rows in prior.items() for sensitivity, factor, restore_scale in (("fast", .85, .65), ("slow", 1.2, .9))}

    def forecast(order):
        return {name: min(predict(order, pairs, by_child, view, weights, root_roles, release_s, idle_role=role)
                          for role in (0, 1)) for name, view in scenarios.items()}

    def score(amounts):
        low, high = min(amounts.values()), max(amounts.values())
        return low - .5 * (high - low), low

    initial = base["planned_successor_order"]
    options = proposed_orders(pairs, weights, root_roles, scenarios, release_s, initial, ready_first=True)
    choice, volumes = max(((order, forecast(order)) for order in options), key=lambda item: score(item[1]))
    choice, volumes = refine_order(choice, volumes, by_child, forecast, score)
    frozen = plan(spec, receipt, release_s, admission_fit=fit, ordered_successors=choice)
    guard_cold_arrivals(frozen)
    frozen["stable_checkpoint_calibration"] = {
        "reference_projected_GET_bytes": volumes, "reference_neutral_GET_bytes": forecast(initial),
        "reference_input_trace_sha256": hashes, "max_natural_observed_tokens_by_role": max_tokens,
        "reference_cold_prefix_tokens_upper": sum(reference_bounds.values()), "selected_cold_prefix_tokens_upper": sum(bounds.values()),
        "root_timing_sensitivity_multipliers": [.85, 1.2],
        "reference_on_time_full_read_bytes_by_case": {name: sum(p["potential_GET_bytes"] for p in pairs
                                                                if view[p["root"]]["end_s"] <= release_s - .15)
                                                       for name, view in scenarios.items()},
        "reference_successor_output_tokens_by_case": {case: sum(rows[p["successor"]]["output_tokens"] for p in pairs)
                                                      for case, rows in prior.items()},
        "forecast_assumptions": "Scale P root residence by the relative unique prefix footprint and vary it around that estimate. Preserve observed post-P time of the parent. Evaluate both native idle tie directions and a pair of historical reservation lifetime scales for the recovery wave.",
    }
    frozen["arrival_policy"] = "Replay original two-turn windows with high shared-prefix read volume per cold token. Schedule the recovery arrivals from the original parent timing and a footprint sensitivity analysis. Retain real predecessor release, native routing, original output budgets and natural EOS."
    return frozen


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("ledger", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--release-s", type=float, required=True)
    cli.add_argument("--reference", type=Path, action="append", default=[])
    cli.add_argument("--max-observed-root", type=int, default=600)
    cli.add_argument("--max-observed-successor", type=int)
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError("The selected replay plan already exists")
    result = make_plan(args.archive, args.reference, args.ledger, args.release_s, limit=args.max_observed_root,
                       successor_limit=args.max_observed_successor)
    write_json(args.output, result)
    print(result["sessions"], result["stable_checkpoint_calibration"])
