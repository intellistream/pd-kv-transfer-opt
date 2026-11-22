"""Screen original Terminal-Bench windows for an r28-sized bandwidth effect on CPU."""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
from itertools import pairwise, product
from math import ceil
from pathlib import Path
from statistics import median, median_low
from typing import NamedTuple

from pd_kv_transfer_opt.common import read_json, read_jsonl, sha256, write_json
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS, ATTENTION_BYTES, STATE_BYTES
from pd_kv_transfer_opt.simulator.rdma_replay import PORTS, analyze, egress_epochs, load_native_archive
from pd_kv_transfer_opt.simulator.terminal_pressure import historical_r28, planned_recovery_start, screen
from pd_kv_transfer_opt.workloads.terminal_restore_checkpoints import root_order
from pd_kv_transfer_opt.workloads.terminal_restore_dense import cold_prefix_bound, cold_prefix_bound_from_rows
from pd_kv_transfer_opt.workloads.terminal_restore_expanded import arrange_successors
from pd_kv_transfer_opt.workloads.terminal_restore_order import prefill_weights, routes
from pd_kv_transfer_opt.workloads.terminal_restore_pairs import historical_outputs
from pd_kv_transfer_opt.workloads.terminal_restore_stable import guard_cold_arrivals, read_reference


class Checkpoints(NamedTuple):
    cold_tokens: int
    potential_bytes: int
    successor_output_tokens: int
    pairs: tuple[dict, ...]


SUCCESSOR_FIRST_GAP_S = .11
SUCCESSOR_GAP_S = .022


def source_options(ledger, source_counts, outcomes, body_sizes, eos_limit):
    """Enumerate subranges of original checkpoints to avoid retaining an expensive prefix reset."""
    grouped = defaultdict(list)
    for row in ledger.values():
        grouped[row["task"]].append(row)
    eligible = {key: median_low([tokens for _, tokens in outcomes[key]]) for key, row in ledger.items()
                if key in body_sizes and not row["truncated"] and type(source_counts[key]) is int
                and 0 < source_counts[key] and row["prompt_tokens"] + max(8192, source_counts[key]) <= 262_144
                and len(outcomes.get(key, ())) >= 4
                and all(reason == "stop" and 0 < count <= eos_limit for reason, count in outcomes[key])}
    result = {}
    for task, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["step_id"])
        eligible_edges = []
        for index, (root, child) in enumerate(pairwise(rows)):
            root_key, key = root["id"], child["id"]
            lcp = child["source_prompt_lcp_to_previous"]
            if (root_key in eligible and key in eligible and child["predecessor"] == root_key
                    and lcp >= max(65_536, .9 * root["prompt_tokens"])):
                blocks = lcp // ATTENTION_BLOCK_TOKENS
                pair = {"task": task, "root": root_key, "successor": key, "checkpoint_blocks": blocks,
                        "potential_GET_bytes": blocks * ATTENTION_BYTES + STATE_BYTES,
                        "estimated_root_output_tokens": eligible[root_key],
                        "estimated_successor_output_tokens": eligible[key]}
                eligible_edges.append((index, blocks, pair))
        found = {}
        for first in range(len(eligible_edges)):
            for last in range(first + 1, len(eligible_edges) + 1):
                chosen, used_blocks = [], set()
                previous = len(rows)
                for index, blocks, pair in reversed(eligible_edges[first:last]):
                    if index < previous - 1 and blocks not in used_blocks:
                        chosen.append((index, pair))
                        previous = index
                        used_blocks.add(blocks)
                chosen.reverse()
                pairs = tuple(pair for _, pair in chosen)
                keys = tuple(pair["successor"] for pair in pairs)
                if keys in found:
                    continue
                cold = rows[chosen[0][0]]["prompt_tokens"]
                for (left, _), (right, _) in pairwise(chosen):
                    shared = min(row["source_prompt_lcp_to_previous"] for row in rows[left + 1:right + 1])
                    cold += rows[right]["prompt_tokens"] - shared // ATTENTION_BLOCK_TOKENS * ATTENTION_BLOCK_TOKENS
                found[keys] = Checkpoints(cold, sum(pair["potential_GET_bytes"] for pair in pairs),
                                          sum(pair["estimated_successor_output_tokens"] for pair in pairs), pairs)
        if found:
            result[task] = list(found.values())
    return result


def select_groups(options, budgets, decode_cost_bytes, bucket=4_000):
    """Choose at most one range per task under two cold-prefill budgets."""
    states = {(0, 0): (0, ())}
    maximum = max(budgets) // bucket
    for variants in options.values():
        by_cost = {}
        for variant in variants:
            cost = ceil(variant.cold_tokens / bucket)
            value = variant.potential_bytes - decode_cost_bytes * variant.successor_output_tokens
            if cost <= maximum and value > by_cost.get(cost, (0, None))[0]:
                by_cost[cost] = (value, variant)
        frontier, best = [], 0
        for cost, (value, variant) in sorted(by_cost.items()):
            if value > best:
                frontier.append((cost, value, variant))
                best = value
        updated = states.copy()
        for (left, right), (score, selected) in states.items():
            for cost, value, variant in frontier:
                for place in {(left + cost, right), (left, right + cost)}:
                    key = tuple(sorted(place))
                    candidate = score + value
                    if key[1] <= maximum and candidate > updated.get(key, (0, ()))[0]:
                        updated[key] = (candidate, (*selected, variant))
        states = updated
    return {limit: max((value for used, value in states.items() if used[1] * bucket <= limit),
                       key=lambda result: result[0])[1] for limit in budgets}


def observed_waves(paths, ledger, d_rate):
    """Measure preparation, egress and finish residual on ready, natural-EOS reference waves."""
    waves = {}
    for path, order in paths:
        manifest, trace, arms = load_native_archive(path)
        spec = manifest["workload"]
        pairs = spec["selection"]["pairs"]
        high = arms[f"{order}-cap200"]
        low = arms[f"{order}-cap100"]
        rows = high[1]
        roots = [row for row in rows.values() if row["predecessor"] is None]
        children = [row for row in rows.values() if row["predecessor"] is not None]
        ready = max(row["end_s"] for arm in (high, low) for row in arm[1].values() if row["predecessor"] is None)
        start = planned_recovery_start(rows)
        if ready > start or any(row["finish_reason"] != "stop" for arm in (high, low) for row in arm[1].values()):
            raise ValueError(f"Expected all-original EOS and a ready predecessor wave in {path.name}/{order}")
        cold = {p: sum(cold_prefix_bound([pair for pair in pairs if rows[pair["root"]]["P"] == p], ledger).values())
                for p in ("P0", "P1")}
        last_root_p, last_child_p = (max(row["P_end_s"] for row in part) for part in (roots, children))
        epochs = egress_epochs(high)
        by_port = [sum(max(0.0, right - max(left, start)) * rates[port] / 8
                       for left, right, rates in epochs) for port in range(PORTS)]
        wire, service_s = sum(by_port), 8 * max(by_port) / 25
        full_gb = spec["arrival_order_screen"]["full_GET_by_relative_successor_first_P"]["same_as_roots"] / 1e9
        decode_s = sum(row["output_tokens"] for row in children) / d_rate
        _, credits, _ = read_reference(path)
        high_credits = credits[f"{order}-cap200"]
        first_P = min(row["P_start_s"] for row in children)
        warm_credit_s = min(row["P_credit_release_s"] for row in high_credits.values() if row["predecessor"]) - first_P
        waves[f"{path.name}/{order}"] = {
            "trace_sha256": trace, "cold_tokens_by_P": cold, "effective_cold_tokens_s_per_P": max(cold.values()) / last_root_p,
            "both_arms_latest_root_EOS_s": ready, "first_successor_offer_s": start,
            "root_EOS_slack_over_high_last_P_s": ready - last_root_p,
            "high_D_egress_GB": wire, "planned_best_tie_full_read_GB": full_gb,
            "high_wire_fraction_of_plan": wire / full_gb, "hottest_port_ratio_to_mean": PORTS * max(by_port) / wire,
            "high_first_successor_P_credit_duration_s": warm_credit_s,
            "high_child_P_span_s": last_child_p - start, "high_port_service_s": service_s,
            "high_nonwire_P_s_per_successor": max(0.0, last_child_p - start - service_s) / len(children),
            "high_child_Q_over_peak_decode_s": decode_s,
            "high_finish_residual_after_P_or_peak_decode_s": max(row["end_s"] for row in children) - start
            - max(last_child_p - start, decode_s),
        }
    return waves


def wave_times(rows, bytes_gb, cold_tokens, costs, throughput_reference, *, favorable=False):
    """Estimate a whole wave with common preparation and overlapping P delivery and D work."""
    n = len(rows)
    root_q = sum(row["estimated_root_output_tokens"] for row in rows)
    child_q = sum(row["estimated_successor_output_tokens"] for row in rows)
    prep = cold_tokens / costs["cold_tokens_s"] + costs["root_slack_s"]
    prep += .5  # Predecessors must actually finish before a compact successor wave.
    share = 1.0 if favorable else costs["wire_fraction"]
    wire_gb = bytes_gb * share
    nonwire = n * costs["P_nonwire_per_request_s"]
    compute = child_q / costs["peak_D_tokens_s"]
    ports = {str(cap): 8 * wire_gb * costs["port_skew"] / cap for cap in (100, 200)}
    delivery = {cap: nonwire + time for cap, time in ports.items()}
    wave = {cap: max(time, compute) for cap, time in delivery.items()}
    exposure = 1.0 if favorable else costs["exposed_fraction"]
    high = prep + wave["200"] + costs["final_D_residual_s"]
    low = high + exposure * (wave["100"] - wave["200"])
    return {"estimated_D_egress_GB": wire_gb, "planned_successor_start_s": prep,
            "max_serialized_per_port_service_s": ports, "common_nonwire_P_s": nonwire,
            "P_delivery_s": delivery, "successor_peak_D_service_s": compute,
            "100_P_lags_peak_D_while_200_keeps_up": delivery["200"] <= compute < delivery["100"],
            "finish_gap_exposure_fraction": exposure, "final_D_residual_s": costs["final_D_residual_s"],
            "estimated_critical_finish_gap_s": low - high,
            "needed_gap_to_match_r28_smaller_loss_s": high * throughput_reference / (1 - throughput_reference),
            "estimated_100_low_throughput_loss_fraction": 1 - high / low,
            "arms": {cap: {"output_tokens": root_q + child_q, "duration_s": finish,
                           "throughput_tokens_s": (root_q + child_q) / finish}
                     for cap, finish in (("100", low), ("200", high))}}


def magnitude_requirements(shape, port_skew, target):
    """Invert the wave model for read volume and P headroom at the target loss."""
    common = shape["planned_successor_start_s"] + shape["final_D_residual_s"]
    compute = shape["successor_peak_D_service_s"]
    nonwire = shape["common_nonwire_P_s"]
    exposure = shape["finish_gap_exposure_fraction"]
    high_cap_s_per_gb = 8 * port_skew / 200
    gap_factor = target / (1 - target)
    # The minimum byte count has the low P past D and either D or P setting the high wave time.
    bytes_when_200_keeps_up = ((common + compute) * gap_factor / exposure + compute - nonwire) / (2 * high_cap_s_per_gb)
    bytes_at_200_headroom = max(0.0, compute - nonwire) / high_cap_s_per_gb
    if bytes_when_200_keeps_up <= bytes_at_200_headroom:
        required_bytes = bytes_when_200_keeps_up
    elif exposure > gap_factor:
        required_bytes = (common + nonwire) * gap_factor / ((exposure - gap_factor) * high_cap_s_per_gb)
    else:
        required_bytes = None
    return {"min_actual_D_egress_GB_at_same_preparation_and_output": required_bytes,
            "max_actual_D_egress_GB_before_200_P_delivery_falls_behind_peak_D": bytes_at_200_headroom,
            "common_time_s_to_remove_at_same_work_and_finish_gap": max(
                0.0, shape["arms"]["200"]["duration_s"] - shape["estimated_critical_finish_gap_s"] / gap_factor)}


def ingress_span(count):
    return SUCCESSOR_FIRST_GAP_S + (count - 2) * SUCCESSOR_GAP_S if count > 1 else 0.0


def source_projection(groups, cold_by_roots, weights, costs, reference, task_roles=None):
    pairs = [pair for variant in groups for pair in variant.pairs]
    grouped = {variant.pairs[0]["task"]: sorted(variant.pairs, key=lambda row: (row["checkpoint_blocks"], row["root"]))
               for variant in groups}
    roots = root_order(grouped, weights, task_roles)
    placement = routes(roots, weights)
    cold = [sum(cold_by_roots(task, tuple(pair["root"] for pair in group if placement[pair["root"]] == p))
                for task, group in grouped.items()) for p in (0, 1)]
    # The native priority heap breaks equal scores by its registered ordinal.
    order, preferred = arrange_successors(grouped, placement, weights)
    total = sum(pair["potential_GET_bytes"] for pair in pairs)
    return {"sessions": len(pairs), "tasks": sorted(grouped), "native_idle_projected_cold_tokens_by_P": cold,
            "projected_root_P_finish_separation_s": abs(cold[1] - cold[0]) / costs["cold_tokens_s"],
            "body_adjusted_successor_ingress_span_s": ingress_span(len(pairs)),
            "hypothetical_full_read_GB": total / 1e9, "projected_GB_if_preferred_idle_first_P": preferred / 1e9,
            "projected_GB_if_other_idle_first_P": (total - preferred) / 1e9,
            "preferred_idle_first_P": "P0",
            "observed_reference_shape": wave_times(pairs, preferred / 1e9, max(cold), costs, reference),
            "all_projected_reads_fully_exposed": wave_times(pairs, preferred / 1e9, max(cold), costs, reference, favorable=True),
            "pairs": pairs, "planned_root_order": roots, "planned_successor_order": order}


def native_order_sensitivity(ledger, counts, outcomes, sizes, cold_by_roots, weights, costs, target, credit_window_s):
    """Repeat the selection with a finer grid and alternate original root orderings."""
    budgets = [200_000, 225_000, 250_000, 275_000, 300_000, 325_000, 350_000, 375_000,
               400_000, 450_000, 475_000, 550_000]
    seen, best, feasible = set(), {}, defaultdict(int)
    root_orders = 0
    for cutoff in (550, 600, 700, 750, 900, 1050, 1200, 1400):
        options = source_options(ledger, counts, outcomes, sizes, cutoff)
        for penalty in range(0, 18_000_001, 3_000_000):
            for budget, groups in select_groups(options, budgets, penalty).items():
                pairs = [pair for group in groups for pair in group.pairs]
                keys = tuple(sorted(pair["root"] for pair in pairs))
                if len(groups) < 2 or ingress_span(len(pairs)) >= credit_window_s or keys in seen:
                    continue
                seen.add(keys)
                for roles in product((0, 1), repeat=len(groups) - 1):
                    if not any(roles):
                        continue
                    candidate = source_projection(groups, cold_by_roots, weights, costs, target, (0, *roles))
                    root_orders += 1
                    for shape in ("observed_reference_shape", "all_projected_reads_fully_exposed"):
                        timing = candidate[shape]
                        if not timing["100_P_lags_peak_D_while_200_keeps_up"]:
                            continue
                        feasible[shape] += 1
                        loss = timing["estimated_100_low_throughput_loss_fraction"]
                        if shape not in best or loss > best[shape][shape]["estimated_100_low_throughput_loss_fraction"]:
                            best[shape] = {"max_prior_natural_output": cutoff, "byte_penalty_per_output_token": penalty,
                                           "grouping_cold_search_budget_tokens_per_P": budget, **candidate}
    return {"distinct_cohorts": len(seen), "tested_root_orderings": root_orders, "passing_shapes": dict(feasible),
            "best": best}


def compact_replays(reports):
    result = {}
    for name, archive in reports.items():
        for order, pair in archive["orders"].items():
            result[f"{name}/{order}"] = {key: pair[key] for key in (
                "requests", "same_observed_P", "same_observed_D", "comparable_P_routes_wire_bytes_and_EOS", "hardware",
                "hardware_low_throughput_loss_fraction", "hardware_finish_gap_s")}
            for model in ("prediction", "short_burst_sensitivity"):
                result[f"{name}/{order}"][model] = {"loss_fraction": pair[model]["low_throughput_loss_fraction"],
                                                    "finish_gap_s": pair[model]["finish_gap_s"],
                                                    "arms": {cap: {key: pair[model][cap][key] for key in (
                                                        "output_tokens", "duration_s", "drained_throughput_tokens_s",
                                                        "D_TX_GB_in_P_span")} for cap in ("100", "200")}}
    return result


def candidate_manifest(candidate, ledger, counts, sizes, fit, meta, r78):
    start = candidate["observed_reference_shape"]["planned_successor_start_s"]
    roots, successors = (candidate[key] for key in ("planned_root_order", "planned_successor_order"))
    first = sizes[successors[0]]
    offers = {key: start + ingress_span(index + 1) + fit["s_per_byte"] * (first - sizes[key])
              for index, key in enumerate(successors)}
    correction = start - min(offers.values())
    parent = {pair["successor"]: pair["root"] for pair in candidate["pairs"]}
    workload = {**meta, "status": "prospective CPU input; no outcome for this combined workload",
                "ignore_eos": False, "temperature": 0, "route_policy": "upstream native without route override",
                "arrival_order_screen": {"planned_root_order": roots, "wire_body_bytes": {k: sizes[k] for k in roots + successors},
                                         "admission_fit": fit}, "request_sources": []}
    for key in roots + successors:
        row = ledger[key]
        workload["request_sources"].append({"id": key, "session_id": parent.get(key, key), "predecessor": parent.get(key),
                                           "step_id": row["step_id"], "source_predecessor": row["predecessor"],
                                           "prompt_sha256": row["token_ids_uint32le_sha256"], "prompt_tokens": row["prompt_tokens"],
                                           "tool_gap_s": 0, "max_tokens": max(8192, counts[key]),
                                           "offered_s": offers[key] + correction if key in parent else 0.0})
    guard_cold_arrivals(workload)
    workload["successor_offer_span_s"] = max(offers.values()) - min(offers.values())
    workload["body_adjusted_successor_ingress_span_s"] = ingress_span(len(successors))
    workload["r78_first_credit_return_after_successor_ingress_s"] = r78["high_first_successor_P_credit_duration_s"]
    workload["order_sensitivity"] = ("The compact offer clock is shared by both arms. Actual predecessor EOS, "
                                      "ingress variation and early P credit returns can change the native order. "
                                      "The static idle-route estimate assumes all offers before any credit return.")
    return workload


def run(args):
    plan = read_json(args.plan)
    archive_root = args.results
    _, scanned = screen(archive_root, plan)
    calibration = [archive_root / name for name in plan["decode_calibration"]]
    other = [archive_root / row["archive"] for row in scanned if row["archive"] not in plan["decode_calibration"]]
    replay = analyze(calibration, other, [], [1.0])
    reference = historical_r28(archive_root / plan["historical_comparison"])
    target = min(row["loss_100_relative_to_200"] for row in reference["orders"].values())
    ledger = {row["id"]: row for row in read_jsonl(args.ledger)}
    receipt, source = map(read_json, (args.eos_receipt, args.source_completions))
    if not receipt["source_revision"] == source["source_revision"] == plan["source_revision"] or (
        receipt["ledger_sha256"] != sha256(args.ledger) or ledger.keys() != source["source_completion_tokens"].keys()
    ):
        raise ValueError("Input fingerprints, source budget and EOS receipts must use one source revision")
    history_paths = [archive_root / name for name in receipt["evidence"]]
    outcomes, evidence = historical_outputs(history_paths, ledger, plan["source_revision"])
    if evidence != receipt["evidence"] or {key: [list(row) for row in rows] for key, rows in outcomes.items()} != receipt["outcomes"]:
        raise ValueError("EOS receipts do not match re-extracted original events")
    sizes = {}
    for archive in history_paths:
        workload = read_json(archive / "manifest.json")["workload"]
        for key, size in workload.get("arrival_order_screen", {}).get("wire_body_bytes", {}).items():
            if key in ledger:
                if key in sizes and sizes[key] != size:
                    raise ValueError(f"The exact source request body size changed for {key}")
                sizes[key] = size
    d_fit = replay["decode_service_fit"]
    d_rate = 32 / (d_fit["first_sequence_period_s"] + 15 * d_fit["additional_sequence_period_s"])
    anchors = [(archive_root / "2026-11-22-r73-terminal-source-restore-expanded", "b2"),
               (archive_root / "2026-11-22-r76-terminal-source-restore-density-measured", "b2"),
               (archive_root / "2026-11-22-r78-terminal-source-restore-stable-tail", "b1"),
               (archive_root / "2026-11-22-r78-terminal-source-restore-stable-tail", "b2")]
    waves = observed_waves(anchors, args.ledger, d_rate)
    r78_key = f"{anchors[2][0].name}/b1"
    anchor = waves[r78_key]
    r78_pair = replay["holdouts"][anchors[2][0].name]["orders"]["b1"]
    reference_service_delta = (max(anchor["high_child_P_span_s"] + anchor["high_port_service_s"],
                                   anchor["high_child_Q_over_peak_decode_s"])
                               - max(anchor["high_child_P_span_s"], anchor["high_child_Q_over_peak_decode_s"]))
    costs = {"peak_D_tokens_s": d_rate,
             "cold_tokens_s": median(w["effective_cold_tokens_s_per_P"] for w in waves.values()),
             "root_slack_s": median(w["root_EOS_slack_over_high_last_P_s"] for w in waves.values()),
             "wire_fraction": median(w["high_wire_fraction_of_plan"] for w in waves.values()),
             "port_skew": median(w["hottest_port_ratio_to_mean"] for w in waves.values()),
             "P_nonwire_per_request_s": median(w["high_nonwire_P_s_per_successor"] for w in waves.values()),
             "final_D_residual_s": median(w["high_finish_residual_after_P_or_peak_decode_s"] for w in waves.values()),
             "exposed_fraction": r78_pair["measured_P_only_fixed_reference_output_and_D_finish_gap_s"] / reference_service_delta}
    counts = source["source_completion_tokens"]
    weights = prefill_weights(sizes)
    ledger_by_task = defaultdict(list)
    for row in ledger.values():
        ledger_by_task[row["task"]].append(row)

    @lru_cache(maxsize=None)
    def cold_by_roots(task, roots):
        return sum(cold_prefix_bound_from_rows([{"task": task, "root": key} for key in roots], ledger_by_task[task]).values())

    budgets = [250_000, 300_000, 350_000, 400_000, 475_000, 550_000, 650_000]
    proposals = []
    for cutoff in (750, 1050, 1400):
        options = source_options(ledger, counts, outcomes, sizes, cutoff)
        for cost in (0, 6_000_000, 12_000_000):
            for budget, groups in select_groups(options, budgets, cost).items():
                if len(groups) < 2:
                    continue
                candidate = source_projection(groups, cold_by_roots, weights, costs, target)
                proposals.append({"source_observed_EOS_max_tokens": cutoff, "selection_penalty_bytes_per_D_token": cost,
                                  "grouping_cold_search_budget_tokens_per_P": budget, **candidate})
    def lean(row):
        return {key: val for key, val in row.items() if key not in ("pairs", "planned_root_order", "planned_successor_order")}

    def best_shape(shape, *, require_ingress_headroom):
        passing = (row for row in proposals if row[shape]["100_P_lags_peak_D_while_200_keeps_up"]
                   and (not require_ingress_headroom
                        or row["body_adjusted_successor_ingress_span_s"] < anchor["high_first_successor_P_credit_duration_s"]))
        return max(passing, key=lambda row: row[shape]["estimated_100_low_throughput_loss_fraction"])

    measured_shape = "observed_reference_shape"
    favorable_shape = "all_projected_reads_fully_exposed"
    selected = best_shape(favorable_shape, require_ingress_headroom=True)
    calibrated = best_shape(measured_shape, require_ingress_headroom=True)
    largest = max(proposals, key=lambda row: row["hypothetical_full_read_GB"])
    broad = native_order_sensitivity(ledger, counts, outcomes, sizes, cold_by_roots, weights, costs, target,
                                     anchor["high_first_successor_P_credit_duration_s"])
    broad["best"] = {shape: lean(row) for shape, row in broad["best"].items()}
    ref78 = read_json(anchors[2][0] / "manifest.json")
    w78 = ref78["workload"]
    metadata = {"baseline_commit": plan["baseline_commit"], "source_repo": w78["source_repo"],
                "source_revision": plan["source_revision"], "tokenizer_fingerprint": w78["tokenizer_fingerprint"],
                "source_ledger_sha256": receipt["ledger_sha256"], "source_completions_sha256": sha256(args.source_completions),
                "EOS_receipt_sha256": sha256(args.eos_receipt),
                "baseline_manifest_sha256": sha256(anchors[2][0] / "manifest.json")}
    workload = candidate_manifest(selected, ledger, counts, sizes, w78["arrival_order_screen"]["admission_fit"], metadata, anchor)
    source_file = args.output.with_name("candidate-workload.json")
    write_json(source_file, {**workload, "settings": ref78["settings"], "shaping": ref78["shaping"],
                             "pairs": selected["pairs"], "screening": lean(selected)})
    write_json(args.output, {**metadata, "candidate_workload_sha256": sha256(source_file),
                             "EOS_evidence_archives": evidence, "r28": reference, "r28_smaller_loss_fraction": target,
                             "decode_training_traces": replay["decode_training_trace_sha256"],
                             "heldout_pairs_at_real_traffic": compact_replays(replay["holdouts"]),
                             "ready_wave_calibration": waves, "wave_costs": costs,
                             "selected": lean(selected), "best_calibrated_with_200_headroom": lean(calibrated),
                             "largest_nominal_volume": lean(largest),
                             "selected_candidate_requirements": {
                                 shape: magnitude_requirements(selected[shape], costs["port_skew"], target)
                                 for shape in (measured_shape, favorable_shape)},
                             "source_search": [lean(row) for row in proposals],
                             "native_arrival_order_sensitivity": broad,
                             "scope": ("The trace replays contain actual measured bytes and EOS, with no pressure multiplier. "
                                       "Source projections pick disjoint original adjacent pairs, with at least four observed "
                                       "natural EOS lengths each. The envelope fixes their historical median-low work and "
                                       "models the pressure of their own KV checkpoints using prior realized read fractions, "
                                       "port skew, and P/D timing. Favorable sensitivity assumes every predicted remote read "
                                       "and passes all predicted wave service delta through to drain. This is a conditional "
                                       "planning calculation. APC/Store residency, proxy release order, and future EOS can "
                                       "change for new combinations, especially if new arrivals outlast the first P credit.")})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--plan", type=Path, required=True)
    cli.add_argument("--results", type=Path, default=Path("results"))
    cli.add_argument("--ledger", type=Path, required=True)
    cli.add_argument("--eos-receipt", type=Path, required=True)
    cli.add_argument("--source-completions", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    run(cli.parse_args())


if __name__ == "__main__":
    main()
