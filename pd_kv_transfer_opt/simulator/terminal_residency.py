"""Screen natural Terminal-Bench ranges with earlier P-side APC accounted for."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from math import ceil, isclose
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.profile_closed_sessions import read_trace, requests_for
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS, ATTENTION_BYTES, STATE_BYTES
from pd_kv_transfer_opt.simulator.decode_supply import decode_samples, fit_decode
from pd_kv_transfer_opt.simulator.rdma_replay import drained_measurement, egress_epochs
from pd_kv_transfer_opt.simulator.terminal_magnitude import source_options
from pd_kv_transfer_opt.simulator.terminal_overlap import early_release_probe
from pd_kv_transfer_opt.simulator.terminal_prefill_calibration import (
    clock_profiles, independent_ingress_extrema, paired_quality, r78_replay, response_error,
)
from pd_kv_transfer_opt.simulator.terminal_supply import (
    HOLDOUT, ORDERS, TRAINING, ColdPrefixes, arrange_successors, clock_sensitivity, evaluate, load_source,
    native, observed_wave, offsets_for_order, prefill_weights, reference_state, root_drain, root_order, routes, scored_prefill,
)


R43_METHODS = "2026-11-20-r43-native-decode-apc-methods"
R69_FORMAL = "2026-11-22-r69-terminal-source-restore-short-guard"
R75_DECODE = "2026-11-22-r75-terminal-source-restore-pilot"
EOS_CUTOFF = 675
ROOT_GUARD_S = .60


@dataclass(frozen=True)
class ReplayInputs:
    ledger: dict
    sizes: dict
    outcomes: dict
    fit: dict
    prefixes: ColdPrefixes
    decode: dict
    r69: dict
    profiles: dict
    cold_fraction: float


def seed(path):
    result = read_json(path / "summary.json")["joint_rescreen"]
    plan = path / "rescreened-workload.json"
    if sha256(plan) != result["workload_sha256"]:
        raise ValueError(f"Frozen workload mismatch in {path.name}")
    return read_json(plan), result


def frozen_inputs(workload):
    rows = workload["request_sources"]
    return {"root_offers": {r["id"]: r["offered_s"] for r in rows if not r["predecessor"]},
            "successor_offers": {r["id"]: r["offered_s"] for r in rows if r["predecessor"]},
            "eos_samples": [{r["id"]: r["observed_EOS_reference_tokens"][order] for r in rows} for order in ORDERS]}


def archive_lookup(path, ledger, prefixes):
    """Measure first P lookup coverage on requests an earlier original turn could cover."""
    manifest = read_json(path / "manifest.json")
    spec = manifest["workload"]["request_sources"]
    trace = path / "r43-method-events.jsonl.gz"
    digest = sha256(trace)
    if digest != manifest["files"][trace.name]["sha256"]:
        raise ValueError("The r43 method bundle changed")
    source_ids = {r["id"]: r["id"].removeprefix("natural-") for r in spec
                  if r["id"].removeprefix("natural-") in ledger}
    if any(r["prompt_tokens"] != ledger[source_ids[r["id"]]]["prompt_tokens"] for r in spec if r["id"] in source_ids):
        raise ValueError("Source prompt lengths must agree with the r43 trace")
    cases = {}
    for label, arm in read_trace(trace).items():
        origin, = (row["monotonic_s"] for row in arm["client"] if row["type"] == "client_origin")
        requests = requests_for(arm, spec, origin)
        lookups = {}
        for event in sorted(arm["p_native"], key=lambda row: row["monotonic_ns"]):
            if event["type"] == "pool_match_tokens" and event["instance"] in ("P0", "P1"):
                lookups.setdefault(event["request_id"], event)
        visited = defaultdict(list)
        revisits = []
        for key, row in sorted(requests.items(), key=lambda item: item[1]["P_start_s"]):
            if key not in source_ids:
                continue
            name = source_ids[key]
            task = ledger[name]["task"]
            earlier = visited[task, row["P"]]
            parent = row["predecessor"]
            if parent in source_ids and requests[parent]["P"] != row["P"] and key in lookups:
                event = lookups[key]
                shared, older = max(((prefixes.shared(task, source_ids[previous], name), previous)
                                     for previous in earlier if requests[previous]["P_end_s"] <= row["P_start_s"]
                                     and ledger[source_ids[previous]]["step_id"] < ledger[name]["step_id"]), default=(0, None))
                blocks = min(shared, ledger[name]["source_prompt_lcp_to_previous"]) // ATTENTION_BLOCK_TOKENS
                if blocks >= 32 and event["instance"] == row["P"]:
                    age = row["P_start_s"] - requests[older]["P_end_s"]
                    revisits.append({"id": key, "earlier_id": older, "age_s": age,
                                     "earlier_attention_blocks": blocks, "observed_local_blocks_capped": min(
                                         blocks, event["local_tokens"] // ATTENTION_BLOCK_TOKENS),
                                     "observed_external_blocks": event["external_tokens"] // ATTENTION_BLOCK_TOKENS})
            earlier.append(key)
        cases[label] = {}
        for label_age, rows in (("all_ages", revisits), ("within_60_s", [row for row in revisits if row["age_s"] <= 60])):
            offered = sum(r["earlier_attention_blocks"] for r in rows)
            local = sum(r["observed_local_blocks_capped"] for r in rows)
            cases[label][label_age] = {"requests": len(rows), "possible_blocks": offered, "observed_local_blocks": local,
                                       "local_fraction_of_earlier_blocks": local / offered if offered else None,
                                       "rows": rows}
    return {"sha256": digest, "cases": cases}


def triple_bounds(options, ledger):
    """Bound additional third-turn GETs when the first P retains the earlier prefix."""
    edges = {row["root"]: row for variants in options.values() for group in variants for row in group.pairs}
    triples = [(first, edges[first["successor"]]) for first in edges.values() if first["successor"] in edges]
    disjoint, seen = [], set()
    for first, second in sorted(triples, key=lambda pair: (pair[0]["task"], ledger[pair[1]["successor"]]["step_id"])):
        ids = (first["root"], second["root"], second["successor"])
        if seen.isdisjoint(ids):
            disjoint.append((first, second))
            seen.update(ids)
    result = {}
    for label, rows in (("all_windows", triples), ("disjoint_windows", disjoint)):
        full = sum(second["potential_GET_bytes"] for _, second in rows)
        retained = sum(max(0, second["checkpoint_blocks"] - first["checkpoint_blocks"]) * ATTENTION_BYTES + STATE_BYTES
                       for first, second in rows)
        result[label] = {"windows": len(rows), "tasks": len({first["task"] for first, _ in rows}),
                         "third_turn_full_checkpoint_GB": full / 1e9, "third_turn_retained_APC_upper_GB": retained / 1e9}
    return result


def archive_coverage(results, x, cases):
    records = {}
    for name, case in cases:
        wave = observed_wave(results / name, case)
        rows = wave["rows"]
        pairs = native(results / name)[0]["workload"]["selection"]["pairs"]
        root_layout = {key: int(row["P"][1:]) for key, row in rows.items() if not row["predecessor"]}
        covered = x.prefixes.earlier_root_attention(pairs, root_layout)
        moved = [p for p in pairs if rows[p["successor"]]["P"] != rows[p["root"]]["P"]]
        records[f"{name}/{case}"] = {"trace_sha256": wave["trace_sha256"], "cross_P": len(moved),
                                    "earlier_other_P_attention_GB": sum(covered[p["successor"]] for p in moved) / 1e9,
                                    "actual_D_egress_GB": wave["wire_GB"]}
    return records


def retention_comparison(workload, source, models, x):
    selected = frozen_inputs(workload)
    stats = []
    for retention in (0.0, .5, 1.0):
        for service, error, refs in models:
            state = clock_sensitivity(selected, workload["pairs"], x.sizes, x.fit, x.prefixes, x.ledger, service, x.decode,
                                      refs, x.profiles, x.cold_fraction, error["r78_positive_first_reply_error_s"],
                                      fixed_wave_start_s=source["successor_wave_start_s"], keep_offers=True,
                                      earlier_root_retention=retention)
            if state is None:
                raise ValueError("The fixed comparison workload must be replayable")
            cases = state["case_pairs"]
            stats.append({"retained_earlier_attention_fraction": retention, "service": service,
                          "quality": paired_quality(cases, len(workload["pairs"]), x.r69["r69_max_200_Gbps_in_20_s"],
                                                    error["r78_positive_finish_gap_error_s"]),
                          "main_ABBA": {order: cases[f"{HOLDOUT}/{order}-cap200/{order}"] for order in ORDERS}})
    return stats


def natural_range_neighbors(workloads, options, x, model):
    """Exchange one original task range and test the nearby natural root layouts."""
    weights = prefill_weights(x.sizes)
    service, _, refs = model
    threshold = x.r69["r69_max_200_Gbps_in_20_s"]
    seen, stats, accepted = set(), Counter(), []
    for workload in workloads:
        original = defaultdict(list)
        for row in workload["pairs"]:
            original[row["task"]].append(row)
        root_offers = frozen_inputs(workload)["root_offers"]
        admissions = {key: t + x.fit["offset_s"] + x.fit["s_per_byte"] * x.sizes[key] for key, t in root_offers.items()}
        old_roles = scored_prefill(admissions, weights, {})
        role_bits = tuple(Counter(old_roles[row["root"]] for row in original[task]).most_common(1)[0][0]
                          for task in sorted(original))
        if role_bits[0]:
            role_bits = tuple(1 - role for role in role_bits)
        permutations = [role_bits] + [tuple(1 - role if i == pos else role for i, role in enumerate(role_bits))
                                      for pos in range(1, len(role_bits))]
        for task, old in original.items():
            for group in options[task]:
                if not max(2, len(old) - 4) <= len(group.pairs) <= len(old) + 5:
                    continue
                groups = {name: sorted(rows, key=lambda p: (p["checkpoint_blocks"], p["root"]))
                          for name, rows in {**original, task: group.pairs}.items()}
                pairs = [p for rows in groups.values() for p in rows]
                key = tuple(sorted(p["root"] for p in pairs))
                if key in seen:
                    continue
                seen.add(key)
                seen_layouts = set()
                for bits in permutations:
                    if not any(bits):
                        continue
                    roots = root_order(groups, weights, bits)
                    order = tuple(roots)
                    if order in seen_layouts:
                        continue
                    seen_layouts.add(order)
                    root_roles = routes(roots, weights)
                    if any(len({root_roles[p["root"]] for p in rows}) > 1 for rows in groups.values()):
                        stats["layouts_with_task_roots_on_both_P"] += 1
                        continue
                    children, total = arrange_successors(groups, root_roles, weights)
                    if 8 * total / 1e9 * min(r["wire_fraction"] for r in refs) / 20 <= threshold:
                        stats["layouts_with_insufficient_potential"] += 1
                        continue
                    stats["native_layouts_evaluated"] += 1
                    layout = (pairs, roots, children, root_roles, x.prefixes.work(roots, root_roles), routes(children, weights))
                    trial = evaluate(layout, x.outcomes, x.ledger, x.sizes, x.fit, service, x.decode, refs, x.r69)
                    if trial is not None and all(trial[k] for k in (
                            "concentration_above_r69", "D_supply_relief_both_roles_and_states",
                            "native_route_arrivals_before_first_score_release", "beats_both_r69_actual_pairs_in_model")):
                        accepted.append((trial["worst_100_low_loss_fraction"], trial, pairs, task))
    return accepted, {"unique_original_cohorts": len(seen), **stats, "nominal_contenders": len(accepted)}


def best_matching_range(choices, models, x):
    """Rank the smaller improvement over the same r69 order after the held-out error."""
    best, completed = None, 0
    for _, trial, pairs, changed in sorted(choices, key=lambda row: row[0], reverse=True):
        verified = []
        for service, error, refs in models:
            timing = clock_sensitivity(trial, pairs, x.sizes, x.fit, x.prefixes, x.ledger, service, x.decode, refs,
                                       x.profiles, x.cold_fraction, error["r78_positive_first_reply_error_s"],
                                       fixed_wave_start_s=verified[0]["wave_start_s"] if verified else None,
                                       earlier_root_retention=1.0)
            if timing is None:
                break
            cases = timing["case_pairs"]
            quality = paired_quality(cases, len(pairs), x.r69["r69_max_200_Gbps_in_20_s"],
                                     error["r78_positive_finish_gap_error_s"])
            limits = independent_ingress_extrema(cases, error["r78_positive_finish_gap_error_s"])
            if not quality["has_comparable_sustained_pressure"] or not all(r["same_P_routes"] and r["same_per_port_GB"]
                                                                         for r in limits.values()):
                break
            margins = {case: row["100_loss_after_r78_error"] - x.r69["actual_r69_loss_by_order"][case]
                       for case, row in limits.items()}
            if best is not None and min(margins.values()) < best[0]:
                break  # The other fitted model can only lower the minimum margin.
            verified.append({**timing, "service": service, "heldout_error": error, "quality": quality,
                             "independent_ingress": limits, "min_margin": min(margins.values()),
                             "main_ABBA": {order: cases[f"{HOLDOUT}/{order}-cap200/{order}"] for order in ORDERS}})
        if len(verified) != len(models):
            continue
        completed += 1
        value = min(model["min_margin"] for model in verified)
        if best is None or value > best[0]:
            best = (value, verified, trial, pairs, changed)
    return best, completed


def freeze_workload(template, pairs, models, selected, x, counts):
    state = models[0]
    for model in models[1:]:
        if any(model[key] != state[key] for key in ("root_offers", "successor_offers")):
            raise ValueError("The fitted layouts must share one offered workload")
    parent = {row["successor"]: row["root"] for row in pairs}
    offers = {**state["root_offers"], **state["successor_offers"]}
    rows = [{**{f: x.ledger[key][f] for f in ("id", "task", "step_id", "prompt_tokens", "token_ids_uint32le_sha256")},
             "predecessor": parent.get(key), "offered_s": t, "tool_gap_s": 0, "max_tokens": max(8192, counts[key]),
             "observed_EOS_reference_tokens": {order: eos[key] for order, eos in zip(
                 ORDERS, selected["eos_samples"], strict=True)}} for key, t in sorted(offers.items(), key=lambda item: (item[1], item[0]))]
    if len(rows) != 2 * len(pairs) or len({r["token_ids_uint32le_sha256"] for r in rows}) != len(rows):
        raise ValueError("Expected disjoint, unmodified original prompt pairs")
    return {**template, "pairs": pairs, "request_sources": rows,
            "maximum_observed_natural_EOS_tokens_at_selection": max(n for key in offers for _, n in x.outcomes[key]),
            "scope": "One frozen natural plan for both original native bandwidth orders. Task ranges are chosen to avoid earlier opposite-P root prefixes."}


def separate_decode_fits(results, services, x):
    """Recalibrate the D shape and r78 error independently of the range selection."""
    scenarios = {}
    for archive in (TRAINING[0][0], R75_DECODE):
        _, digest, arms = native(results / archive)
        decode = fit_decode([row for arm in arms.values() for row in decode_samples(*arm)])
        models = []
        for service in services:
            holdout = {"service": service, "r78_same_P_holdout": r78_replay(results, service, decode)}
            error = response_error(results, holdout, decode)
            refs = [reference_state(observed_wave(results / name, case), x.prefixes, service, decode) for name, case in TRAINING]
            models.append((service, error, refs))
        scenarios[archive] = (replace(x, decode=decode), models, digest)
    return scenarios


def latest_root_eos(selected, x, refs):
    roots = sorted(selected["root_offers"], key=selected["root_offers"].__getitem__)
    weights = prefill_weights(x.sizes)
    ends = []
    for root_residual, _ in x.profiles.values():
        jitter = offsets_for_order(roots, root_residual)
        admissions = {key: t + x.fit["offset_s"] + x.fit["s_per_byte"] * x.sizes[key] + jitter[key]
                      for key, t in selected["root_offers"].items()}
        dest = scored_prefill(admissions, weights, {})
        order = sorted(admissions, key=admissions.__getitem__)
        work = x.prefixes.work(order, dest)
        for ref, eos in zip(refs, selected["eos_samples"], strict=True):
            cold = {**ref, "cold_tokens_s_per_P": ref["cold_tokens_s_per_P"] * x.cold_fraction}
            ends.append(root_drain(order, dest, admissions, work, eos, x.sizes, cold, x.decode)["latest_EOS_s"])
    return max(ends)


def calibrated_replay(selected, pairs, x, models, wave_start):
    trials = []
    for service, error, refs in models:
        state = clock_sensitivity(selected, pairs, x.sizes, x.fit, x.prefixes, x.ledger, service, x.decode,
                                  refs, x.profiles, x.cold_fraction, error["r78_positive_first_reply_error_s"],
                                  fixed_wave_start_s=wave_start, keep_offers=True, earlier_root_retention=1.0)
        if state is None:
            raise ValueError("The frozen guard must clear the previous generation under the calibration")
        cases = state["case_pairs"]
        uncertainty = error["r78_positive_finish_gap_error_s"]
        limits = independent_ingress_extrema(cases, uncertainty)
        quality = paired_quality(cases, len(pairs), x.r69["r69_max_200_Gbps_in_20_s"], uncertainty)
        trials.append({**state, "service": service, "heldout_error": error, "quality": quality,
                       "independent_ingress": limits,
                       "main_ABBA": {order: cases[f"{HOLDOUT}/{order}-cap200/{order}"] for order in ORDERS}})
    return trials


def guard_and_replay(workload, selected_start, x, models, results):
    selected = frozen_inputs(workload)
    alternatives = separate_decode_fits(results, [m[0] for m in models], x)
    latest = {name: max(latest_root_eos(selected, context, model[2]) for model in costs)
              for name, (context, costs, _) in alternatives.items()}
    start = max(selected_start, ceil((max(latest.values()) + ROOT_GUARD_S) * 10) / 10)
    shift = start - selected_start
    if shift:
        workload = {**workload, "request_sources": [{**r, "offered_s": r["offered_s"] + shift}
                                                    if r["predecessor"] else r for r in workload["request_sources"]]}
        workload["request_sources"].sort(key=lambda row: (row["offered_s"], row["id"]))
        selected = frozen_inputs(workload)
    primary = calibrated_replay(selected, workload["pairs"], x, models, start)
    diagnostics = {}
    for name, (context, costs, digest) in alternatives.items():
        variants = calibrated_replay(selected, workload["pairs"], context, costs, start)
        diagnostics[name] = {"trace_sha256": digest, "decode_service": context.decode, "latest_root_EOS_s": latest[name],
                             "models": [{**{k: v for k, v in m.items() if k not in (
                                 "case_pairs", "root_offers", "successor_offers")},
                                 "paired_profiles": len(m["case_pairs"])} for m in variants]}
    return workload, primary, {"preparation_shift_s": shift, "pooled_fit_screened_start_s": selected_start,
                               "final_successor_start_s": start, "archive_only_fits": diagnostics}


def verify_candidate(args, models, x, counts):
    """Verify the source identity and re-run the frozen paired protocol without searching."""
    report = read_json(args.output)["candidate"]
    plan = args.output.with_name("candidate-workload.json")
    digest = sha256(plan)
    if digest != report["workload_sha256"]:
        raise ValueError("The candidate plan changed")
    workload = read_json(plan)
    for field, path in (("ledger_sha256", args.ledger), ("evidence_receipt_sha256", args.eos_receipt),
                        ("source_budget_sha256", args.source_completions)):
        if sha256(path) != workload[field]:
            raise ValueError(f"Source receipt changed for {field}")
    requests = workload["request_sources"]
    ids = {r["id"] for r in requests}
    if len(ids) != 2 * report["windows"] or len(requests) != len(ids) or len({r["token_ids_uint32le_sha256"] for r in requests}) != len(ids):
        raise ValueError("The frozen requests must be unique original windows")
    for row in requests:
        key = row["id"]
        original = x.ledger[key]
        if (any(row[field] != original[field] for field in ("task", "step_id", "prompt_tokens", "token_ids_uint32le_sha256"))
                or original["truncated"] or row["max_tokens"] != max(8192, counts[key])
                or row["predecessor"] is not None and (row["predecessor"] not in ids or row["predecessor"] != original["predecessor"])
                or any(("stop", n) not in x.outcomes[key] for n in row["observed_EOS_reference_tokens"].values())):
            raise ValueError(f"Original identity, dependency, or EOS mismatch for {key}")
    formal = {}
    expected_order = ["b1-cap100", "b1-cap200", "b2-cap200", "b2-cap100"]
    for name in dict.fromkeys([R69_FORMAL, HOLDOUT, *(n for n, _ in TRAINING)]):
        manifest, trace, arms = native(args.results / name)
        if (manifest["baseline_commit"] != workload["baseline_commit"] or manifest["settings"] != workload["settings"]
                or manifest["case_order"] != expected_order or manifest["workload"]["source_revision"] != workload["source_revision"]
                or manifest["workload"]["route_policy"] != workload["route_policy"] or workload["ignore_eos"]):
            raise ValueError(f"Archive did not use the frozen native protocol in {name}")
        cases = {}
        for label, arm in arms.items():
            epochs = egress_epochs(arm)
            cases[label] = {**drained_measurement(arm[1]),
                            "D_TX_GB_by_port_in_P_span": [sum((b - a) * rates[i] / 8 for a, b, rates in epochs) for i in range(8)]}
        formal[name] = {"trace_sha256": trace, "cases": cases}
    observed = formal[R69_FORMAL]["cases"]
    for order in ORDERS:
        low, high = (observed[f"{order}-cap{cap}"]["throughput_tokens_s"] for cap in (100, 200))
        if not isclose(1 - low / high, x.r69["actual_r69_loss_by_order"][order], abs_tol=1e-10):
            raise ValueError("The formal r69 throughput comparator changed")
    results = []
    selected = frozen_inputs(workload)
    for (service, error, refs), original in zip(models, report["models"], strict=True):
        for retained in (0.0, 1.0):
            timing = clock_sensitivity(selected, workload["pairs"], x.sizes, x.fit, x.prefixes, x.ledger, service, x.decode,
                                       refs, x.profiles, x.cold_fraction, error["r78_positive_first_reply_error_s"],
                                       fixed_wave_start_s=original["wave_start_s"], keep_offers=True, earlier_root_retention=retained)
            if timing is None or timing["case_pairs"].keys() != original["case_pairs"].keys():
                raise ValueError("The frozen input must cover the same valid paired trials")
            cases = timing["case_pairs"]
            for label, trial in cases.items():
                before = original["case_pairs"][label]
                if trial["P_routes_by_cap"] != before["P_routes_by_cap"] or trial["same_P_routes"] != report["windows"]:
                    raise ValueError(f"Prefill route mismatch in {label}")
                for cap, arm in trial["arms"].items():
                    expected = before["arms"][cap]
                    if (arm["output_tokens"] != expected["output_tokens"]
                            or not isclose(arm["duration_s"], expected["duration_s"], abs_tol=1e-7)
                            or any(not isclose(a, b, abs_tol=1e-7) for a, b in zip(
                                arm["D_TX_GB_by_port"], expected["D_TX_GB_by_port"], strict=True))):
                        raise ValueError(f"Frozen replay changed timing or wire work in {label}/{cap}")
                if trial["arms"]["100"]["port_hot_s"] <= trial["arms"]["200"]["port_hot_s"]:
                    raise ValueError(f"Expected the low cap to take longer to serve the same GETs in {label}")
            extremes = independent_ingress_extrema(cases, error["r78_positive_finish_gap_error_s"])
            quality = paired_quality(cases, report["windows"], x.r69["r69_max_200_Gbps_in_20_s"],
                                     error["r78_positive_finish_gap_error_s"])
            if not quality["has_comparable_sustained_pressure"]:
                raise ValueError("Every replay must retain comparable bytes and post-start supply pressure")
            results.append({"service": service, "retained_earlier_attention_fraction": retained,
                            "trials": len(cases), "metrics": quality, "independent_ingress": extremes,
                            "minimum_continuous_D_gap_s": min(c["ongoing_D_supply_gap"]["longest_100_fewer_ready_s"]
                                                              for c in cases.values())})
    start = report["root_ready_guard"]["final_successor_start_s"]
    _, _, guard = guard_and_replay(workload, start, x, models, args.results)
    extra_fits = guard["archive_only_fits"]
    achieved = {order: min(row["independent_ingress"][order]["100_loss_after_r78_error"]
                           for group in [results, *(r["models"] for r in extra_fits.values())] for row in group) for order in ORDERS}
    for name, replay in extra_fits.items():
        recorded = report["root_ready_guard"]["archive_only_fits"][name]
        if (not all(m["quality"]["has_comparable_sustained_pressure"] for m in replay["models"])
                or any(not all(r["same_P_routes"] and r["same_per_port_GB"] for r in m["independent_ingress"].values())
                       for m in replay["models"])
                or replay != recorded):
            raise ValueError(f"The separate decode-fit verification did not match {name}")
    if guard["final_successor_start_s"] != start or any(not isclose(achieved[order], report[
            "minimum_independent_ingress_loss_by_order"][order], abs_tol=1e-10) for order in ORDERS):
        raise ValueError("Frozen preparation or the conservative throughput comparison changed")
    write_json(args.output.with_name("frozen-replay-verification.json"), {"workload_sha256": digest,
               "archive_observations": formal, "paired_models": results, "case_order": expected_order,
               "separate_decode_service_fits_verified": list(extra_fits),
               "minimum_independent_ingress_loss_by_order": achieved,
               "scope": "R69/r78 and reference archives are measured. The frozen candidate is a model projection with matched historical EOS."})


def run(args):
    balanced, calibration = seed(args.calibration)
    wider, wide_result = seed(args.wider)
    prior = read_json(args.prior / "summary.json")
    ledger, counts, outcomes, sizes = load_source(args.ledger, args.eos_receipt, args.source_completions,
                                                 balanced["source_revision"], args.results)
    fit = native(args.results / HOLDOUT)[0]["workload"]["arrival_order_screen"]["admission_fit"]
    x = ReplayInputs(ledger, sizes, outcomes, fit, ColdPrefixes(ledger), prior["decode_fit"], prior["r69_comparison"],
                     clock_profiles(args.results, fit), prior["guarded_clock_sensitivity"]["root_speed_fraction"])
    options = source_options(ledger, counts, outcomes, sizes, EOS_CUTOFF)
    models = [(m["service"], m["heldout_error"], [reference_state(observed_wave(args.results / name, case),
                x.prefixes, m["service"], x.decode) for name, case in TRAINING]) for m in calibration["models"]]
    if args.verify_only:
        verify_candidate(args, models, x, counts)
        return
    candidates, history, best, chosen = [balanced, wider], [], None, None
    for _ in range(args.rounds):
        choices, inspected = natural_range_neighbors(candidates, options, x, models[0])
        result, complete = best_matching_range(choices, models, x)
        inspected.update(completed_joint_replays=complete, best_margin=None if result is None else result[0])
        history.append(inspected)
        print(inspected, flush=True)
        if result is None or (best is not None and result[0] <= best[0]):
            break
        best = result
        _, fitted, trial, pairs, _ = best
        chosen = freeze_workload(balanced, pairs, fitted, trial, x, counts)
        candidates = [chosen]
    arms = (*TRAINING, (TRAINING[0][0], "b1-cap200"), (TRAINING[0][0], "b2-cap100"),
            (TRAINING[1][0], "b2-cap100"), *((HOLDOUT, f"{order}-cap{cap}") for order in ORDERS for cap in (100, 200)))
    report = {"r69": x.r69, "archive_earlier_attention": archive_coverage(args.results, x, arms),
              "r43_return_APC": archive_lookup(args.results / R43_METHODS, ledger, x.prefixes),
              "third_turn_bounds": triple_bounds(options, ledger), "search": history,
              "earlier_root_sensitivity": {"59_windows": retention_comparison(balanced, calibration, models, x),
                                           "68_windows": retention_comparison(wider, wide_result, models, x)},
              "earlier_release_probe": early_release_probe(wider, frozen_inputs(wider), models, x, ROOT_GUARD_S, 3.0)}
    if best is not None:
        value, verified, _, pairs, family = best
        workload = args.output.with_name("candidate-workload.json")
        chosen, verified, guard = guard_and_replay(chosen, verified[0]["wave_start_s"], x, models, args.results)
        write_json(workload, chosen)
        extrema = {order: min(m["independent_ingress"][order]["100_loss_after_r78_error"]
                              for group in [verified, *(r["models"] for r in guard["archive_only_fits"].values())]
                              for m in group) for order in ORDERS}
        report["candidate"] = {"windows": len(pairs), "last_task_range_replacement": family, "models": verified,
                               "screening_margin_before_final_guard": value, "root_ready_guard": guard,
                               "minimum_independent_ingress_loss_by_order": extrema,
                               "worst_same_order_margin_after_r78_error": min(extrema[order] - x.r69["actual_r69_loss_by_order"][order]
                                                                              for order in ORDERS),
                               "workload_sha256": sha256(workload)}
    write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    for key in ("prior", "calibration", "wider", "ledger", "eos-receipt", "source-completions", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--verify-only", action="store_true", help="Verify the already-frozen plan and rerun both cap orders")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
