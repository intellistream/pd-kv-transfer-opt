"""Audit the archived prefill response order and re-evaluate the frozen natural wave."""

from __future__ import annotations

import argparse
from collections import defaultdict
from gzip import open as gzip_open
from itertools import product
import json
from math import isclose
from pathlib import Path
from statistics import mean

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.simulator.decode_supply import simulate_decode
from pd_kv_transfer_opt.simulator.rdma_replay import egress_epochs
from pd_kv_transfer_opt.simulator.terminal_supply import (
    ColdPrefixes, HOLDOUT, METHODS, ORDERS, TRAINING, clock_sensitivity, ingress_residuals, load_source,
    loss_after_residual, native, observed_wave, reference_state, run_successors, search,
)
from pd_kv_transfer_opt.simulator.terminal_wave import P_ROLES, decode_deliveries, prefill_wave


def measured_first_loads(path):
    manifest = read_json(path / "manifest.json")
    children = {r["id"] for r in manifest["workload"]["request_sources"] if r["predecessor"]}
    trace = path / "native-method-events.jsonl.gz"
    digest = sha256(trace)
    if digest != manifest["artifacts"][trace.name]["sha256"]:
        raise ValueError("Method events do not match their archive manifest")
    event_by_role = defaultdict(list)
    widths = set()
    with gzip_open(trace, "rt", encoding="utf-8") as stream:
        for record in map(json.loads, stream):
            row = record["data"]
            if (row["type"] == "worker_load_begin" and row["instance"] in P_ROLES
                    and children.intersection(row["request_ids"])):
                event_by_role[record["case"], row["instance"]].append(row)
                widths.add(len(row["request_ids"]))
    result = {}
    for (case, role), events in sorted(event_by_role.items()):
        first = min(events, key=lambda row: row["monotonic_ns"])
        result.setdefault(case, {})[role] = first["request_ids"]
    if any(set(roles) != set(P_ROLES) or any(len(keys) != 1 for keys in roles.values()) for roles in result.values()):
        raise ValueError("The observed first successor load was not a singleton on both P instances")
    return {"method_trace_sha256": digest, "first_loads": result, "observed_widths": sorted(widths)}


def wire_times(epochs, start):
    """Interpolate the time each fraction of the sampled MAC volume has left D."""
    points = [(max(a, start), b, sum(rates) / 8) for a, b, rates in epochs if b > start and sum(rates) > 0]
    moved = sum((b - a) * rate for a, b, rate in points)
    if not moved:
        raise ValueError("Expected a D egress wave")
    times = []
    for q in (.10, .25, .50, .75, .90, .95, .99):
        left = moved * q
        for a, b, rate in points:
            volume = (b - a) * rate
            if left <= volume:
                times.append(a + left / rate)
                break
            left -= volume
    return times


def role_quantiles(wave, reply):
    return [percentile([t for k, t in reply.items() if wave["jobs"][k]["P"] == p], q)
            for p in P_ROLES for q in (.05, .25, .50, .75, .95, 1)]


def role_and_wire_fit(results, evidence):
    cases = (*TRAINING, (TRAINING[0][0], "b1-cap200"),
             (TRAINING[0][0], "b2-cap100"), (TRAINING[1][0], "b2-cap100"))
    observed = []
    for name, case in cases:
        path = results / name
        wave = observed_wave(path, case)
        arm = native(path)[2][case]
        observed.append((wave, int(case.rsplit("cap", 1)[1]), role_quantiles(wave, wave["P_reply_s"]),
                         wire_times(egress_epochs(arm), wave["wave_start_s"])))

    candidates = []
    widths = [w for w in evidence["observed_widths"] if w >= 2]
    for width, launch, nonwire, efficiency, warm_rate in product(
            widths, (.05, .15, .30, .45, .60), (.03, .12, .22, .32, .42),
            (.82, .88, .91, .94, .97, 1.0), (12_400, 18_500, 24_800)):
        costs = {"first_GET_batch_width": 1, "GET_batch_width": width, "first_GET_launch_s": launch,
                 "batch_nonwire_s": nonwire, "rail_efficiency": efficiency, "warm_tokens_s": warm_rate}
        errors = []
        for wave, cap, p_times, actual_wire in observed:
            reply, epochs, _, _ = prefill_wave(wave["jobs"], cap, wave["shares"], costs)
            role_error = mean(abs(a - b) for a, b in zip(role_quantiles(wave, reply), p_times, strict=True))
            wire_error = mean(abs(a - b) for a, b in zip(wire_times(epochs, wave["wave_start_s"]), actual_wire, strict=True))
            errors.append((role_error, wire_error))
        role, wire = (mean(x) for x in zip(*errors, strict=True))
        candidates.append((role + wire, costs, {"P_role_quantile_MAE_s": role, "MAC_quantile_MAE_s": wire}))

    best = min(candidates, key=lambda row: row[0])
    alternate = min((row for row in candidates if row[1]["GET_batch_width"] != best[1]["GET_batch_width"]), key=lambda row: row[0])
    return [{"service": costs, "training_error_s": errors} for _, costs, errors in (best, alternate)], cases


def r78_replay(results, service, decode):
    path = results / HOLDOUT
    low, high = (observed_wave(path, f"b1-cap{cap}") for cap in (100, 200))
    arms = {100: low, 200: high}
    jobs = high["jobs"]
    actual_phases = {k: high["rows"][k]["first_s"] - high["P_reply_s"][k] for k in jobs}
    raw = native(path)[2]
    modeled = {cap: prefill_wave(jobs, cap, high["shares"], service) for cap in arms}
    frozen_D = {key: high["rows"][key]["D"] for key in jobs}
    wave_by_cap = {}
    for cap, wave in arms.items():
        reply, epochs, _, _ = modeled[cap]
        actual_wire = wire_times(egress_epochs(raw[f"b1-cap{cap}"]), wave["wave_start_s"])
        modeled_wire = wire_times(epochs, wave["wave_start_s"])
        fitted = run_successors(jobs, cap, high["shares"], high["wave_start_s"],
                                {**high, "common_D_tail_residual_s": 0}, service, decode, measured_placements=frozen_D)["report"]
        wave_by_cap[str(cap)] = {"observed_full_output_tokens": sum(r["output_tokens"] for r in wave["rows"].values()),
                                 "observed_finish_s": max(r["end_s"] for r in wave["rows"].values()),
                                 "observed_wave_first_D_token_s": min(wave["rows"][k]["first_s"] for k in jobs),
                                 "predicted_wave_first_D_token_s": fitted["first_D_token_s"],
                                 "observed_last_P_reply_s": max(wave["P_reply_s"].values()), "predicted_last_P_reply_s": max(reply.values()),
                                 "observed_95pct_wire_time_s": actual_wire[-2], "predicted_95pct_wire_time_s": modeled_wire[-2]}

    outcomes = {}
    for order_source, time_source, handoff in product(("observed", "modeled"), ("observed", "modeled"), ("per_request", "median")):
        state = {}
        for cap, wave in arms.items():
            clocks = {"observed": wave["P_reply_s"], "modeled": modeled[cap][0]}
            replies = {}
            for role in P_ROLES:
                keys = [k for k, row in jobs.items() if row["P"] == role]
                ordered = sorted(keys, key=lambda k: (clocks[order_source][k], k))
                times = sorted(clocks[time_source][k] for k in keys)
                replies.update(zip(ordered, times, strict=True))
            requests = {k: {"offered_s": t, "predecessor": None, "ready_s": t, "first_s": t + (
                actual_phases[k] if handoff == "per_request" else high["successor_handoff_s"]), "end_s": high["rows"][k]["end_s"],
                             "D": frozen_D[k], "output_tokens": jobs[k]["output_tokens"]} for k, t in replies.items()}
            sim = simulate_decode(requests, decode)
            full_output = sum(r["output_tokens"] for r in high["rows"].values())
            state[str(cap)] = {"finish_s": sim["finish_s"], "tail_id": sim["tail_id"], "output_tokens": full_output,
                               "throughput_tokens_s": full_output / sim["finish_s"]}
        outcomes[f"{order_source}/{time_source}/{handoff}"] = {
            "arms": state, "gap_s": state["100"]["finish_s"] - state["200"]["finish_s"]}
    return {"trace_sha256": high["trace_sha256"], "wave_by_cap": wave_by_cap, "paired_gap_attribution": outcomes}


def clock_profiles(results, fit):
    profiles = {}
    for name in dict.fromkeys([HOLDOUT, *(path for path, _ in TRAINING)]):
        m, _, arms = native(results / name)
        residuals = ingress_residuals(arms, m["workload"]["arrival_order_screen"]["wire_body_bytes"], fit)
        profiles.update({f"{name}/{case}": offsets for case, offsets in residuals.items()})
    return profiles


def frozen_candidate(results, prior, workload, ledger, sizes, service, decode):
    m, _, _ = native(results / HOLDOUT)
    fit = m["workload"]["arrival_order_screen"]["admission_fit"]
    profiles = clock_profiles(results, fit)
    prefixes = ColdPrefixes(ledger)
    refs = [reference_state(observed_wave(results / name, case), prefixes, service, decode) for name, case in TRAINING]
    rows = workload["request_sources"]
    selected = {"root_offers": {r["id"]: r["offered_s"] for r in rows if not r["predecessor"]},
                "successor_offers": {r["id"]: r["offered_s"] for r in rows if r["predecessor"]},
                "eos_samples": [{r["id"]: r["observed_EOS_reference_tokens"][order] for r in rows} for order in ORDERS]}
    guard = prior["guarded_clock_sensitivity"]
    cases = {}
    for name, slow, early in (("nominal", 1.0, 0.0), ("previous_clock_stress", guard["root_speed_fraction"], guard["early_credit_stress_s"])):
        trial = clock_sensitivity(selected, workload["pairs"], sizes, fit, prefixes, ledger, service, decode, refs,
                                  profiles, slow, early, fixed_wave_start_s=guard["wave_start_s"], keep_offers=True)
        if trial is None:
            cases[name] = {"modeled_reference_infeasible": True}
            continue
        if any(trial[k] != selected[k] for k in ("root_offers", "successor_offers")):
            raise ValueError("The protected replay changed the frozen arrival plan")
        observed = trial["case_pairs"]
        cases[name] = {"minimum_low_cap_loss": min(row["100_low_loss_fraction"] for row in observed.values()),
                       "minimum_same_P_routes": min(row["same_P_routes"] for row in observed.values()),
                       "maximum_wire_fraction_change": max(abs(row["100_to_200_total_wire_fraction_change"]) for row in observed.values()),
                       "pairs_with_ongoing_supply_gap": sum(row["ongoing_D_supply_gap"]["100_fewer_ready_s"] > 0 for row in observed.values()),
                       "main_ABBA": {order: observed[f"{HOLDOUT}/{order}-cap200/{order}"] for order in ORDERS}}
    return cases


def response_error(results, model, decode):
    attr = model["r78_same_P_holdout"]["paired_gap_attribution"]
    optimism = max(0.0, attr["modeled/modeled/median"]["gap_s"] - attr["observed/observed/per_request"]["gap_s"])
    lead_s = 0.0
    for order in ORDERS:
        wave = observed_wave(results / HOLDOUT, f"{order}-cap200")
        fitted = run_successors(wave["jobs"], 200, wave["shares"], wave["wave_start_s"],
                                {**wave, "common_D_tail_residual_s": 0}, model["service"], decode)
        lead_s = max(lead_s, fitted["report"]["first_D_token_s"] - min(wave["rows"][k]["first_s"] for k in wave["jobs"]))
    return {"r78_positive_finish_gap_error_s": optimism, "r78_positive_first_reply_error_s": lead_s}


def paired_quality(case_pairs, n, threshold_rate, optimism):
    worst = min(row["100_low_loss_fraction"] for row in case_pairs.values())
    reduced = min(loss_after_residual(row, optimism) for row in case_pairs.values())
    peak = min(row["arms"]["200"]["peak_20_s_D_TX_Gbps"] for row in case_pairs.values())
    same_routes = min(row["same_P_routes"] for row in case_pairs.values())
    same_wire = all(all(isclose(a, b, rel_tol=1e-9, abs_tol=1e-8) for a, b in zip(
        row["arms"]["100"]["D_TX_GB_by_port"], row["arms"]["200"]["D_TX_GB_by_port"], strict=True)) for row in case_pairs.values())
    gap_s = min(row["ongoing_D_supply_gap"]["100_fewer_ready_s"] for row in case_pairs.values())
    return {"has_comparable_sustained_pressure": same_routes == n and same_wire and peak > threshold_rate and gap_s > 0,
            "min_same_P_routes": same_routes, "same_per_port_bytes": same_wire, "min_200_peak_20s_Gbps": peak,
            "min_ongoing_D_gap_s": gap_s, "worst_raw_loss": worst, "worst_loss_after_r78_error": reduced}


def archive_ingress_check(results, decode, services):
    """Check the calibrated profiles on actual per-arm ingress and egress placement."""
    cases = {}
    for name, order in product((HOLDOUT, "2026-11-22-r68-terminal-source-restore-short-eos"), ORDERS):
        waves = {cap: observed_wave(results / name, f"{order}-cap{cap}") for cap in (100, 200)}
        high = waves[200]
        outputs = {k: row["output_tokens"] for k, row in high["jobs"].items()}
        placements = {k: high["rows"][k]["D"] for k in outputs}
        finishes = []
        for costs in services:
            arms = {}
            for cap, wave in waves.items():
                jobs = {k: {**row, "output_tokens": outputs[k]} for k, row in wave["jobs"].items()}
                shared_ingress = {k: {**row, "P_admit_s": high["jobs"][k]["P_admit_s"]} for k, row in jobs.items()}
                replies = {"observed": wave["P_reply_s"],
                           "modeled": prefill_wave(jobs, cap, wave["shares"], costs)[0],
                           "with_paired_200_ingress": prefill_wave(shared_ingress, cap, wave["shares"], costs)[0]}
                arms[str(cap)] = {label: decode_deliveries(jobs, times, decode, 16, high["successor_handoff_s"],
                                                         placements=placements)["finish_s"] for label, times in replies.items()}
            gaps = {label: arms["100"][label] - arms["200"][label] for label in arms["100"]}
            finishes.append({"service": costs, "finish_s_by_arm": arms, "finish_gap_s": gaps})
        cases[f"{name}/{order}"] = {"shared_P_routes": sum(waves[100]["jobs"][k]["P"] == high["jobs"][k]["P"] for k in outputs),
                                    "successors": len(outputs), "full_output_tokens": sum(r["output_tokens"] for r in high["rows"].values()),
                                    "D_TX_GB_by_arm": {str(cap): w["wire_GB"] for cap, w in waves.items()},
                                    "archive_sha256": high["trace_sha256"], "models": finishes}
    return cases


def screen_revised(results, prior, workload, ledger, counts, outcomes, sizes, fits, *, wider=False):
    decode, target = (prior[k] for k in ("decode_fit", "r69_comparison"))
    m, _, _ = native(results / HOLDOUT)
    fit = m["workload"]["arrival_order_screen"]["admission_fit"]
    profiles = clock_profiles(results, fit)
    prefixes = ColdPrefixes(ledger)
    references = [[reference_state(observed_wave(results / name, case), prefixes, model["service"], decode)
                   for name, case in TRAINING] for model in fits]
    errors = [response_error(results, model, decode) for model in fits]
    cold = prior["guarded_clock_sensitivity"]["root_speed_fraction"]
    choices, summary = search(outcomes, counts, ledger, sizes, fit, prefixes, fits[0]["service"], decode, references[0], target,
                              wider=wider)
    print(f"Fit and initial search finished with {summary['simulated_native_layouts']} candidate layouts", flush=True)
    floor = max(target["actual_r69_loss_by_order"].values())
    threshold_rate = target["r69_max_200_Gbps_in_20_s"]
    best, evaluated, fully_comparable = None, 0, 0
    for nominal, selected, pairs, _, _, _, cutoff in sorted(choices, key=lambda row: row[0], reverse=True):
        if not nominal[0] or not selected["beats_both_r69_actual_pairs_in_model"]:
            continue
        primary, secondary = [], []
        for i, (model, refs, error) in enumerate(zip(fits, references, errors, strict=True)):
            earlier = error["r78_positive_first_reply_error_s"]
            stress = clock_sensitivity(selected, pairs, sizes, fit, prefixes, ledger, model["service"], decode, refs, profiles, cold,
                                       earlier, fixed_wave_start_s=primary[0]["wave_start_s"] if primary else None)
            if stress is None:
                break
            quality = paired_quality(stress["case_pairs"], len(pairs), threshold_rate, error["r78_positive_finish_gap_error_s"])
            primary.append(stress)
            secondary.append(quality)
            if i == 0 and best and best[0][0] and (
                    not quality["has_comparable_sustained_pressure"] or quality["worst_loss_after_r78_error"] < best[0][1]):
                break  # The weaker of two fits can only lower this score.
        if len(secondary) != len(fits):
            continue
        evaluated += 1
        comparable = all(row["has_comparable_sustained_pressure"] for row in secondary)
        fully_comparable += comparable
        score = (comparable, min(row["worst_loss_after_r78_error"] for row in secondary), min(row["worst_raw_loss"] for row in secondary))
        if best is None or score > best[0]:
            best = score, pairs, selected, cutoff, primary, secondary
    if best is None:
        return {"initial_search": summary, "completed_joint_replays": evaluated, "comparable_joint_replays": fully_comparable}, None
    score, pairs, selected, cutoff, states, quality = best
    replay_rows = []
    for state, metrics, model, error in zip(states, quality, fits, errors, strict=True):
        if state["root_offers"] != states[0]["root_offers"] or state["successor_offers"] != states[0]["successor_offers"]:
            raise ValueError("Both fitted models must replay the same arrival plan")
        replay_rows.append({"service": model["service"], "quality": metrics, "heldout_error": error,
                            "case_pairs": state["case_pairs"], "main_ABBA": {
                                order: state["case_pairs"][f"{HOLDOUT}/{order}-cap200/{order}"] for order in ORDERS}})
    root_of = {p["successor"]: p["root"] for p in pairs}
    offers = {**states[0]["root_offers"], **states[0]["successor_offers"]}
    rows = [{**{k: ledger[key][k] for k in ("id", "task", "step_id", "prompt_tokens", "token_ids_uint32le_sha256")},
             "predecessor": root_of.get(key), "offered_s": t, "tool_gap_s": 0, "max_tokens": max(8192, counts[key]),
             "observed_EOS_reference_tokens": {order: eos[key] for order, eos in zip(ORDERS, selected["eos_samples"], strict=True)}}
            for key, t in sorted(offers.items(), key=lambda row: (row[1], row[0]))]
    new_work = {**workload, "pairs": pairs, "request_sources": rows, "maximum_observed_natural_EOS_tokens_at_selection": cutoff,
                "scope": "Independent original adjacent rounds with one shared protected arrival plan for both models and bandwidths."}
    new_work.pop("projected_native_P_roles_without_early_credit", None)
    report = {"initial_search": summary, "completed_joint_replays": evaluated, "comparable_joint_replays": fully_comparable,
              "successor_wave_start_s": states[0]["wave_start_s"], "windows": len(pairs), "r69_actual_max_loss": floor,
              "minimum_raw_loss_across_fits": score[2], "minimum_loss_after_r78_error_across_fits": score[1],
              "comparable_on_both_fits": score[0], "exceeds_r69_after_r78_error": score[0] and score[1] > floor,
              "models": replay_rows}
    return report, new_work


def independent_ingress_extrema(case_pairs, finish_error_s):
    """Pair the fastest low and slowest high cap under independent archive ingress profiles."""
    extremes = {}
    for order in ORDERS:
        trials = [(label, case) for label, case in case_pairs.items() if label.endswith(f"/{order}")]
        label_low, fastest_low = min(trials, key=lambda row: row[1]["arms"]["100"]["duration_s"])
        label_high, slowest_high = max(trials, key=lambda row: row[1]["arms"]["200"]["duration_s"])
        low, high = (case["arms"][cap] for case, cap in ((fastest_low, "100"), (slowest_high, "200")))
        if low["output_tokens"] != high["output_tokens"]:
            raise ValueError("A reference order must keep its output work fixed across ingress timings")
        extremes[order] = {
            "fastest_100_case": label_low, "slowest_200_case": label_high,
            "fastest_100": {k: low[k] for k in ("output_tokens", "duration_s", "throughput_tokens_s")},
            "slowest_200": {k: high[k] for k in ("output_tokens", "duration_s", "throughput_tokens_s")},
            "same_P_routes": fastest_low["P_routes_by_cap"]["100"] == slowest_high["P_routes_by_cap"]["200"],
            "same_per_port_GB": all(isclose(a, b, rel_tol=1e-9, abs_tol=1e-8) for a, b in zip(
                low["D_TX_GB_by_port"], high["D_TX_GB_by_port"], strict=True)),
            "100_low_loss_fraction": 1 - high["duration_s"] / low["duration_s"],
            "100_loss_after_r78_error": loss_after_residual({"arms": {"100": low, "200": high}}, finish_error_s),
        }
    return extremes


def audit_rescreened(args, prior, ledger, sizes):
    audit = read_json(args.output)
    rescreen = audit["joint_rescreen"]
    path = args.output.with_name("rescreened-workload.json")
    digest = sha256(path)
    if digest != rescreen["workload_sha256"]:
        raise ValueError("The rescreened request manifest has changed")
    workload = read_json(path)
    rows = workload["request_sources"]
    chosen = {"root_offers": {r["id"]: r["offered_s"] for r in rows if not r["predecessor"]},
              "successor_offers": {r["id"]: r["offered_s"] for r in rows if r["predecessor"]},
              "eos_samples": [{r["id"]: r["observed_EOS_reference_tokens"][order] for r in rows} for order in ORDERS]}
    m, _, _ = native(args.results / HOLDOUT)
    fit = m["workload"]["arrival_order_screen"]["admission_fit"]
    profiles = clock_profiles(args.results, fit)
    prefixes, decode = ColdPrefixes(ledger), prior["decode_fit"]
    results = []
    for model in rescreen["models"]:
        service = model["service"]
        references = [reference_state(observed_wave(args.results / name, c), prefixes, service, decode) for name, c in TRAINING]
        state = clock_sensitivity(chosen, workload["pairs"], sizes, fit, prefixes, ledger, service, decode, references, profiles,
                                  prior["guarded_clock_sensitivity"]["root_speed_fraction"], model["heldout_error"]["r78_positive_first_reply_error_s"],
                                  fixed_wave_start_s=rescreen["successor_wave_start_s"], keep_offers=True, compare_fixed_D=True)
        if state is None or state["case_pairs"].keys() != model["case_pairs"].keys():
            raise ValueError("The diagnostic needs the complete previous ABBA replay")
        group = {}
        for label, case in state["case_pairs"].items():
            prior_case = model["case_pairs"][label]
            if (case["P_routes_by_cap"] != prior_case["P_routes_by_cap"] or not isclose(
                    case["100_low_loss_fraction"], prior_case["100_low_loss_fraction"], abs_tol=1e-10)):
                raise ValueError(f"The fixed-D diagnostic changed the original replay in {label}")
            group[label] = case["fixed_to_200_D_placements"]
        independent_ingress = independent_ingress_extrema(model["case_pairs"], model["heldout_error"]["r78_positive_finish_gap_error_s"])
        results.append({"service": service, "minimum_fixed_D_100_loss": min(r["100_low_loss_fraction"] for r in group.values()),
                        "independent_ingress_extrema": independent_ingress,
                        "case_pairs": group, "main_ABBA": {order: group[f"{HOLDOUT}/{order}-cap200/{order}"] for order in ORDERS}})
    archive_checks = archive_ingress_check(args.results, decode, [model["service"] for model in audit["prior_and_revised_models"]])
    write_json(args.output.with_name("d-placement-attribution.json"), {"workload_sha256": digest, "models": results,
              "observed_ingress_and_wire_check": {"cases": archive_checks,
                  "scope": "Use each arm's observed P routes and D wire volume with the 200 arm's EOS, D placement and median D handoff. "
                           "Swap observed ingress timestamps only. Cases with changed P routes are conditional checks, not equal-wire A/B."},
              "scope": "Keep the native P completions from the frozen replay. Replace only the low-cap successor D placements with "
                       "the paired high-cap placements. This is an attribution control, not the native-route primary result."})


def run(args):
    prior = read_json(args.prior / "summary.json")
    path = args.prior / "candidate-workload.json"
    workload = read_json(path)
    digest = sha256(path)
    if digest != prior["candidate_workload_sha256"]:
        raise ValueError("The previous result does not describe this candidate")
    ledger, counts, outcomes, sizes = load_source(args.ledger, args.eos_receipt, args.source_completions, workload["source_revision"], args.results)
    if args.audit_rescreen:
        audit_rescreened(args, prior, ledger, sizes)
        return
    evidence = measured_first_loads(args.results / METHODS)
    fits, training = role_and_wire_fit(args.results, evidence)
    alternatives = [{"service": prior["GET_model"]}, *fits]
    for model in alternatives:
        service = model["service"]
        model["r78_same_P_holdout"] = r78_replay(args.results, service, prior["decode_fit"])
        model["frozen_candidate"] = frozen_candidate(args.results, prior, workload, ledger, sizes, service, prior["decode_fit"])
    result = {"candidate_workload_sha256": digest, "first_load_method_evidence": evidence,
              "trace_training_cases": training, "prior_and_revised_models": alternatives,
              "scope": "Models are fit before inspecting r78. The r78 stage swap fixes D placement and natural EOS from "
                       "the 200 arm and exchanges P times and order independently within each measured P. "
                       "The frozen workload projections replay its exact original offers with modeled native routing."}
    if args.rescreen or args.wider_screen:
        report, new_work = screen_revised(args.results, prior, workload, ledger, counts, outcomes, sizes, fits, wider=args.wider_screen)
        if new_work is not None:
            path = args.output.with_name("rescreened-workload.json")
            write_json(path, new_work)
            report["workload_sha256"] = sha256(path)
        result["joint_rescreen"] = report
    write_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--eos-receipt", type=Path, required=True)
    parser.add_argument("--source-completions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    followup = parser.add_mutually_exclusive_group()
    followup.add_argument("--rescreen", action="store_true")
    followup.add_argument("--wider-screen", action="store_true", help="Screen 65–88 original windows against the nominal 128 GiB Store budget")
    followup.add_argument("--audit-rescreen", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
