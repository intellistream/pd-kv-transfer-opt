"""Replay r43 native 200 Gbps completion with width-specific Decode stage savings."""

import argparse
from collections import Counter, defaultdict
from itertools import pairwise
from math import isclose
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.compare_session_runs import DECODE, timeline
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, check_observed, gauge, phase_times
from pd_kv_transfer_opt.profiling.replay_decode_tail import calibrate_native_period
from pd_kv_transfer_opt.profiling.screen_terminal_throughput import ARCHIVES, sampled_decode_windows, width_band
from pd_kv_transfer_opt.simulator.decode_supply import DECODE_ROLES, decode_samples, fit_decode, native_arms, simulate_decode


STAGES = ("compute", "collective", "host_runtime_ceiling")
SCOPES = {"1": (1,), "2–4": tuple(range(2, 5)), "5–8": tuple(range(5, 9)),
          "9–16": tuple(range(9, 17)), "1–4": tuple(range(1, 5)), "1–16": tuple(range(1, 17))}
TARGETS_PCT = (5, 10)
CAPTURES = ("r34", "r37")


def verified_json(directory, filename, manifest):
    path = directory / filename
    digest = sha256(path)
    if digest != manifest["files"][filename]["sha256"]:
        raise ValueError(f"Digest mismatch in {path}")
    return read_json(path), digest


def historical_priors(results):
    root = results / "2026-11-20-r37-decode-critical-step"
    manifest = read_json(root / "manifest.json")
    sensitivity, digest = verified_json(root, "sensitivity.json", manifest)
    period = sensitivity["tail"]["D0"]["worker_period_ms"]["p50"]
    priors = {}
    for name in CAPTURES:
        breakdown, trace_hash = verified_json(root, f"{name}-breakdown.json", manifest)
        if len(breakdown["sources"]) != 4:
            raise ValueError(f"Expected all four ranks in {name}")
        fit = calibrate_native_period(period, breakdown)["calibrated_parts_ms"]
        parts = dict(zip(STAGES, (fit["compute_ms"], fit["collective_after_last_ms"],
                                  fit["uncovered_before_first_two_joins_residual_ms"]), strict=True))
        parts["other"] = period - sum(parts.values())
        coverage = breakdown["coverage"]
        priors[name] = {
            "breakdown_sha256": trace_hash, "aligned_steps": breakdown["aligned_steps"],
            "anchored_native_stage_ms": parts, "shares": {key: value / period for key, value in parts.items()},
            "captured_period_ms_p50": coverage["period_ms"]["p50"],
            "old_capture_20pct_saving_ms_p50": {
                stage: coverage[f"save_{key}_0.2_ms"]["p50"] for stage, key in
                zip(STAGES, ("compute", "collective_after_last", "submission_gap"), strict=True)
            },
        }
    r34 = results / "2026-11-20-r34-native-decode-path"
    manifest = read_json(r34 / "manifest.json")
    summary, summary_hash = verified_json(r34, "summary.json", manifest)
    if summary["trace_sha256"] != sensitivity["r34_trace_sha256"]:
        raise ValueError("Native step cadence and width archive disagree")
    widths = {}
    for width in (1, 2, 4, 8, 12, 16):
        data = summary["step_cost_by_requests_and_history"][f"{width}/64K+"]
        costs = data["stage_cost_ms"]
        widths[str(width)] = {"steps": data["steps"], "period_ms": costs["rank0_worker_gap_ms"]["p50"],
                              "slowest_device_forward_ms": costs["slowest_rank_forward_device_ms"]["p50"]}
    return {"native_period_ms": period, "sensitivity_sha256": digest, "r34_summary_sha256": summary_hash,
            "r34_baseline": {key: summary["baseline"][key] for key in
                             ("output_tokens", "duration_s", "throughput_tokens_s")},
            "r34_width_context_64K_plus": widths, "captures": priors}


def step_parts(period_ms, singleton_ms, shares, spread):
    """Partition a step without adding early-rank waiting to collective service."""
    anchor = period_ms if spread == "constant_shares" else min(period_ms, singleton_ms)
    parts = {key: share * anchor for key, share in shares.items()}
    parts["compute"] += period_ms - anchor
    return parts


def confirmed_exposure(arm, shares, spread):
    """Use an interval only if the original active set and width are supported."""
    costs = defaultdict(lambda: defaultdict(Counter))
    for width, active, seconds, tokens in arm["windows"]:
        steps = tokens / width
        parts = step_parts(1000 * seconds / steps, arm["periods_ms"][1], shares, spread)
        for key in active:
            costs[width][key].update({stage: steps * parts[stage] / 1000 for stage in STAGES})
    return costs


def stream_intervals(requests, spans):
    """Use visible streams as a width proxy between engine counter samples."""
    intervals = defaultdict(list)
    for role in DECODE_ROLES:
        cohort = {key: spans[key] for key, row in requests.items() if row["D"] == role}
        boundaries = sorted({t for row in cohort.values() for t in (row["first_s"], row["end_s"])})
        for begin, end in pairwise(boundaries):
            active = {key for key, row in cohort.items() if row["first_s"] <= begin < row["end_s"]}
            if active:
                intervals[role].append((begin, end, active))
    return intervals


def tail_stream_widths(intervals, path):
    widths = Counter()
    for rows in intervals.values():
        for begin, end, active in rows:
            if path & active:
                widths[width_band(len(active))] += end - begin
    return dict(widths)


def width_counter_check(telemetry, intervals, path, origin):
    counts = Counter()
    for sample in telemetry:
        for engine in sample.get("engines", []):
            role = engine["role"]
            if role not in intervals:
                continue
            at = engine["monotonic_ns"] / NS - origin
            running = gauge(engine.get("values", {}), "num_requests_running")
            if running is None:
                continue
            active = next((active for begin, end, active in intervals[role] if begin <= at < end), set())
            if not path & active:
                continue
            width = len(active)
            same_band = running > 0 and width_band(width) == width_band(running)
            counts.update(samples=1, exact=width == running, same_band=same_band,
                          width_9_to_16_samples=width >= 9, width_9_to_16_same_band=width >= 9 and same_band)
    return dict(counts)


def stream_exposure(arm, by_width):
    costs = defaultdict(lambda: defaultdict(Counter))
    for rows in arm["stream_intervals"].values():
        for begin, end, active in rows:
            width = len(active)
            if width not in by_width:
                continue
            shares = {stage: by_width[width][stage] / arm["periods_ms"][width] for stage in STAGES}
            for key in active:
                costs[width][key].update({stage: (end - begin) * share for stage, share in shares.items()})
    return costs


def paired_sweep(evaluate, baselines):
    """Check every integer dose to allow the critical session or batching to change."""
    thresholds = dict.fromkeys(TARGETS_PCT)
    last_fail = dict.fromkeys(TARGETS_PCT, 0)
    at_twenty, ceiling = None, None
    best = 0.0
    for dose in range(1, 101):
        arms = {}
        for name, baseline in baselines.items():
            result = evaluate(name, dose / 100)
            arms[name] = {**result, "gain_pct": 100 * (baseline["duration_s"] / result["duration_s"] - 1)}
        common = min(a["gain_pct"] for a in arms.values())
        best = max(best, common)
        point = {"component_cut_pct": dose, "both_arms_gain_pct": common, "arms": arms}
        for target in TARGETS_PCT:
            if common + 1e-9 < target:
                thresholds[target] = None
                last_fail[target] = dose
            elif thresholds[target] is None:
                thresholds[target] = point
        if dose == 20:
            at_twenty = point
        if dose == 100:
            ceiling = point
    return {"target_common_gain_pct": thresholds, "last_dose_below_target_pct": last_fail,
            "cut_20pct": at_twenty, "cut_100pct": ceiling, "best_grid_common_gain_pct": best}


def batches(results):
    archive = results / ARCHIVES["r43"][0]
    manifest = read_json(archive / "manifest.json")
    workload = manifest["workload"]
    engines = manifest["settings"]["engines"]
    slots = engines["D"]["max_num_seqs"]
    if (workload["ignore_eos"] or workload["route_policy"] != "upstream native without route override"
            or slots != 16 or not all(engine["enable_prefix_caching"] for engine in engines.values())
            or not engines["D"]["recompute_scheduler_enable"]):
        raise ValueError("Expected the native r43 deployment with both APCs and natural EOS")
    digest, raw = native_arms(archive)
    captured = {name: raw[name] for name in ("b1-cap200", "b2-cap200")}
    arms, evidence = {}, {}
    for name, (data, requests, origin) in captured.items():
        shaping, = data["shaping.json"]
        if (shaping["aggregate_Gbps"] != 200 or len(shaping["readback"]) != 8
                or any(port["configured_Mbps"] != 25_000 or port["current_Mbps"] != 25_000
                       for port in shaping["readback"].values())
                or any(event.get("fixed_routes") for event in data["proxy"])):
            raise ValueError(f"Invalid 200 Gbps readback or forced routing in {name}")
        source, = (key for key in captured if key != name)
        fit = fit_decode(list(decode_samples(*captured[source])))
        costs = phase_times(requests)
        observed = timeline(requests, costs)
        chain = check_observed(requests)
        path = set(chain["tail_chain"])
        windows = tuple(sampled_decode_windows(data["telemetry"], requests, origin))
        spans, counts = Counter(), Counter()
        for width, active, seconds, tokens in windows:
            if path.intersection(active):
                spans[width_band(width)] += seconds
                counts[width_band(width)] += tokens / width
        periods = {n: 1000 * (fit["first_sequence_period_s"] + (n - 1) * fit["additional_sequence_period_s"])
                   for n in range(1, slots + 1)}
        dynamic = simulate_decode(requests, fit, slots=slots, include_requests=True)
        stream = stream_intervals(requests, requests)
        simulated_stream = stream_intervals(requests, dynamic["per_request"])
        over_slots_s = sum(end - begin for rows in stream.values() for begin, end, active in rows if len(active) > slots)
        baseline = {"duration_s": dynamic["finish_s"] - min(row["offered_s"] for row in requests.values()),
                    "output_tokens": dynamic["output_tokens"], "throughput_tokens_s": dynamic["drained_throughput_tokens_s"],
                    "terminal_request": dynamic["tail_id"]}
        if observed["terminal_request"] != baseline["terminal_request"]:
            raise ValueError(f"The unmodified width model chose a different tail in {name}")
        arms[name] = {"requests": requests, "costs": costs, "windows": windows, "periods_ms": periods, "service": fit,
                      "stream_intervals": stream, "observed": observed, "dynamic": baseline}
        evidence[name] = {"observed": observed, "dynamic": baseline, "fit_from": source, "fit": fit,
                          "dynamic_baseline_throughput_error_pct": 100 * (baseline["throughput_tokens_s"] /
                                                                         observed["throughput_tokens_s"] - 1),
                          "tail_chain_eligible_seconds": dict(spans), "tail_chain_eligible_steps": dict(counts),
                          "tail_chain_stream_width_seconds": tail_stream_widths(stream, path),
                          "original_tail_chain_model_width_seconds": tail_stream_widths(simulated_stream, path),
                          "tail_chain_stream_vs_engine": width_counter_check(data["telemetry"], stream, path, origin),
                          "stream_proxy_over_slots_both_D_seconds": over_slots_s,
                          "observed_required_seconds_saved": {
                              str(g): observed["duration_s"] * g / (100 + g) for g in TARGETS_PCT}}
    return {"trace_sha256": digest, "workload_sha256": workload["workload_sha256"], "arms": evidence}, arms


def replay_budgets(arms, shares, spread):
    by_width = {name: {n: step_parts(period, arm["periods_ms"][1], shares, spread)
                       for n, period in arm["periods_ms"].items()} for name, arm in arms.items()}
    exposures = {
        "confirmed_windows": {name: confirmed_exposure(arm, shares, spread) for name, arm in arms.items()},
        "observed_stream_width": {name: stream_exposure(arm, by_width[name]) for name, arm in arms.items()},
    }
    observed = {name: arm["observed"] for name, arm in arms.items()}
    dynamic = {name: arm["dynamic"] for name, arm in arms.items()}
    cases = {name: {} for name in (*exposures, "rebatch_all_steps")}
    for scope, widths in SCOPES.items():
        for model in cases.values():
            model[scope] = {}
        for stage in STAGES:
            for model, exposure in exposures.items():
                eligible = {name: {key: sum(exposure[name][w].get(key, {}).get(stage, 0.) for w in widths)
                                   for key in arm["requests"]} for name, arm in arms.items()}

                def edit_observed(name, fraction):
                    arm = arms[name]
                    phases = {key: {**cost, DECODE: cost[DECODE] - fraction * eligible[name][key]}
                              for key, cost in arm["costs"].items()}
                    if any(row[DECODE] < 0 for row in phases.values()):
                        raise ValueError("Eligible width intervals overlap or exceed request residence")
                    return timeline(arm["requests"], phases)

                cases[model][scope][stage] = paired_sweep(edit_observed, observed)
            rates = {name: {w: by_width[name][w][stage] / 1000 for w in widths} for name in arms}

            def edit_batch(name, fraction):
                arm = arms[name]
                replay = simulate_decode(arm["requests"], arm["service"],
                                         step_savings_s={w: value * fraction for w, value in rates[name].items()})
                duration = replay["finish_s"] - min(row["offered_s"] for row in arm["requests"].values())
                return {"duration_s": duration, "throughput_tokens_s": replay["drained_throughput_tokens_s"],
                        "output_tokens": replay["output_tokens"], "terminal_request": replay["tail_id"]}

            cases["rebatch_all_steps"][scope][stage] = paired_sweep(edit_batch, dynamic)
    for name, arm in arms.items():
        for width, parts in by_width[name].items():
            if not isclose(sum(parts.values()), arm["periods_ms"][width], abs_tol=1e-10):
                raise ValueError("Stage parts must reconcile with the effective period")
    return {"fitted_stage_ms_per_step": by_width, **cases}


def analyze(results):
    priors = historical_priors(results)
    evidence, arms = batches(results)
    scenarios = {}
    for capture, info in priors["captures"].items():
        for split in ("widening_compute", "constant_shares"):
            scenarios[f"{capture}_{split}"] = replay_budgets(arms, info["shares"], split)
    return {
        "target_common_throughput_gain_pct": list(TARGETS_PCT), "r34_r37_priors": priors, "r43": evidence,
        "scenarios": scenarios,
        "scope": "Freeze planned offers, measured output counts with the natural EOS/original limit, observed P/D "
                 "placements and upstream residence. Successor releases and the tail can change. The confirmed "
                 "replay edits only agreeing counter windows. The stream-width replay freezes cohorts inferred "
                 "from observed client first and last token times, verified at engine counter samples. The rebatch "
                 "replay estimates all widths from the "
                 "other archived arm's measured pace and recomputes batch occupancy. Stage fractions transfer "
                 "old 100 Gbps single-sequence observations to 200 Gbps effective cadence. The two width "
                 "scalings keep either non-compute step costs constant or all shares constant. The host/runtime "
                 "budget is a ceiling for the inferred front gap only. Targets use a 1 percentage-point dose "
                 "grid and are required to hold in both arms at every remaining tested dose of that scenario. "
                 "The modeled percent cut must be exposed on the critical path. It is not realized hardware gain.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, analyze(args.results))


if __name__ == "__main__":
    main()
