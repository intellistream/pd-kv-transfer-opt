"""Propagate measured Decode-step sensitivities through the original r34 sessions."""

import argparse
from pathlib import Path
from statistics import mean

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_decode_steps import sampled_steps
from pd_kv_transfer_opt.profiling.analyze_kv_lifetime import change_finish
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, check_observed, read_trace, requests_for


def covered_tails(catalog, measured, trace, origin):
    timed, _ = sampled_steps(trace["d_native"])
    terminal = {}
    for role, tail in measured["terminal"].items():
        request = tail["last_request"]
        start = origin + tail["end_s"] - tail["tail_alone_after_first_token_s"]
        applicable = sorted((s for s in timed if s["instance"] == role and len(s["rows"]) == 1
                             and s["rows"][0][0] == request and s["worker_enter_ns"]/NS >= start
                             and s["mode"] == "FULL" and s["replayed_all_ranks"] and s["padded_tokens"] == 1),
                            key=lambda s: s["worker_enter_ns"])
        outside = [s for s in applicable if not any(s["worker_enter_ns"] < w["stopped_ns"]
                                                   and s["update_end_ns"] > w["request_ns"]
                                                   for w in trace["cann_windows"] if w["role"] == role)]
        assert len(outside) == tail["solo_after_first_samples"]["steps"]
        first, last = outside[0], outside[-1]
        n_steps = last["rows"][0][1] - first["rows"][0][1]
        assert 0 < n_steps <= catalog[request]["output_tokens"]
        duration = (last["worker_enter_ns"]-first["worker_enter_ns"])/NS
        terminal[role] = {"request": request, "observed_outputs": catalog[request]["output_tokens"],
                          "matched_non_cann_samples": len(outside), "covered_steps": n_steps,
                          "covered_interval_s": duration, "first_offset": first["rows"][0][1],
                          "last_offset": last["rows"][0][1], "effective_ms_per_token": duration*1000/n_steps,
                          "worker_period_ms": distribution(s["cost"]["rank0_worker_gap_ms"] for s in outside),
                          "slowest_rank_forward_ms": distribution(s["cost"]["slowest_rank_forward_device_ms"]
                                                                  for s in outside),
                          "rank0_after_forward_stream_ms": distribution(s["cost"]["rank0_after_forward_stream_ms"]
                                                                        for s in outside)}
    return terminal


def calibrate_native_period(period_ms, capture):
    """Use the observed device partition as a calibration, with front slack as the residual."""
    observed = capture["coverage"]
    components = {key: observed[key]["mean"] for key in (
        "compute_ms", "collective_after_last_ms", "copy_ms", "device_event_wait_ms")}
    components["after_first_two_prejoins_uncovered_ms"] = mean(
        row["uncovered_ms"] - row["first_prejoin_uncovered_ms"] - row["second_prejoin_uncovered_ms"]
        for row in capture["steps"])
    residual = period_ms - sum(components.values())
    if residual < 0:
        raise ValueError("Device and post-front coverage exceed the native cadence")
    return {"native_period_ms": period_ms, "calibrated_parts_ms": {
        **components, "uncovered_before_first_two_joins_residual_ms": residual},
        "scope": "Estimated from captured device components and native median cadence"}


def analyze(root, probe):
    trace_path = root / "r34-events.jsonl.gz"
    manifest = read_json(root / "manifest.json")
    measured = read_json(root / "summary.json")
    assert sha256(trace_path) == measured["trace_sha256"]
    trace, = read_trace(trace_path).values()
    origin, = (row["monotonic_s"] for row in trace["client"] if row["type"] == "client_origin")
    catalog = requests_for(trace, manifest["workload"]["request_sources"], origin)
    check = check_observed(catalog)
    terminal = covered_tails(catalog, measured, trace, origin)
    before = measured["baseline"]["duration_s"]
    output = sum(request["output_tokens"] for request in catalog.values())
    assert output == measured["baseline"]["output_tokens"]
    baseline = output/before

    def counterfactual(delta_ms, roles):
        advance = {tail["request"]: tail["covered_steps"]*delta_ms/1000 for role,tail in terminal.items() if role in roles}
        timing = change_finish(catalog, advance, "first_to_end")
        duration = before - timing["advance_s"]
        throughput = output/duration
        return {"assumed_step_saving_ms": delta_ms, "advance_s": timing["advance_s"], "duration_s": duration,
                "throughput_tokens_s": throughput, "gain_pct": 100*(throughput/baseline-1),
                "stage_reduction_s": advance, "new_tail_chain": timing["resulting_tail_chain"]}

    settings = {scope: {str(ms): counterfactual(ms, roles) for ms in (0.25, 0.5, 1, 2)}
                for scope, roles in (("D0", ("D0",)), ("D0_D1", ("D0", "D1")))}
    profile_based = {}
    for key, stats in probe["coverage"].items():
        if not key.startswith("save_"):
            continue
        profile_based[key] = {scope: counterfactual(stats["p50"], roles)
                              for scope, roles in (("D0", ("D0",)), ("D0_D1", ("D0", "D1")))}
    return {"r34_trace_sha256": sha256(trace_path),
            "fixed_actual_requests": len(catalog), "fixed_output_tokens": output, "baseline_duration_s": before,
            "baseline_throughput_tokens_s": baseline, "replay_baseline_error_s": check["max_error_s"],
            "other_session_end_s": check["next_session_end_s"],
            "tail": terminal, "conditional_absolute_step_saving": settings,
            "conditional_stage_saving_from_profile": profile_based,
            "native_calibration_from_probe": calibrate_native_period(terminal["D0"]["worker_period_ms"]["p50"], probe)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("r34", type=Path)
    parser.add_argument("probe_breakdown", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--old-capture", type=Path)
    args = parser.parse_args()
    output = analyze(args.r34, read_json(args.probe_breakdown))
    output["probe_breakdown_sha256"] = sha256(args.probe_breakdown)
    if args.old_capture is not None:
        output["native_calibration_from_old_capture"] = calibrate_native_period(
            output["tail"]["D0"]["worker_period_ms"]["p50"], read_json(args.old_capture))
        output["old_capture_sha256"] = sha256(args.old_capture)
    write_json(args.output, output)


if __name__ == "__main__":
    main()
