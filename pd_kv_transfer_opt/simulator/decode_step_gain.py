"""From net milliseconds saved per covered D step to native r43 drained throughput."""

import argparse
from collections import defaultdict
from pathlib import Path

from pd_kv_transfer_opt.common import write_json
from pd_kv_transfer_opt.profiling.compare_session_runs import DECODE, timeline
from pd_kv_transfer_opt.simulator.decode_stage_budget import batches, historical_priors
from pd_kv_transfer_opt.simulator.decode_supply import simulate_decode


SCOPES = {
    "1-4": tuple(range(1, 5)),
    "9-16": tuple(range(9, 17)),
    "1-4_and_9-16": (*range(1, 5), *range(9, 17)),
    "1-16": tuple(range(1, 17)),
}
MODELS = ("confirmed_windows", "observed_stream_width", "rebatch")


def exposures(arm, widths):
    confirmed, streams = defaultdict(float), defaultdict(float)
    for width, active, _, tokens in arm["windows"]:
        if width in widths:
            for key in active:
                confirmed[key] += tokens / width
    for rows in arm["stream_intervals"].values():
        for begin, end, active in rows:
            width = len(active)
            if width in widths:
                steps = (end - begin) * 1000 / arm["periods_ms"][width]
                for key in active:
                    streams[key] += steps
    return {"confirmed_windows": confirmed, "observed_stream_width": streams}


def project(arm, model, widths, steps, ms):
    requests = arm["requests"]
    if model == "rebatch":
        base = arm["dynamic"]
        sim = simulate_decode(requests, arm["service"], step_savings_s=dict.fromkeys(widths, ms / 1000))
        origin = min(row["offered_s"] for row in requests.values())
        result = {"duration_s": sim["finish_s"] - origin, "output_tokens": sim["output_tokens"],
                  "throughput_tokens_s": sim["drained_throughput_tokens_s"], "terminal_request": sim["tail_id"]}
    else:
        base = arm["observed"]
        phases = {key: {**row, DECODE: row[DECODE] - steps[model][key] * ms / 1000}
                  for key, row in arm["costs"].items()}
        if any(row[DECODE] < 0 for row in phases.values()):
            raise ValueError("Step savings exceed the observed Decode residence")
        result = timeline(requests, phases)
    return {**result, "gain_pct": 100 * (base["duration_s"] / result["duration_s"] - 1),
            "earlier_finish_s": base["duration_s"] - result["duration_s"]}


def analyze(results):
    references, arms = batches(results)
    priors = historical_priors(results)
    doses = {"1_ms": 1., "1.5_ms": 1.5, "2_ms": 2.}
    for name, capture in priors["captures"].items():
        doses[f"{name}_compute_20pct"] = capture["old_capture_20pct_saving_ms_p50"]["compute"]
    scopes = {}
    for scope, widths in SCOPES.items():
        steps = {name: exposures(arm, widths) for name, arm in arms.items()}
        scope_cases = {}
        for label, ms in doses.items():
            models = {}
            for model in MODELS:
                cases = {name: project(arm, model, widths, steps[name], ms) for name, arm in arms.items()}
                models[model] = {"arms": cases, "both_arms_gain_pct": min(row["gain_pct"] for row in cases.values())}
            scope_cases[label] = {"net_ms_per_eligible_step": ms, "models": models}
        scopes[scope] = scope_cases
    return {"r43": references, "historical_compute_dose_source": {name: capture["breakdown_sha256"]
                                                               for name, capture in priors["captures"].items()},
            "scope_cases": scopes,
            "interpretation": "Only net overlap-adjusted savings that advance the next effective D step are "
                              "applied. Same per-arm EOS token counts, original routes and offers. Reevaluate "
                              "closed-session releases and the terminal chain. Old 20 percent compute doses "
                              "are four-rank critical-path projections on the older singleton captures, "
                              "used here as conditional absolute savings at the selected widths."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, analyze(args.results))


if __name__ == "__main__":
    main()
