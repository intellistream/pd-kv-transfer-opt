"""Select a recent benchmark workload and replay a frozen native bandwidth pair."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import gzip
from pathlib import Path

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.screen_bidirectional_workload import peak_demand
from pd_kv_transfer_opt.simulator.agent_replay import GB, Replay
from pd_kv_transfer_opt.simulator.agent_trace_gap import read_bytes
from pd_kv_transfer_opt.simulator.freeinference_ab import (
    PUBLISHED_BASELINE,
    calibration,
    paired,
    report,
    workload_info,
)
from pd_kv_transfer_opt.workloads.freeinference import load


def candidate_plans(sessions, metadata, service, byte_density_gate):
    # A single complete trace per benchmark question in any workload.
    waits = {s.name: sum(t.tool_gap_ms for t in s.turns) / 1000 for s in sessions}
    floors = {s.name: waits[s.name] + s.output * service.decode_first_s for s in sessions}
    candidates = sorted(sessions, key=lambda s: (-s.source_read_bytes / max(1., floors[s.name]), s.name))

    def unique(choices):
        taken = set()
        for s in choices:
            q = metadata[s.name]["first_user_sha256"]
            if q not in taken:
                taken.add(q)
                yield s

    dense = list(unique(s for s in candidates if s.source_read_bytes >= byte_density_gate * s.output
                        and floors[s.name] <= 180))
    short = list(unique(s for s in candidates if floors[s.name] <= 120 and waits[s.name] <= 45))
    medium = list(unique(s for s in candidates if floors[s.name] <= 240 and waits[s.name] <= 120))
    pressure = list(unique(s for s in candidates if floors[s.name] <= 360 and waits[s.name] <= 180))
    definitions = (("density", dense, 24, 0), ("density_paced", dense, 24, 1),
                   ("short_16", short, 16, 0), ("short_32", short, 32, 0),
                   ("medium_24", medium, 24, 0), ("medium_36", medium, 36, 0),
                   ("medium_36_paced", medium, 36, .5), ("pressure_48", pressure, 48, 0),
                   ("pressure_64", pressure, 64, 0), ("pressure_64_paced", pressure, 64, .5))
    plans, seen = {}, set()
    for name, choices, size, gap in definitions:
        picked = sorted(choices[:size], key=lambda s: (s.source_start_ms, s.name))
        roots = {s.name: i * gap for i, s in enumerate(picked)}
        fingerprint = tuple(roots.items())
        if len(picked) > 1 and fingerprint not in seen:
            plans[name] = picked, roots
            seen.add(fingerprint)
    return plans, {"sessions_passing_density_and_180s_floor": len(dense),
                   "sessions_passing_45s_original_wait_and_120s_floor": len(short),
                   "sessions_passing_120s_original_wait_and_240s_floor": len(medium),
                   "sessions_passing_180s_original_wait_and_360s_floor": len(pressure),
                   "optimistic_actual_MB_per_output_floor_at_D_peak": byte_density_gate / 1e6}


def peak_work_GB(offers):
    offers.sort()
    return {str(t): peak_demand(offers, t)["Gbps"] * t / 8 for t in (10, 30, 60, 120)}


def observed_read_correction(anchors):
    ratios = []
    for rounds in anchors.values():
        for row in rounds.values():
            if not row["conditional"]["comparable_P_routes_wire_bytes_and_EOS"]:
                continue
            low = row["closed_loop"]["100"]["GET_GB"]
            high = row["closed_loop"]["200"]["GET_GB"]
            ratios += [row["low_successor_egress_GB"] / low, row["high_successor_egress_GB"] / high]
    return max(ratios)


def run(args):
    anchored = read_json(args.anchor)
    metadata = read_json(args.receipt)
    if metadata["cohort_sha256"] != sha256(args.corpus):
        raise ValueError("Unverified rendered benchmark ledger")
    header, sessions = load(args.corpus)
    model_fits = calibration(args.results)
    pilot_service = model_fits["forward"][0]
    D_peak = 2 * pilot_service.slots / (
        pilot_service.decode_first_s + (pilot_service.slots - 1) * pilot_service.decode_add_s)
    gates = anchored["pressure_target"]
    rate_min, _ = gates["effective_100_GB_s_range"]
    target = gates["target_loss_relative_to_200"]
    density = rate_min * GB / ((1 - target) * D_peak)
    min_read_30 = gates["steady_30s_60s_GB"]["30"][0]
    plans, eligible = candidate_plans(sessions, metadata["sessions"], pilot_service, density)
    if not plans:
        raise ValueError("Complete benchmark sessions did not cover the screening plans")

    trials = {}
    for name, (selected, offers) in plans.items():
        run = Replay(selected, offers, pilot_service, 100).run()
        result = report(run)
        requests = run.requests.values()
        actual = peak_work_GB([(r.row["GET_submit_s"], r.row["P"], r.row["GET_bytes"], r.key)
                               for r in requests if r.row["GET_bytes"]])
        adjacent = peak_work_GB([(r.offered, r.row["P"], read_bytes(r.turn.adjacent_pages), r.key)
                                 for r in requests if r.turn.adjacent_pages])
        trials[name] = {"workload": workload_info(selected, offers), "100_initial": result,
                        "peak_offered_GET_GB": actual,
                        "all_adjacent_reads_at_release_same_clock_GB": adjacent,
                        "fraction_of_minimum_30s_read_budget": actual["30"] / min_read_30,
                        "natural_single_sequence_time_floor_s": max(
                            offers[s.name] + sum(t.tool_gap_ms for t in s.turns) / 1000
                            + s.output * pilot_service.decode_first_s for s in selected)}
        print("pilot", name, len(selected), round(result["GET_GB"], 1), round(actual["30"], 1),
              round(result["finish_s"], 1), round(result["drain_after_last_P_reply_s"], 2), flush=True)

    # Require enough simultaneous real work before looking at a bandwidth gain.
    # With no qualifying pilot, replay the closest controls to quantify the shortfall.
    ranked = sorted(trials, key=lambda name: (-trials[name]["fraction_of_minimum_30s_read_budget"],
                                             trials[name]["100_initial"]["drain_after_last_P_reply_s"], name))
    chosen = ranked[0]
    selected, offers = plans[chosen]
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    with gzip.open(args.trace, "wt", encoding="utf-8", compresslevel=4) as dest:
        for order, (service, _) in model_fits.items():
            results[order] = paired(selected, offers, service, order, workload=chosen, dest=dest)
        sparse = paired(selected, offers, replace(model_fits["reverse"][0], checkpoint_stride=4),
                        "reverse", workload=chosen + "_sparse", dest=dest)
        # Same service parameters and cold initial state in strict reversed order.
        reverse_same = paired(selected, offers, pilot_service, "reverse", detailed=False)
        strict_repeat = all(reverse_same["arms"][cap]["finish_s"] == results["forward"]["arms"][cap]["finish_s"]
                            and reverse_same["arms"][cap]["D_to_P_GB_by_port"] ==
                            results["forward"]["arms"][cap]["D_to_P_GB_by_port"] for cap in (100, 200))
        if not strict_repeat:
            raise RuntimeError("Fresh reverse replay did not reproduce the deterministic arm")

    correction = observed_read_correction(anchored["calibration"])
    main_pilot = trials[chosen]
    ratios = [metadata["sessions"][s.name]["reconstructed_to_source_prompt_ratio_median"] for s in selected]
    report_data = {"baseline_commit": PUBLISHED_BASELINE, "anchor_sha256": sha256(args.anchor),
                   "corpus": {**header, "sha256": sha256(args.corpus), "source_files": metadata["sources"],
                              "extraction_counts": metadata["counts"], "compatible_sessions": len(sessions)},
                   "screen": {"eligibility": eligible, "trials": trials, "ranked_by_30s_actual_GET": ranked,
                              "minimum_30s_actual_GET_GB": min_read_30,
                              "all_eligible_source_models_harnesses": dict(Counter(
                                  f"{r['harness']}/{r['source_model']}" for r in metadata["sessions"].values()))},
                   "models": {name: {"service": asdict(s), **m} for name, (s, m) in model_fits.items()},
                   "selected": {"pilot": chosen, "requests_trace_sha256": sha256(args.trace),
                                "metadata": [metadata["sessions"][s.name] for s in selected],
                                "Qwen_rendered_to_provider_input_ratio_median": percentile(ratios, .5),
                                "max_anchor_MAC_over_predicted_GET_ratio": correction,
                                "peak_30s_GB_with_anchor_read_shortfall_correction": correction * main_pilot["peak_offered_GET_GB"]["30"],
                                "ABBA": results, "strict_same_calibration_ABBA": strict_repeat,
                                "sparse_checkpoint_reverse": sparse},
                   "scope": "Public complete BrowseComp-Plus benchmark sessions, rendered with the audited Qwen tokenizer. "
                   "Source completion-token counts and stop/tool-call reasons drive EOS after dependent recorded gaps. "
                   "Empty native dual APC on both sides, same Store and routing policy, and equal root offers per arm. "
                   "Only D egress changes inside each pair. The corrected demand is a sizing sensitivity."}
    write_json(args.output, report_data)
    for order, case in results.items():
        print("selected", chosen, order,
              [round(case["arms"][cap]["throughput_tokens_s"], 2) for cap in (100, 200)],
              round(case["100_loss_relative_to_200"] * 100, 3), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("corpus", "receipt", "anchor", "trace", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
