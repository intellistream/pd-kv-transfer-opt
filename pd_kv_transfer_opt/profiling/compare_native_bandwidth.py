"""Compare a historical and a current native bandwidth run on identical session inputs."""

import argparse
from collections import Counter
from math import isclose
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union
from pd_kv_transfer_opt.profiling.profile_closed_sessions import (
    NS, check_observed, compare_arms, measure_ports, read_trace, replay, requests_for,
)
from pd_kv_transfer_opt.profiling.screen_mac_demand import replay_port


def decode_prompt_sources(arm):
    if "D_prompt_token_sources" in arm:
        return arm["D_prompt_token_sources"]
    sources = Counter()
    for role in ("D0", "D1"):
        for name, number in arm["engine_delta"][role].items():
            if name.startswith("vllm:prompt_tokens_by_source_total{"):
                bucket = name.split('source="')[1].split('"')[0]
                sources[bucket] += number
    if not sources:
        raise ValueError("Missing recorded D prompt sources")
    return dict(sources)


def p_http_overlap(row, spans):
    start, stop = row["P_start_s"], row["P_end_s"]
    return interval_union((max(start, left), min(stop, right)) for left, right in spans if left < stop and start < right)


def decode_egress_screen(events, rows, origin, cap, end, output, tail):
    intervals, _, hot, _ = measure_ports(events["telemetry"], round(origin * NS), end, cap)
    if cap == 100:
        costs = {key: {**row["phases_s"], "P_HTTP": max(0, row["phases_s"]["P_HTTP"] - p_http_overlap(row, hot))}
                 for key, row in rows.items()}
        _, done, _ = replay(rows, costs)
        complete = max(done.values())
        return {"sampled_any_D_port_near_cap_s": interval_union(hot),
                "tail_P_HTTP_overlap_s": sum(p_http_overlap(rows[key], hot) for key in tail),
                "removed_all_sampled_overlap_sensitivity": {"saved_s": end - complete, "output_tokens_s": output / complete}}
    ports = [replay_port(intervals, port, 12.5, end) for port in range(8)]
    waiting = [span for _, spans in ports for span in spans]
    return {"replayed_D_cap_100_Gbps_from_samples": {
        "longest_port_queue_s": max(row["longest_queue_s"] for row, _ in ports),
        "peak_queue_delay_s": max(row["peak_single_port_queue_delay_s"] for row, _ in ports),
        "tail_P_HTTP_overlap_s": sum(p_http_overlap(rows[key], waiting) for key in tail),
    }}


def measured_arm(events, workload, arm):
    origin, = [row["monotonic_s"] for row in events["client"] if row["type"] == "client_origin"]
    rows = requests_for(events, workload, origin)
    tail = check_observed(rows)
    total = arm["total"]
    tokens = sum(row["output_tokens"] for row in rows.values())
    duration = max(row["end_s"] for row in rows.values()) - min(row["offered_s"] for row in rows.values())
    if tokens != total["output_tokens"] or not isclose(tokens / duration, total["output_tokens_per_s"], rel_tol=1e-12):
        raise ValueError("Counterfactual input disagrees with recorded throughput")
    measured = {"output_tokens": tokens, "duration_s": duration, "throughput_tokens_s": tokens / duration,
                "tails": tail, "D_prompt_sources": decode_prompt_sources(arm),
                "D_MAC_TX_GB": arm.get("D_TX_GB", arm.get("D_tx_GB")),
                "D_MAC_RX_GB": arm.get("D_RX_GB", arm.get("D_rx_GB")),
                "D_MAC_peak_TX_1s_Gbps": arm.get("D_TX_peak_1s_Gbps", arm.get("D_peak_one_s_Gbps")),
                "D_aggregate_near_cap_s": arm.get("D_aggregate_above_85pct_s", arm.get("D_MAC_TX_at_least_85pct_cap_s")),
                "D_egress_screen": decode_egress_screen(events, rows, origin, arm["cap_Gbps"],
                                                        duration, tokens, tail["tail_chain"])}
    return rows, measured


def factor_throughput(low, high):
    """Average both substitution orders for the measured numerator and denominator."""
    n0, n1 = (row["output_tokens"] for row in (low, high))
    t0, t1 = (row["duration_s"] for row in (low, high))
    n_term = (n1 - n0) * (1 / t0 + 1 / t1) / 2
    t_term = (n0 + n1) * (1 / t1 - 1 / t0) / 2
    return {"output_token_delta": n1 - n0, "seconds_saved": t0 - t1,
            "output_work_term_tokens_s": n_term, "complete_time_term_tokens_s": t_term,
            "low_throughput_shortfall_pct": 100 * (1 - low["throughput_tokens_s"] / high["throughput_tokens_s"])}


def load(directory):
    manifest = read_json(directory / "manifest.json")
    path = directory / manifest.get("event_trace", "r28-events.jsonl.gz")
    if sha256(path) != manifest["artifacts"][path.name]["sha256"]:
        raise ValueError("Unverified event trace")
    summary = read_json(directory / "summary.json")["arms"]
    entries = {case: measured_arm(arm, manifest["workload"]["request_sources"], summary[case])
               for case, arm in read_trace(path).items()}
    return manifest, entries


def analyze(original, current):
    old_manifest, old = load(original)
    new_manifest, new = load(current)
    if old_manifest["workload"] != new_manifest["workload"] or old_manifest["software"] != new_manifest["software"]:
        raise ValueError("The runs must have identical workloads and software fingerprints")
    cases = old_manifest["case_order"]
    if new_manifest["case_order"] != cases or set(old) != set(cases) or set(new) != set(cases):
        raise ValueError("Different arm coverage or run order")
    if old_manifest["settings"]["engines"]["P"] != new_manifest["settings"]["engines"]["P"]:
        raise ValueError("The P setups differ")

    data = {}
    tail_ids = {summary["tails"]["tail"] for run in (old, new) for _, summary in run.values()}
    for manifest, run in ((old_manifest, old), (new_manifest, new)):
        entry = {"decode_settings": manifest["settings"]["engines"]["D"],
                 "shaping": manifest.get("shaping", "D egress only"), "arms": {}, "pairs": {}}
        for case in cases:
            rows, summary = run[case]
            entry["arms"][case] = {**summary, "tail_requests_across_runs": {key: rows[key] for key in sorted(tail_ids)}}
        for pair in ("b1", "b2"):
            low, low_measured = run[f"{pair}-cap100"]
            high, high_measured = run[f"{pair}-cap200"]
            entry["pairs"][pair] = {**factor_throughput(low_measured, high_measured),
                                      **compare_arms(low, high, low_measured, high_measured)}
        data[manifest["run_id"]] = entry

    cross_run = {}
    for case in cases:
        low, low_measured = old[case]
        high, high_measured = new[case]
        phases = compare_arms(low, high, low_measured, high_measured)
        cross_run[case] = {**factor_throughput(low_measured, high_measured),
                           "old_to_new_stage_swaps_s": phases["stage_swap_attribution_s"],
                           "matched_route_requests": phases["matched_route_requests"],
                           "matched_output_lengths": phases["matched_output_lengths"],
                           "D_MAC_RX_reduction_pct": 100 * (1 - high_measured["D_MAC_RX_GB"] / low_measured["D_MAC_RX_GB"])}
    return {"exact_workload_and_software_match": True, "case_order": cases, "runs": data,
            "old_to_new_same_cap": cross_run,
            "scope": "Work and duration contributions account for the observed throughput gap. "
                     "Stage swaps hold the observed session dependency graph. "
                     "The egress screen removes sampled P HTTP overlap on the 100 Gbps trace or replays 200 Gbps "
                     "departures as piecewise-constant per-port input at 100 Gbps. It keeps observed stage costs "
                     "outside the removed spans and does not model rebatching. "
                     "Changing deployment and shaping together does not isolate a single switch."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(args.original, args.current))


if __name__ == "__main__":
    main()
