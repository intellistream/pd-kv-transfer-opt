"""Audit fixed-work bandwidth replay, independent GET timing and the r28 comparison."""

from __future__ import annotations

import argparse
from collections import defaultdict
from gzip import open as gzip_open
from itertools import combinations
import json
from math import comb, isclose
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import get_batches
from pd_kv_transfer_opt.profiling.compare_native_bandwidth import decode_prompt_sources, factor_throughput, load
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, compare_arms, measure_arms
from pd_kv_transfer_opt.simulator.decode_supply import simulate_decode
from pd_kv_transfer_opt.simulator.rdma_replay import HIGH_CAP, LOW_CAP, PORTS, egress_epochs, load_native_archive
from pd_kv_transfer_opt.simulator.terminal_pressure import ORDERS, planned_recovery_start, supply


FACTORS = ("P_HTTP", "other_before_first", "D_route", "EOS_work")
FIXED_REQUEST = ("session_id", "predecessor", "offered_s", "tool_gap_s", "P")


def compare_decode_inputs(low, high, service, slots):
    """Vary observed phase durations, placement and EOS on a fixed request DAG."""
    if low.keys() != high.keys() or any(low[key][field] != high[key][field] for key in low for field in FIXED_REQUEST):
        raise ValueError("Replay annotations must describe the same request DAG and P placement")
    scenarios = {}
    first = min(row["offered_s"] for row in low.values())
    for count in range(len(FACTORS) + 1):
        for factors in combinations(FACTORS, count):
            selected = {factor: high if factor in factors else low for factor in FACTORS}
            requests = {}
            for key, base in selected["other_before_first"].items():
                duration = selected["P_HTTP"][key]["phases_s"]["P_HTTP"]
                delta = duration - base["phases_s"]["P_HTTP"]
                requests[key] = {**base, "first_s": base["first_s"] + delta,
                                 "phases_s": {**base["phases_s"], "P_HTTP": duration},
                                 "D": selected["D_route"][key]["D"],
                                 "output_tokens": selected["EOS_work"][key]["output_tokens"]}
            replay = simulate_decode(requests, service, slots=slots)
            scenarios[frozenset(factors)] = {"output_tokens": replay["output_tokens"],
                                            "duration_s": replay["finish_s"] - first,
                                            "throughput_tokens_s": replay["drained_throughput_tokens_s"],
                                            "tail_id": replay["tail_id"]}
    all_factors, none = frozenset(FACTORS), frozenset()
    only_prefill = frozenset({"P_HTTP"})
    endpoints = {"100": scenarios[none], "200": scenarios[all_factors]}
    anchors = {"100": scenarios[only_prefill], "200": scenarios[all_factors - only_prefill]}
    terms = {}
    for metric, sign in (("duration_s", -1), ("throughput_tokens_s", 1)):
        by_factor = {}
        for factor in FACTORS:
            by_factor[factor] = sum(sign * (after[metric] - scenarios[mask - {factor}][metric])
                                    / (len(FACTORS) * comb(len(FACTORS) - 1, len(mask) - 1))
                                    for mask, after in scenarios.items() if factor in mask)
        expected = sign * (endpoints["200"][metric] - endpoints["100"][metric])
        if not isclose(sum(by_factor.values()), expected, abs_tol=1e-8):
            raise ValueError("Stage effects must sum to the modeled aggregate change")
        terms[metric] = by_factor
    return {"all_observed_inputs_in_each_cap": endpoints, "only_P_HTTP_taken_from_other_cap": anchors,
            "symmetric_seconds_saved_by_factor": terms["duration_s"],
            "symmetric_throughput_gain_by_factor_tokens_s": terms["throughput_tokens_s"]}


def artifact(path, manifest):
    digest = sha256(path)
    if digest != manifest["artifacts"][path.name]["sha256"]:
        raise ValueError(f"Archive hash mismatch for {path}")
    return read_json(path), digest


def verify_method_groups(events, matches, largest):
    expected = {(row["P"], tuple(row["request_ids"])): row for row in matches}
    cases = defaultdict(list)
    with gzip_open(events, "rt", encoding="utf-8") as stream:
        for record in map(json.loads, stream):
            row = record["data"]
            if row["instance"].startswith("P"):
                cases[record["case"]].append(row)
    joined_in_both = set(expected)
    for cap in (LOW_CAP, HIGH_CAP):
        rows = cases[f"b1-cap{cap}"]
        pending = {}
        observed = defaultdict(dict)
        for row in sorted(rows, key=lambda event: event["monotonic_ns"]):
            rank = row["instance"], row["pid"]
            kind = row["type"]
            if kind == "worker_load_begin":
                batch = row["instance"], tuple(sorted(row["request_ids"]))
                if batch in expected:
                    pending[rank] = {"batch": batch, "begin_ns": row["monotonic_ns"], "bytes": 0, "get_ns": 0, "errors": 0}
            elif kind == "store_end" and rank in pending and row["operation"] == "batch_get_into_multi_buffers":
                pending[rank]["bytes"] += row["payload_bytes"]
                pending[rank]["get_ns"] += row["elapsed_ns"]
                pending[rank]["errors"] += row["negative_returns"]
            elif kind == "worker_load_end" and rank in pending:
                load = pending.pop(rank)
                batch = load.pop("batch")
                if rank in observed[batch]:
                    raise ValueError("A matched rank appears more than once in the archived load")
                observed[batch][rank] = {**load, "end_ns": row["monotonic_ns"], "load_ns": row["elapsed_ns"]}
        if pending:
            raise ValueError("A matched rank did not finish its loading interval")
        _, chunks, _ = get_batches(rows)
        joined = set()
        for batch, declaration in expected.items():
            ranks = observed[batch].values()
            recorded = declaration[f"{cap}_Gbps"]
            if (len(ranks) != 4 or sorted(row["bytes"] for row in ranks) != declaration["rank_bytes"]
                    or any(row["errors"] for row in ranks)
                    or max(row["get_ns"] for row in ranks) / NS != recorded["slowest_GET_sum_s"]
                    or max(row["load_ns"] for row in ranks) / NS != recorded["slowest_worker_load_s"]):
                raise ValueError("Raw rank load events disagree with the recorded payload match")
            first = min(row["begin_ns"] for row in ranks)
            last = max(ranks, key=lambda row: row["end_ns"])
            role, ids = batch
            chunks_for_load = [row for row in chunks if row["instance"] == role and row["dispatch_ns"] <= first
                               and last["end_ns"] <= row.get("ready_ns", 0) and set(ids) <= {r[0] for r in row["rows"]}]
            if len(chunks_for_load) == 1:
                joined.add(batch)
                chunk, = chunks_for_load
                if batch == (largest["P"], tuple(largest["request_ids"])):
                    measured = {"dispatch_to_ready_s": (chunk["ready_ns"] - chunk["dispatch_ns"]) / NS,
                                "dispatch_to_last_rank_load_start_s": (last["begin_ns"] - chunk["dispatch_ns"]) / NS,
                                "last_rank_GET_inside_load_s": last["get_ns"] / NS,
                                "last_rank_other_load_s": (last["load_ns"] - last["get_ns"]) / NS,
                                "last_rank_load_to_ready_s": (chunk["ready_ns"] - last["end_ns"]) / NS}
                    if measured != largest[f"{cap}_Gbps"]["engine_chunk_using_same_P_clock"]:
                        raise ValueError("The largest matched chunk disagrees with the raw same-clock events")
        joined_in_both.intersection_update(joined)
    if (largest["P"], tuple(largest["request_ids"])) not in joined_in_both:
        raise ValueError("The largest match must join a unique scheduler chunk in both caps")
    return len(joined_in_both)


def matched_get(path, selected_manifest):
    manifest, trace_digest, _ = load_native_archive(path)
    settings = lambda m: {k: v for k, v in m["settings"].items() if k != "serving_observer"}
    if (not manifest["independent_method_pass"] or manifest["workload"] != selected_manifest["workload"]
            or settings(manifest) != settings(selected_manifest) or manifest["software"] != selected_manifest["software"]):
        raise ValueError("Independent method timing must use the selected workload and configuration")
    chain, digest = artifact(path / "native-chunk-chain.json", manifest)
    summary, _ = artifact(path / "native-method-summary.json", manifest)
    events = path / "native-method-events.jsonl.gz"
    if sha256(events) != chain["method_events_sha256"] or chain["method_events_sha256"] != summary["method_events_sha256"]:
        raise ValueError("The matched GET records do not refer to the archived method trace")
    matches = summary["matched_four_rank_GET_batches"]
    if len(matches) != chain["matched_batches"] or any(sum(row["rank_bytes"]) != row["bytes"] for row in matches):
        raise ValueError("The recorded matched GET payloads do not reconcile")
    largest = chain["longest_matched_payload"]
    joined = verify_method_groups(events, matches, largest)
    if joined != chain["joined_engine_chunks"]:
        raise ValueError("The independently joined chunks disagree with the archive")
    phases = [largest[f"{cap}_Gbps"]["engine_chunk_using_same_P_clock"] for cap in (LOW_CAP, HIGH_CAP)]
    for row in phases:
        if not isclose(sum(value for name, value in row.items() if name != "dispatch_to_ready_s"),
                       row["dispatch_to_ready_s"], abs_tol=.005):
            raise ValueError("The matched engine chunk intervals do not reconcile")
    return {"archive": path.name, "timing_trace_sha256": trace_digest, "matched_record_sha256": digest,
            "method_events_sha256": chain["method_events_sha256"], "matched_batches": len(matches),
            "matched_payload_bytes": sum(row["bytes"] for row in matches),
            "higher_cap_GET_faster_batches": sum(row["200_Gbps"]["slowest_GET_sum_s"] < row["100_Gbps"]["slowest_GET_sum_s"]
                                                 for row in matches), "chunks_joined": joined,
            "largest_joined_batch": largest,
            "largest_batch_get_seconds_saved": phases[0]["last_rank_GET_inside_load_s"] - phases[1]["last_rank_GET_inside_load_s"],
            "largest_batch_dispatch_seconds_saved": phases[0]["dispatch_to_ready_s"] - phases[1]["dispatch_to_ready_s"]}


def verify_frozen_replay(screened, recorded, manifest, arms, archive):
    if (recorded["source_manifest_sha256"] != sha256(archive / "manifest.json")
            or any(manifest["workload"][name] != value for name, value in recorded["workload"].items())
            or recorded["settings"] != manifest["settings"]):
        raise ValueError("The selected replay differs from the frozen workload or engine settings")
    if any(row["finish_reason"] != "stop" for _, requests, _ in arms.values() for row in requests.values()):
        raise ValueError("An arm of the selected workload did not reach natural EOS")
    for order in ORDERS:
        high = arms[f"{order}-cap{HIGH_CAP}"]
        expected_tokens = sum(row["output_tokens"] for row in high[1].values())
        intervals = egress_epochs(high)
        measured = [sum((end - start) * rates[port] * 1e9 / 8 for start, end, rates in intervals)
                    for port in range(PORTS)]
        for name in ("prediction", "short_burst_sensitivity"):
            case = screened["selected_bandwidth_pairs"][order][name]
            a, b = (case[str(cap)] for cap in (LOW_CAP, HIGH_CAP))
            if (case["pressure_window_length_multiple"] != 1 or a["D_port_TX_bytes"] != b["D_port_TX_bytes"]
                    or any(not isclose(byte_count, actual, rel_tol=1e-12) for byte_count, actual in
                           zip(b["D_port_TX_bytes"], measured, strict=True))
                    or a["output_tokens"] != b["output_tokens"] or b["output_tokens"] != expected_tokens):
                raise ValueError("Bandwidth replay altered the frozen output or offered port bytes")


def tail_evidence(requests, summary):
    final = summary["observed_dependencies"]["tail"]
    late = max((key for key, row in requests.items() if row["predecessor"] is not None),
               key=lambda key: requests[key]["ready_s"] - requests[key]["offered_s"])
    subjects = {final, late, max(requests, key=lambda key: requests[key]["P_end_s"])}
    fields = ("predecessor", "offered_s", "ready_s", "P_end_s", "first_s", "end_s", "output_tokens", "P", "D")
    return {"tail_id": final, "most_delayed_successor": late,
            "requests": {key: {name: requests[key][name] for name in fields} for key in sorted(subjects)}}


def audit(screening, results):
    screened = read_json(screening)
    frozen = screening.with_name("selected-workload.json")
    if sha256(frozen) != screened["selected_workload_sha256"]:
        raise ValueError("The frozen workload digest is invalid")
    recorded = read_json(frozen)
    current_path = results / screened["selection"]["archive"]
    manifest, digest, arms = load_native_archive(current_path)
    if digest != screened["selection"]["trace_sha256"] or digest in screened["decode_calibration_traces"].values():
        raise ValueError("This audit requires the selected trace held out from calibration")
    verify_frozen_replay(screened, recorded, manifest, arms, current_path)
    raw, raw_digest = artifact(current_path / "summary.json", manifest)
    slots = {role: row["max_num_seqs"] for role, row in manifest["settings"]["engines"].items()}
    recorded_metrics, current, history = {}, {}, {}
    for case, (stream, requests, origin) in arms.items():
        cap = int(case.rsplit("-cap", 1)[1])
        measurement = measure_arms(stream, requests, origin, cap, slots)
        recorded_metrics[case] = measurement
        actual = raw["arms"][case]["total"]
        if (measurement["output_tokens"] != actual["output_tokens"]
                or not isclose(measurement["throughput_tokens_s"], actual["output_tokens_per_s"], rel_tol=1e-12)):
            raise ValueError(f"Selected arm {case} disagrees with the independently recomputed throughput")
        current[case] = {key: measurement[key] for key in ("output_tokens", "duration_s", "throughput_tokens_s",
                         "D_TX_GB", "D_aggregate_above_85pct_s", "D_any_port_above_85pct_s")}
        current[case].update(D_prompt_sources=decode_prompt_sources(raw["arms"][case]),
                             tail=tail_evidence(requests, measurement),
                             supply=supply(requests, requests, planned_recovery_start(requests)),
                             D_without_ready_work_during_port_pressure_s={role: measurement["P_D_resource_samples"][role]
                             ["overlap_with_any_hot_port_s"] for role in ("D0", "D1")})

    prior_path = results / screened["historical_r28"]["archive"]
    historical_summary_digest = sha256(prior_path / "summary.json")
    if historical_summary_digest != screened["historical_r28"]["summary_sha256"]:
        raise ValueError("The historical comparison changed after selection")
    old_manifest, old = load(prior_path)
    for order in ORDERS:
        low, high = (f"{order}-cap{cap}" for cap in (LOW_CAP, HIGH_CAP))
        older_low, older_high = old[low], old[high]
        low_rows, high_rows = arms[low][1], arms[high][1]
        history[order] = {"arms": {"100": older_low[1], "200": older_high[1]},
                          "observed_work_and_time": factor_throughput(older_low[1], older_high[1]),
                          "observed_phase_swaps": compare_arms(older_low[0], older_high[0], older_low[1], older_high[1])}
        current[order] = {"observed_work_and_time": factor_throughput(current[low], current[high]),
                          "observed_phase_swaps": compare_arms(low_rows, high_rows, recorded_metrics[low], recorded_metrics[high]),
                          "calibrated_decode_attribution": compare_decode_inputs(low_rows, high_rows,
                                                                                 screened["decode_service_fit"], slots["D"])}
    methods = current_path.with_name(current_path.name + "-methods")
    return {"screening_sha256": sha256(screening), "selected_workload_sha256": sha256(frozen),
            "current": {"archive": current_path.name, "trace_sha256": digest, "summary_sha256": raw_digest,
                        "arms_and_pairs": current, "matched_GET": matched_get(methods, manifest)},
            "r28": {"archive": prior_path.name, "trace_sha256": sha256(prior_path / old_manifest.get("event_trace", "r28-events.jsonl.gz")),
                    "summary_sha256": historical_summary_digest,
                    "D_APC_enabled": old_manifest["settings"]["engines"]["D"]["enable_prefix_caching"], "pairs": history},
            "scope": ("The primary 100/200 replay retains the exact same archived bytes and natural EOS on each anchor. "
                      "Diagnostic swaps vary only observed P or D factors to compare exposure after controlling output and routing. "
                      "The historical trace is analyzed in its own deployment. Its observed throughput ratio is not a pure bandwidth estimate.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, audit(args.screening, args.results))


if __name__ == "__main__":
    main()
