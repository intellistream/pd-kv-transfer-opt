"""Compare natural and matched service runs on the same closed sessions."""

import argparse
from collections import Counter
from itertools import pairwise, takewhile
from math import isclose
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_native_prefill_path import GET, batch_and_request_index, clipped_union
from pd_kv_transfer_opt.profiling.profile_closed_sessions import (
    NS, check_observed, compare_arms, gauge, measure_ports, phase_times, read_trace, replay, requests_for,
)


ARCHIVES = {
    "r28": "2026-11-19-r28-natural-bandwidth",
    "r33": "2026-11-20-r33-native-prefill-path",
    "r38": "2026-11-20-r38-packed-reproduction",
    "r39": "2026-11-20-r39-bottleneck-verification",
}
DECODE = "first_to_end"
ANCHORS = ("natural-crack-7z-hash:31", "natural-extract-elf:11")
D_ROLES = ("D0", "D1")


def last_key(profiles):
    return max(profiles, key=lambda key: profiles[key]["end_s"])


def total_output(profiles):
    return sum(row["output_tokens"] for row in profiles.values())


def timeline(profiles, costs):
    _, finish, chain = replay(profiles, costs)
    duration = max(finish.values()) - min(row["offered_s"] for row in profiles.values())
    tokens = total_output(profiles)
    return {"duration_s": duration, "output_tokens": tokens, "throughput_tokens_s": tokens / duration,
            "terminal_request": chain[-1]}


def singleton_samples(telemetry, origin, role, start, end):
    points = []
    for sample in telemetry:
        for engine in sample.get("engines", []):
            if engine["role"] == role:
                values = engine.get("values", {})
                points.append((engine["monotonic_ns"] / NS - origin,
                               gauge(values, "num_requests_running"), gauge(values, "num_requests_waiting"),
                               gauge(values, "generation_tokens_total")))
    seconds, tokens, intervals = 0.0, 0, 0
    for (left, running, waiting, produced), (right, run_end, wait_end, end_tokens) in pairwise(points):
        if (start <= left < right <= end and right - left <= 6 and running == run_end == 1
                and waiting == wait_end == 0 and produced is not None and end_tokens is not None
                and produced < end_tokens):
            seconds += right - left
            tokens += int(end_tokens - produced)
            intervals += 1
    return {"intervals": intervals, "covered_s": seconds, "output_tokens": tokens,
            "ms_per_token": 1000 * seconds / tokens if tokens else None}


def decode_tails(profiles, telemetry, origin):
    out = {}
    for role in D_ROLES:
        group = {key: row for key, row in profiles.items() if row["D"] == role}
        key = last_key(group)
        row = group[key]
        latest_peer = max(value["end_s"] for req, value in group.items() if req != key)
        start = max(row["first_s"], latest_peer)
        out[role] = {"requests": len(group), "output_tokens": total_output(group), "terminal_request": key,
                     "end_s": row["end_s"], "terminal_output_tokens": row["output_tokens"],
                     "terminal_post_first_s": row["phases_s"][DECODE], "terminal_TPOT_ms": row["effective_TPOT_ms"],
                     "singleton_s": row["end_s"] - start,
                     "singleton_counter": singleton_samples(telemetry, origin, role, start, row["end_s"])}
    return out


def routing_and_transport(profiles, telemetry, origin, cap):
    switches = Counter()
    for row in profiles.values():
        if row["predecessor"] is None:
            continue
        previous = profiles[row["predecessor"]]
        for role in ("P", "D"):
            switches[role] += row[role] != previous[role]
    stop = profiles[last_key(profiles)]["end_s"]
    intervals, hot, _, peak = measure_ports(telemetry, round(origin * NS), stop, cap)
    mac_tx_gb = sum((end - begin) * rate / 8 for begin, end, rate, _ in intervals)
    return {"switches_from_preceding_turn": dict(switches), "D_MAC_TX_GB": mac_tx_gb,
            "D_MAC_TX_peak_1s_Gbps": peak, "D_MAC_TX_at_or_above_85pct_s": sum(end - begin for begin, end in hot)}


def compare_same_session(first, second, session):
    """Account for actual time differences on one session, even when the global tail changes."""
    a, b = ({key: row for key, row in profiles.items() if row["session_id"] == session}
            for profiles in (first, second))
    a_end, b_end = (profiles[last_key(profiles)]["end_s"] for profiles in (a, b))
    changes, turns = Counter(), {}
    for key, left in a.items():
        right = b[key]
        n, m = left["output_tokens"], right["output_tokens"]
        left_cost, right_cost = left["phases_s"], right["phases_s"]
        unit_a, unit_b = left_cost[DECODE] / n, right_cost[DECODE] / m
        work = (n - m) * (unit_a + unit_b) / 2
        pace = (n + m) * (unit_a - unit_b) / 2
        before = sum(left_cost[p] - right_cost[p] for p in left_cost if p != DECODE)
        changes.update({"before_first_s": before, "output_length_s": work, "effective_cadence_s": pace})
        turns[key] = {"output_tokens": [n, m], "outputs_after_first_s": [left_cost[DECODE], right_cost[DECODE]],
                      "P_HTTP_s": [left_cost["P_HTTP"], right_cost["P_HTTP"]],
                      "output_length_s": work, "effective_cadence_s": pace}
        previous = left["predecessor"]
        release_a = a[previous]["end_s"] if previous is not None else 0.0
        release_b = b[previous]["end_s"] if previous is not None else 0.0
        changes["release_gap_s"] += (left["ready_s"] - release_a) - (right["ready_s"] - release_b)
    if not isclose(sum(changes.values()), a_end - b_end, abs_tol=1e-6):
        raise ValueError(f"Unreconciled session clock in {session}")
    return {"finish_s": [a_end, b_end], "time_savings_s": dict(changes), "turns": turns}


def decode_scale_to_target(profiles, target_throughput):
    target = total_output(profiles) / target_throughput
    costs = phase_times(profiles)

    def finish(scale):
        return timeline(profiles, {key: {**stages, DECODE: scale * stages[DECODE]}
                                   for key, stages in costs.items()})["duration_s"]

    if target >= finish(1):
        return {"required_duration_s": target, "decode_residency_reduction_pct": 0.0}
    if target < finish(0):
        return {"required_duration_s": target, "decode_residency_reduction_pct": None}
    left, right = 0.0, 1.0
    for _ in range(42):
        middle = (left + right) / 2
        if finish(middle) <= target:
            left = middle
        else:
            right = middle
    return {"required_duration_s": target, "decode_residency_reduction_pct": 100 * (1 - left)}


def compare(first, second, before, after):
    stage = compare_arms(first, second, before, after)
    base_session, new_session = (profiles[last_key(profiles)]["session_id"] for profiles in (first, second))
    session = compare_same_session(first, second, base_session)
    return {"before": before, "after": after,
            "throughput_change_pct": 100 * (after["throughput_tokens_s"] / before["throughput_tokens_s"] - 1),
            "elapsed_savings_s": before["duration_s"] - after["duration_s"],
            "same_route_requests": stage["matched_route_requests"], "same_output_lengths": stage["matched_output_lengths"],
            "baseline_terminal_session": base_session,
            "baseline_session_comparison": session,
            "target_terminal_session": new_session,
            "target_finish_after_baseline_session_s": second[last_key(second)]["end_s"] - session["finish_s"][1],
            "observed_phase_swap_s": stage["stage_swap_attribution_s"]}


def cold_p1(events, profiles):
    rows = [row for row in events if row["instance"] == "P1"]
    arrivals, batches, scheduled, _, waits, errors = batch_and_request_index(rows)
    if errors:
        raise ValueError(f"Could not join P1 batches: {errors[:3]}")
    batches.sort(key=lambda row: row["dispatch_ns"])
    cold = list(takewhile(lambda batch: all(profiles[key]["predecessor"] is None for key, _, _ in batch["rows"]), batches))
    first = batches[0]["rows"][0][0]
    first_group = list(takewhile(lambda batch: all(key == first for key, _, _ in batch["rows"]), cold))
    shape = [row["rows"] for row in first_group]
    prefix = batches[:len(cold) + 1]
    starts = [row["dispatch_ns"] for row in prefix]
    reads = [(row["monotonic_ns"] - row["elapsed_ns"], row["monotonic_ns"]) for row in rows
             if row["type"] == "store_end" and row["operation"] == GET]

    def window(count):
        work = Counter()
        for batch in prefix[:count]:
            for key, _, tokens in batch["rows"]:
                work[key] += tokens
        spans = (starts[0], starts[count])
        times = [row.get("device", {}).get("forward_device_ms") for row in prefix[:count]]
        return {"batches": count, "scheduled_tokens": sum(work.values()), "tokens_by_request": dict(sorted(work.items())),
                "wall_s": (spans[1] - spans[0]) / NS, "result_wait_s": clipped_union(waits["P1"], spans) / NS,
                "Store_GET_union_s": clipped_union(reads, spans) / NS,
                "timed_batches": sum(value is not None for value in times),
                "timed_rank0_forward_s": sum(value for value in times if value is not None) / 1000}

    anchors = {key: window(batches.index(scheduled[key][0][0])) for key in ANCHORS}
    worker_phases = ("worker_enter_to_forward_s", "forward_host_s", "forward_return_to_next_worker_s")
    worker_rows, worker_totals = [], Counter()
    for batch, following in pairwise(prefix[:len(first_group) + 1]):
        device = batch["device"]
        gap = (device["forward_cpu_return_ns"], following["device"]["worker_enter_ns"])
        marks = (device["worker_enter_ns"], device["forward_enter_ns"], *gap)
        spans = dict(zip(worker_phases, ((b - a) / NS for a, b in pairwise(marks)), strict=True))
        if min(spans.values()) < 0:
            raise ValueError(f"Out-of-order P1 worker events in batch {batch['batch']}")
        spans["result_wait_overlapping_worker_gap_s"] = clipped_union(waits["P1"], gap) / NS
        worker_totals.update(spans)
        worker_rows.append({"offset": batch["rows"][0][1], **spans})
    measured = {"first_request": first, "first_request_only_window": window(len(first_group)),
                "first_request_rank0_worker_timeline": {"totals_s": dict(worker_totals), "batches": worker_rows},
                "before_first_chunk": anchors, "all_cold_batches_before_first_successor_batch": len(cold)}
    return measured, (prefix, arrivals, scheduled, shape)


def warm_p1_replay(profiles, state, reference):
    prefix, arrivals, scheduled, shapes = state
    warm_prefix, _, _, warm_shapes = reference
    if not shapes or shapes != warm_shapes:
        raise ValueError("The warm and original first-request batches must match exactly")
    original_starts = [row["dispatch_ns"] for row in prefix]
    gaps = [right - left for left, right in pairwise(original_starts)]
    warm_starts = [row["dispatch_ns"] for row in warm_prefix[:len(shapes) + 1]]
    gaps[:len(shapes)] = [right - left for left, right in pairwise(warm_starts)]
    shift = {prefix[0]["batch"]: 0.0}
    dispatch = original_starts[0]
    for index, batch in enumerate(prefix[1:-1], 1):
        dispatch = max(dispatch + gaps[index - 1], max(arrivals[key]["arrival_ns"] for key, _, _ in batch["rows"]))
        shift[batch["batch"]] = (dispatch - original_starts[index]) / NS
    changes = {key: shift[req[-1][0]["batch"]] for key, req in scheduled.items()
               if profiles[key]["predecessor"] is None and all(row["batch"] in shift for row, _, _ in req)}
    costs = phase_times(profiles)
    for key, delta in changes.items():
        costs[key] = {**costs[key], "P_HTTP": costs[key]["P_HTTP"] + delta}
        if costs[key]["P_HTTP"] < 0:
            raise ValueError(f"Prefill replay made a negative stage in {key}")
    return {"roots_with_modeled_P_change": changes, "modeled": timeline(profiles, costs)}


def singleton_sensitivity(profiles, tails):
    changes = {row["terminal_request"]: 0.1 * row["singleton_s"] for row in tails.values()}
    estimates = {}
    for label, requests in (("last_only", {last_key(profiles)}), ("both_D_endpoints", changes.keys())):
        costs = {key: {**row, DECODE: row[DECODE] - changes[key] if key in requests else row[DECODE]}
                 for key, row in phase_times(profiles).items()}
        estimates[label] = timeline(profiles, costs)
    return estimates


def analyze(results):
    traces, profiles, totals, tails, p1, p1_states = {}, {}, {}, {}, {}, {}
    source = None
    transport = {}
    r39_p1_processes = set()
    for run, folder in ARCHIVES.items():
        directory = results / folder
        manifest = read_json(directory / "manifest.json")
        trace = directory / f"{run}-events.jsonl.gz"
        digest = sha256(trace)
        files = manifest.get("files", manifest.get("artifacts", {}))
        if digest != (files[trace.name]["sha256"] if files else manifest["trace_sha256"]):
            raise ValueError(f"Wrong trace digest for {run}")
        traces[run] = digest
        workload = manifest["workload"]
        spec = [{key: value for key, value in row.items() if key != "max_tokens"} for row in workload["request_sources"]]
        if source is not None and source != spec:
            raise ValueError(f"The cohorts in {run} differ beyond the generation budget")
        source = spec
        for case, raw in read_trace(trace).items():
            origin, = (row["monotonic_s"] for row in raw["client"] if row["type"] == "client_origin")
            name = f"{run}/{case}"
            rows = profiles[name] = requests_for(raw, workload["request_sources"], origin)
            check_observed(rows)
            totals[name] = timeline(rows, phase_times(rows))
            tails[name] = decode_tails(rows, raw["telemetry"], origin)
            if run in {"r28", "r33", "r39"}:
                cap = int(case.rsplit("cap", 1)[-1])
                transport[name] = routing_and_transport(rows, raw["telemetry"], origin, cap)
            if run in {"r33", "r39"}:
                p1[name], p1_states[name] = cold_p1(raw["p_native"], rows)
            if run == "r39":
                r39_p1_processes.update(event["pid"] for event in raw["p_native"]
                                        if event["instance"] == "P1" and event["type"] == "p_chunk_dispatch")

    cross, matched, sensitivities = {}, {}, {}
    for key in ("b1-cap100", "b2-cap100", "b1-cap200", "b2-cap200"):
        a, b = f"r28/{key}", f"r39/{key}"
        cross[key] = compare(profiles[a], profiles[b], totals[a], totals[b])
        cross[key]["matching_r39_rate_with_r28_output_in_model"] = decode_scale_to_target(
            profiles[a], totals[b]["throughput_tokens_s"])
    a = "r33/b1-cap100"
    batch_alignment = {}
    for key in ("b1-cap100", "b2-cap100"):
        b = f"r39/{key}"
        matched[key] = compare(profiles[a], profiles[b], totals[a], totals[b])
        if matched[key]["same_route_requests"] != len(profiles[a]) or matched[key]["same_output_lengths"] != len(profiles[a]):
            raise ValueError(f"Unmatched work in {key}")
        batch_alignment[b] = {}
        for anchor in ANCHORS:
            before = p1[a]["before_first_chunk"][anchor]
            after = p1[b]["before_first_chunk"][anchor]
            original_batches = p1_states[a][0][:before["batches"]]
            replay_batches = p1_states[b][0][:after["batches"]]
            same_batches = sum(sorted(left["rows"]) == sorted(right["rows"])
                               for left, right in zip(original_batches, replay_batches))
            batch_alignment[b][anchor] = {
                "batches": [before["batches"], after["batches"]],
                "same_batches_at_same_position": same_batches,
                "same_tokens_by_request": before["tokens_by_request"] == after["tokens_by_request"],
            }
    first_shape = p1_states[a][3]
    if any(state[3] != first_shape for state in p1_states.values()):
        raise ValueError("P1 first-request batches differ across the compared arms")
    for name in ("r28/b1-cap100", "r28/b2-cap100"):
        sensitivities[name] = {"observed": totals[name],
                               "ten_percent_faster_singleton_decode": singleton_sensitivity(profiles[name], tails[name])}
    for name in (a, "r39/b1-cap100", "r39/b2-cap100"):
        rows, state = profiles[name], p1_states[name]
        identity = warm_p1_replay(rows, state, state)["modeled"]
        if identity != totals[name]:
            raise ValueError(f"With original gaps, prefill replay must reproduce {name}")
        sensitivities[name] = {"observed": totals[name],
                               "P1_first_request_gaps_from_r39_b2_100": warm_p1_replay(
                                   rows, state, p1_states["r39/b2-cap100"]),
                               "ten_percent_faster_singleton_decode": singleton_sensitivity(rows, tails[name])}
    return {"source_trace_sha256": traces, "observed_runs": totals, "D_endpoint_progress": tails,
            "routing_and_transport": transport,
            "r28_to_r39_same_cap": cross, "r33_to_r39_matched_100": matched, "P1_cold_batches": p1,
            "P1_batch_alignment_to_r33_100": batch_alignment,
            "r39_distinct_P1_dispatch_processes": len(r39_p1_processes),
            "conditional_replays": sensitivities}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Directory containing the trace archives")
    parser.add_argument("output", type=Path, help="Path to the comparison JSON")
    args = parser.parse_args()
    write_json(args.output, analyze(args.results))


if __name__ == "__main__":
    main()
