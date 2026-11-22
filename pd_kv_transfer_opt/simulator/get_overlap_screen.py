"""Screen r69 cross-batch GET overlap using only the archived method trace."""

from __future__ import annotations

import argparse
from itertools import pairwise
from math import isclose
from pathlib import Path
from statistics import median_low

from pd_kv_transfer_opt.common import write_json
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS
from pd_kv_transfer_opt.simulator.freeinference_ab import calibration
from pd_kv_transfer_opt.simulator.get_criticality import (
    METHODS, against_baseline, join_loads, load_method_evidence, p_chunk_readies,
    p_reply_advances, prefill_chain, scenario, worker_loads,
)
from pd_kv_transfer_opt.simulator.rdma_replay import PORTS, drained_measurement, egress_epochs, load_native_archive
from pd_kv_transfer_opt.simulator.terminal_pressure import planned_recovery_start


FOCUS = ("P1/226", "P1/228")
TX_COUNTER = "mac_tx_total_oct_num"


def free_spans(begin, end, busy):
    for left, right in busy:
        if right <= begin:
            continue
        if left >= end:
            break
        if begin < left:
            yield begin, left
        begin = max(begin, right)
    if begin < end:
        yield begin, end


def port_evidence(samples, span, p_to_d_ns, cap):
    """Use the whole poll's bytes when testing identifiable slack in a subinterval."""
    begin, end = (t + p_to_d_ns for t in span)
    if begin == end:
        return []
    limit = cap / PORTS / 8  # bytes per nanosecond at the nominal readback
    result = []
    for points in zip(*(sample["ports"] for sample in samples), strict=True):
        covered, estimated_tx, remaining = 0, 0., 0.
        polls = []
        for before, after in pairwise(points):
            left, right = before["monotonic_ns"], after["monotonic_ns"]
            width = min(right, end) - max(left, begin)
            if width <= 0:
                continue
            tx_bytes = after["counters"][TX_COUNTER] - before["counters"][TX_COUNTER]
            if tx_bytes < 0 or before["port"] != after["port"]:
                raise ValueError("Invalid port counter sequence")
            covered += width
            estimated_tx += tx_bytes * width / (right - left)
            remaining += max(0., limit * width - tx_bytes)
            polls.append((right - left) / NS)
        if covered != end - begin:
            raise ValueError("D MAC polls did not cover the candidate window")
        result.append({"port": points[0]["port"], "raw_poll_lengths_s": polls,
                       "poll_mean_MAC_TX_Gbps_weighted_by_overlap": 8 * estimated_tx / covered,
                       "nominal_residual_bytes_identifiable_from_poll_totals": remaining})
    return result


def batches(events, requests, payload_bytes):
    p_events = [row for row in events if row["instance"] in ("P0", "P1")]
    if any(row.get("load_async") for row in p_events if row["type"] == "worker_load_begin"):
        raise ValueError("Expected synchronous native GET")
    chunks = prefill_chain(p_events, requests)
    loads = worker_loads(p_events)
    joined = join_loads(chunks, loads, payload_bytes)
    return p_events, chunks, loads, joined


def get_ranks(joined, loads, chunk):
    group = joined[chunk["id"]]
    if not group["unique_dispatch_join"]:
        raise ValueError("Uncertain scheduling join for a target or its predecessor")
    return loads[chunk["id"].split("/")[0], tuple(group["request_ids_in_GET"])]


def evaluate_batch(chunk, previous, target, prior, p0_reads, events, samples, cap, p_to_d_ns, origin_ns):
    def relative(t):
        return (t + p_to_d_ns - origin_ns) / NS

    first = [rank["reads"][0] for rank in target]
    first_bytes = sum(n for _, _, n in first)
    native = min(begin for begin, _, _ in first)
    prior_end = max(end for rank in prior for _, end, _ in rank["reads"])
    release = max(chunk["dispatch"], prior_end)
    if not release <= previous["ready"] <= native:
        raise ValueError("The recorded starts do not have the expected adjacent-batch order")
    rivals = [(begin, end) for begin, end in p0_reads if begin < native and end > release]

    def window(end):
        spans = list(free_spans(release, end, rivals))
        span = max(spans, key=lambda row: row[1] - row[0], default=(release, release))
        width = span[1] - span[0]
        budget = cap / 8 * width
        latest = max(rank["end"] for rank in target)
        shifted = max(rank["end"] - min(width, stop - start)
                      for rank, (start, stop, _) in zip(target, first, strict=True))
        # A deliberately favorable sensitivity for splitting and staging an equal share on every rank.
        fraction = min(1., budget / first_bytes)
        shifted_by_bytes = max(rank["end"] - (stop - start) * fraction
                               for rank, (start, stop, _) in zip(target, first, strict=True))
        return {"longest_gap_s": width / NS,
                "longest_gap_client_clock_s_approx": list(map(relative, span)),
                "all_gaps_client_clock_s_approx": [list(map(relative, gap)) for gap in spans],
                "max_wire_bytes_at_empty_nominal_cap": budget,
                "per_port_polls": port_evidence(samples, span, p_to_d_ns, cap),
                "first_call_overlap_credit_s": (latest - shifted) / NS,
                "first_call_equal_fraction_line_rate_credit_s": (latest - shifted_by_bytes) / NS}

    return {
        "chunk_id": chunk["id"], "predecessor_chunk_id": previous["id"],
        "scheduler_batch_id": chunk["batch"], "dispatch_client_clock_s_approx": relative(chunk["dispatch"]),
        "last_predecessor_GET_end_client_clock_s_approx": relative(prior_end),
        "predecessor_result_client_clock_s_approx": relative(previous["ready"]),
        "predecessor_GET_end_to_result_s": (previous["ready"] - prior_end) / NS,
        "native_first_GET_start_client_clock_s_approx": relative(native),
        "worker_load_enter_client_clock_s_approx": [relative(rank["start"]) for rank in target],
        "native_first_GET_start_range_client_clock_s_approx": [relative(native), relative(max(a for a, _, _ in first))],
        "dispatch_to_first_native_GET_s": (native - chunk["dispatch"]) / NS,
        "first_call_rank_pid_bytes_duration_s": [[rank["worker"], n, (b - a) / NS]
                                                  for rank, (a, b, n) in zip(target, first, strict=True)],
        "first_call_total_descriptor_bytes": first_bytes,
        "total_descriptor_bytes": sum(n for rank in target for _, _, n in rank["reads"]),
        "P0_competing_API_intervals_client_clock_s_approx": [list(map(relative, rival)) for rival in rivals],
        "before_predecessor_result": window(previous["ready"]),
        "before_native_GET": window(native),
        "positive_token_lookups_before_dispatch": {
            row["request_id"]: {"local_tokens": row["local_tokens"], "external_tokens": row["external_tokens"]}
            for row in events if row["type"] == "pool_match_tokens" and row["instance"] == "P1"
            and row["request_id"] in chunk["ids"] and row["monotonic_ns"] <= chunk["dispatch"]
            and row["external_tokens"] > 0
        },
    }


def replay_overlap(chunks, requests, fits, targets, origin_s):
    since = planned_recovery_start(requests)
    baseline = scenario(requests, {}, fits, since)
    measured = drained_measurement(requests)
    reference = p_chunk_readies(chunks, {}, propagate=True, hold_dispatch=True)
    if any(reference[c["id"]] != c["ready"] for sequence in chunks.values() for c in sequence):
        raise ValueError("Zero-change replay did not reproduce the measured P dispatch chain")
    if not measured["all_natural_EOS"] or not isclose(
            baseline["fixed_observed_downstream"]["duration_s"], measured["duration_s"], abs_tol=1e-6):
        raise ValueError("Zero-change replay did not reproduce the measured natural-EOS run")

    def run(cuts):
        after = p_chunk_readies(chunks, cuts, propagate=True, hold_dispatch=True)
        replies = p_reply_advances(chunks, cuts, propagate=True, hold_dispatch=True)
        timing = against_baseline(scenario(requests, replies, fits, since), baseline)
        return {"GET_wait_credit_s": cuts, "throughput": timing,
                "P_batch_result_advances_s": {k: (v - after[k]) / NS for k, v in reference.items() if v != after[k]},
                "P_request_reply_advances_s": {k: gain for k, gain in replies.items() if gain},
                "reference_tail_P_reply_advance_s": replies[baseline["fixed_observed_downstream"]["tail"]]}

    previews = {}
    for boundary in ("before_predecessor_result", "before_native_GET"):
        cuts = {key: row[boundary]["first_call_overlap_credit_s"] for key, row in targets.items()}
        previews[boundary] = run(cuts)
        if boundary == "before_predecessor_result":
            previews["individual_batches_before_predecessor_result"] = {key: run({key: cut})
                                                                         for key, cut in cuts.items()}
            previews["ideal_equal_fraction_line_rate_before_predecessor_result"] = run({
                key: row[boundary]["first_call_equal_fraction_line_rate_credit_s"] for key, row in targets.items()
            })
    return {"hardware_baseline": measured, "baseline_replay": baseline, "overlap_sensitivity": previews,
            "client_origin_monotonic_s": origin_s}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fits = {name: meta["decode_fit"] for name, (_, meta) in calibration(args.results).items()}
    path = args.results / METHODS
    manifest, digest, arms = load_native_archive(path)
    events, totals, method_digest = load_method_evidence(path, manifest)
    if manifest["settings"]["P_pool_GiB"] != 0:
        raise ValueError("Expected data objects in D's Store segment")
    profiles = {case: batches(events[case], requests, totals[case]["P_GET_descriptor_bytes"])
                for case, (_, requests, _) in arms.items()}
    low_joined = profiles["b1-cap100"][3]
    low_groups = {key: set(low_joined[key]["request_ids_in_GET"]) for key in FOCUS}
    result = {}
    for case, (streams, requests, origin) in arms.items():
        cap = int(case.rsplit("-cap", 1)[1])
        p_events, chunks, loads, joined = profiles[case]
        groups = {}
        for key, group in joined.items():
            restored = set(group["request_ids_in_GET"])
            groups[key] = {focus: sorted(ids & restored) for focus, ids in low_groups.items() if ids & restored}
        targets = FOCUS if cap == 100 else tuple(sorted(key for key, members in groups.items() if members))
        if not targets or any(not key.startswith("P1/") for key in targets):
            raise ValueError("Unexpected route for the 200 Gbps overlap cohort")
        origin_ns = round(origin * NS)
        anchor, = (row["anchor"] for row in streams["client"] if row["type"] == "client_origin")
        p_to_d_ns = (median_low(row["unix_ns"] - row["monotonic_ns"] for row in p_events)
                     - (anchor["unix_ns"] - anchor["monotonic_ns"]))
        p0 = sorted((a, b) for (role, _), ranks in loads.items() if role == "P0"
                    for rank in ranks for a, b, _ in rank["reads"])
        evidence = {}
        for previous, chunk in pairwise(chunks["P1"]):
            key = chunk["id"]
            if key in targets:
                evidence[key] = evaluate_batch(chunk, previous, get_ranks(joined, loads, chunk),
                                               get_ranks(joined, loads, previous), p0, p_events,
                                               streams["telemetry"], cap, p_to_d_ns, origin_ns)
                evidence[key]["matched_100_Gbps_GET_members"] = groups[key]
        if set(evidence) != set(targets):
            raise ValueError("Missing a target batch")
        master = [row["master"]["master_evicted_key_count"] for row in streams["telemetry"] if "master" in row]
        result[case] = {
            "requests": len(requests),
            "P_device_forward_events": sum(row["type"] in {"p_chunk_device", "p_chunk_rank_forward"}
                                           for row in p_events),
            "P_GET_descriptor_bytes_unchanged_in_sensitivity": totals[case]["P_GET_descriptor_bytes"],
            "measured_D_MAC_TX_GB_in_P_HTTP_span": sum((b - a) * sum(rates) / 8
                                                       for a, b, rates in egress_epochs(arms[case])),
            "master_evicted_key_count_at_snapshots": master, "batches": evidence,
            **replay_overlap(chunks, requests, fits, evidence, origin),
        }

    report = {
        "archive": METHODS, "method_trace_sha256": method_digest, "request_trace_sha256": digest, "cases": result,
        "safety_evidence": "The trace contains batch dispatch and rank-level Store API timing, not destination "
                           "page generations, zeroing fences, pinned sources, registered staging space or leaf port routes.",
        "network_evidence": "The candidate windows exclude observed P0 GETs after the predecessor's last GET. "
                            "Predecessor device compute and link idleness are unobserved. MAC polls are positioned "
                            "by a wall-clock offset and cannot certify instantaneous per-port credit.",
        "intervention": "Sensitivity only. Each cap retains its own planned offers, request dependencies, EOS, routes, "
                        "dispatch timestamps, batch contents and descriptor work. The first GET on each rank receives a "
                        "time credit clipped to its observed duration while all earlier work keeps its original cost. "
                        "No transfer is rescheduled and there is no calibrated observation of interference or staging cost.",
    }
    write_json(args.output, report)


if __name__ == "__main__":
    main()
