"""Replay r69's late GETs with eight shared D egress budgets and rank-local copy."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from gzip import open as gzip_open
from heapq import heappop, heappush
from itertools import pairwise
import json
from math import inf, isclose
from pathlib import Path
from statistics import median_low

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import ATTENTION_OBJECT_BYTES
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS
from pd_kv_transfer_opt.simulator.freeinference_ab import calibration
from pd_kv_transfer_opt.simulator.get_criticality import METHODS, against_baseline, load_method_evidence, p_reply_advances, scenario
from pd_kv_transfer_opt.simulator.get_overlap_screen import batches, port_evidence
from pd_kv_transfer_opt.simulator.get_restore_audit import shareable_blocks, source_prefixes
from pd_kv_transfer_opt.simulator.rdma_replay import PORTS, drained_measurement, load_native_archive
from pd_kv_transfer_opt.simulator.terminal_pressure import planned_recovery_start


TAIL_GET = "P0/177"
REQUESTS = ("winning-avg-corewars:117", "caffe-cifar-10:74", "regex-chess:43")
TARGETS = {f"caffe-cifar-10:{i}" for i in (82, 79, 74, 68, 58)}
PORT_SHAPES = ("pooled_MAC", "separate_MAC_windows")
EPS = 1e-9


def inspect_key_evidence(path, operation):
    """Inventory the actual retained GET or PUT fields without exporting row contents."""
    count, fields = 0, set()
    with gzip_open(path, "rt", encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            if row["type"] == "store_end" and row["operation"] == operation:
                count += 1
                fields.update((field, type(value).__name__) for field, value in row.items())
    if not count:
        raise ValueError("Expected archived Store calls")
    return {"path_basename": path.name, "sha256": sha256(path), "calls": count,
            "fields_and_value_types": sorted(fields)}


def pick_tapes(chunks, loads, joined, audit, families, tips):
    selected = (TAIL_GET, *(cid for cid, row in joined.items() if TARGETS.intersection(row["request_ids_in_GET"])))
    members = {key for cid in selected[1:] for key in joined[cid]["request_ids_in_GET"] if key in TARGETS}
    if members != TARGETS or len(set(selected)) != len(selected):
        raise ValueError("Some trace requests were not resolved to native GETs")
    steps = {c["id"]: c for seq in chunks.values() for c in seq if c["id"] in selected}
    ranks_for = {cid: loads[step["id"].split("/")[0], tuple(joined[cid]["request_ids_in_GET"])]
                 for cid, step in steps.items()}
    origin_ns = min(step["start"] for step in steps.values())
    flows, tapes, chain, chosen_plan = {}, {}, defaultdict(list), {}
    for cid, step in steps.items():
        original_plan = audit["batches"][cid]["read_plan"]
        observed = {row["request"]: row["attention_span_in_blocks"] for row in original_plan}
        matched = [row for row in original_plan if row["request"] in TARGETS]
        chosen_plan[cid] = []
        bytes_per_rank = {}
        for i, row in enumerate(matched):
            request = row["request"]
            count = shareable_blocks(request, [prior["request"] for prior in matched[:i]], observed, families, tips)
            bytes_per_rank[request] = count * ATTENTION_OBJECT_BYTES
            chosen_plan[cid].append({**row, "same_rank_shareable_attention_bytes": bytes_per_rank[request] * len(ranks_for[cid]),
                                     "repeat_attention_blocks_per_rank": count})
        roles = step["id"].split("/")[0]
        chain[roles].append(cid)
        rank_tapes = {}
        last_worker = max(rank["end"] for rank in ranks_for[cid])
        for rank in ranks_for[cid]:
            worker = rank["worker"]
            keys = []
            for i, ((begin, end, total), row) in enumerate(zip(rank["reads"], original_plan, strict=True)):
                if total * len(ranks_for[cid]) != row["native_bytes_all_ranks"] or begin < step["start"]:
                    raise ValueError("Restore audit differs from the native worker tape")
                identity = cid, worker, i
                duplicate = bytes_per_rank.get(row["request"], 0)
                if not 0 <= duplicate < total:
                    raise ValueError("Some native read bytes must remain for each request")
                flows[identity] = {"start": (begin - origin_ns) / NS, "end": (end - origin_ns) / NS, "bytes": total,
                                   "potential_saving": duplicate, "request": row["request"]}
                keys.append(identity)
            rank_tapes[worker] = {"keys": keys, "post": (rank["end"] - rank["reads"][-1][1]) / NS}
        tapes[cid] = {"start": (step["start"] - origin_ns) / NS, "dispatch": (step["dispatch"] - origin_ns) / NS,
                      "ready": (step["ready"] - origin_ns) / NS, "post": (step["ready"] - last_worker) / NS,
                      "ranks": rank_tapes}
    if any(int(b.split("/")[1]) != int(a.split("/")[1]) + 1 for seq in chain.values() for a, b in pairwise(seq)):
        raise ValueError("Intervening P stages must be accounted for before linking these batches")
    chosen = {(joined[cid]["P"], tuple(joined[cid]["request_ids_in_GET"])) for cid in selected}
    other_calls = [(a, b) for group, ranks in loads.items() if group not in chosen
                   for rank in ranks for a, b, _ in rank["reads"]]
    return flows, tapes, chain, origin_ns, other_calls, chosen_plan


def mac_weights(samples, cap, p_to_d_ns, origin_ns, flows, shape):
    def measured(active):
        a = round(min(row["start"] for row in active) * NS) + origin_ns
        b = round(max(row["end"] for row in active) * NS) + origin_ns
        rates = [point["poll_mean_MAC_TX_Gbps_weighted_by_overlap"]
                 for point in port_evidence(samples, (a, b), p_to_d_ns, cap)]
        if len(rates) != PORTS or not sum(rates):
            raise ValueError("MAC sampling did not cover the replay")
        return [rate / sum(rates) for rate in rates]

    if shape == "pooled_MAC":
        common = measured(list(flows.values()))
        return {role: common for role in ("P0", "P1")}
    return {role: measured([row for key, row in flows.items() if key[0].startswith(role)])
            for role in ("P0", "P1")}


def wire_profile(flows, weights, cap):
    """Anchor native service in API intervals and fit physically feasible port shares."""
    ceiling = (cap * NS) / (8 * PORTS)
    edges = sorted({t for row in flows.values() for t in (row["start"], row["end"])})
    base = {key: row["bytes"] / (row["end"] - row["start"]) for key, row in flows.items()}
    active = [[key for key, row in flows.items() if row["start"] <= a < row["end"]] for a in edges[:-1]]
    # Preserve as much measured skew as each nominal port ceiling permits with these API clocks.
    mix = 1.
    for keys in active:
        flat = sum(base[key] for key in keys) / PORTS
        if flat > ceiling * (1 + EPS):
            raise ValueError("Uniform reconstructed GET service exceeds the nominal cap")
        for p in range(PORTS):
            skewed = sum(base[key] * weights[key[0].split("/")[0]][p] for key in keys)
            if skewed > flat:
                mix = min(mix, max(0., (ceiling - flat) / (skewed - flat)))
    fit = {role: tuple(1 / PORTS + mix * (n - 1 / PORTS) for n in row) for role, row in weights.items()}
    rates = {key: tuple(base[key] * share for share in fit[key[0].split("/")[0]]) for key in flows}
    available = [tuple(sum(rates[key][p] for key in keys) for p in range(PORTS)) for keys in active]
    fallback = tuple(max(row[p] for row in available) for p in range(PORTS))
    return edges, available, fallback, rates, fit, mix


def replay_wire(flows, tapes, chain, profile, *, saved_fraction, local_GBps, reclaim_limit, copy_placement):
    edges, available, fallback, rates, fit, _ = profile
    pending, active, started, finished = [], {}, {}, {}
    rank_ends, stage_begin, batch_ends = defaultdict(list), {}, {}
    p_bytes = [0.] * PORTS
    after = {a: b for seq in chain.values() for a, b in pairwise(seq)}
    removed = {key: row["potential_saving"] * saved_fraction for key, row in flows.items()}
    copied_s = {key: n / (local_GBps * NS) for key, n in removed.items()}

    def open_batch(cid, predecessor=None):
        stage = tapes[cid]
        begin = stage["start"] if predecessor is None else max(stage["dispatch"], predecessor)
        stage_begin[cid] = begin
        for r in stage["ranks"].values():
            first = r["keys"][0]
            heappush(pending, (begin + flows[first]["start"] - stage["start"], first))

    for seq in chain.values():
        open_batch(seq[0])
    t = pending[0][0]
    while pending or active:
        if not active:
            t = max(t, pending[0][0])
        while pending and pending[0][0] <= t + EPS:
            ready, key = heappop(pending)
            started[key] = ready
            port_shares = fit[key[0].split("/")[0]]
            active[key] = [(flows[key]["bytes"] - removed[key]) * share for share in port_shares]
        upcoming = pending[0][0] if pending else inf
        region = bisect_right(edges, t + EPS)
        limit = available[region - 1] if 0 < region < len(edges) and any(available[region - 1]) else fallback
        if region < len(edges):
            upcoming = min(upcoming, edges[region])
        service = {key: [0.] * PORTS for key in active}
        for p in range(PORTS):
            on_port = [key for key, remaining in active.items() if remaining[p] > .05]
            if not on_port:
                continue
            factor = min(reclaim_limit, limit[p] / sum(rates[key][p] for key in on_port))
            for key in on_port:
                speed = rates[key][p] * factor
                service[key][p] = speed
                upcoming = min(upcoming, t + active[key][p] / speed)
        if upcoming <= t or upcoming == inf:
            raise ValueError("The finite per-port replay stopped progressing")
        progress = upcoming - t
        for key, remaining in active.items():
            for p, speed in enumerate(service[key]):
                sent = min(remaining[p], speed * progress)
                remaining[p] -= sent
                p_bytes[p] += sent
        t = upcoming
        for key in [key for key, remaining in active.items() if max(remaining) <= .05]:
            finished[key] = t
            del active[key]
            cid, worker, index = key
            tape = tapes[cid]["ranks"][worker]
            copy_inline = copied_s[key] if copy_placement == "inline" else 0.
            if index + 1 < len(tape["keys"]):
                next_key = tape["keys"][index + 1]
                gap = flows[next_key]["start"] - flows[key]["end"]
                heappush(pending, (t + gap + copy_inline, next_key))
            else:
                copy_tail = copy_inline if copy_placement == "inline" else sum(copied_s[k] for k in tape["keys"])
                rank_ends[cid].append(t + tape["post"] + copy_tail)
                if len(rank_ends[cid]) == len(tapes[cid]["ranks"]):
                    batch_ends[cid] = max(rank_ends[cid]) + tapes[cid]["post"]
                    if cid in after:
                        open_batch(after[cid], batch_ends[cid])
    if not isclose(sum(p_bytes), sum(row["bytes"] - removed[key] for key, row in flows.items()), abs_tol=8.):
        raise ValueError("The per-port replay did not conserve bytes")
    return {"GET_start": started, "GET_end": finished, "stage_begin": stage_begin, "batch_end": batch_ends,
            "ports_bytes": p_bytes, "bytes_saved": sum(removed.values())}


def request_timing(flows, start, end, origin_s):
    grouped = defaultdict(list)
    for key, row in flows.items():
        grouped[key[0], row["request"]].append(key)
    result = defaultdict(dict)
    for (cid, request), ids in grouped.items():
        result[cid][request] = [min(start[key] for key in ids) + origin_s, max(end[key] for key in ids) + origin_s]
    return result


def compare_arm(arm, events, total, reference, families, tips, fits, cap):
    streams, requests, d_origin = arm
    p_events, chunks, loads, joined = batches(events, requests, total)
    flows, tapes, chain, p_origin, other_calls, plan = pick_tapes(chunks, loads, joined, reference, families, tips)
    first_offer = d_origin * NS
    clock, = (r["anchor"] for r in streams["client"] if r["type"] == "client_origin")
    p_to_d = median_low(r["unix_ns"] - r["monotonic_ns"] for r in p_events) - clock["unix_ns"] + clock["monotonic_ns"]
    origin_s = (p_origin + p_to_d - first_offer) / NS
    since = planned_recovery_start(requests)
    baseline = scenario(requests, {}, fits, since, track=REQUESTS)
    measured = drained_measurement(requests)
    if not measured["all_natural_EOS"] or not isclose(
            baseline["fixed_observed_downstream"]["duration_s"], measured["duration_s"], abs_tol=1e-6):
        raise ValueError("Zero-change replay did not reproduce the observed natural-EOS run")

    def record(profile, fraction, copy, reclaim, placement):
        replay = replay_wire(flows, tapes, chain, profile, saved_fraction=fraction, local_GBps=copy,
                             reclaim_limit=reclaim, copy_placement=placement)
        for key, begin in replay["GET_start"].items():
            a = round(begin * NS) + p_origin
            b = round(replay["GET_end"][key] * NS) + p_origin
            if any(left < b and a < right for left, right in other_calls):
                raise ValueError("A displaced read overlapped another GET outside the modeled resource scope")
        if not fraction:
            max_error = max(abs(replay["GET_end"][key] - row["end"]) for key, row in flows.items())
            if max_error > 1e-6 or any(abs(replay["batch_end"][cid] - tape["ready"]) > 1e-6 for cid, tape in tapes.items()):
                raise ValueError("The port replay must recover its observed GET and batch clocks with no intervention")
            return {"max_native_GET_calibration_error_us": max_error * 1e6, "ports_bytes": replay["ports_bytes"]}
        cuts = {cid: (tape["ready"] - tape["start"]) - (replay["batch_end"][cid] - replay["stage_begin"][cid])
                for cid, tape in tapes.items()}
        advance = p_reply_advances(chunks, cuts, propagate=True, hold_dispatch=True)
        result = against_baseline(scenario(requests, advance, fits, since, track=REQUESTS), baseline)
        for name, row in result.items():
            row["P0_tail_request_finishes_earlier_s"] = (baseline[name]["tracked_requests"][REQUESTS[0]]["end_s"]
                                                         - row["tracked_requests"][REQUESTS[0]]["end_s"])
        ranks_177 = [key for key in flows if key[0] == TAIL_GET]
        return {"modeled_bytes_saved": replay["bytes_saved"], "modeled_D_port_bytes": replay["ports_bytes"],
                "request_GET_envelope_s_from_origin": request_timing(flows, replay["GET_start"], replay["GET_end"], origin_s),
                "GET_cut_in_batch_service_s": cuts, "P0_177_last_GET_end_advance_s":
                    max(flows[key]["end"] for key in ranks_177) - max(replay["GET_end"][key] for key in ranks_177),
                "P0_177_last_GET_end_s_from_origin": origin_s + max(replay["GET_end"][key] for key in ranks_177),
                "throughput": result}

    conditions = []
    for shape in PORT_SHAPES:
        weights = mac_weights(streams["telemetry"], cap, p_to_d, p_origin, flows, shape)
        profile = wire_profile(flows, weights, cap)
        original = record(profile, 0., 10., 1., "inline")
        configurations = [(1., 10., r, "inline") for r in (1., 1.5, inf)]
        configurations += [(1., c, 1.5, "inline") for c in (5., 20.)]
        configurations += [(.5, 10., 1.5, "inline"), (1., 10., 1.5, "before_forward")]
        for fraction, copy, reclaim, placement in configurations:
            conditions.append({"port_footprint_model": shape, "MAC_tx_fractions_nominal": weights,
                               "fitted_flow_port_fractions": profile[-2], "fit_MAC_skew_fraction": profile[-1],
                               "baseline": original, "candidate_fraction_realized": fraction,
                               "local_copy_GBps_per_rank": copy, "flow_rate_ceiling_multiple": reclaim if reclaim < inf else None,
                               "copy_placement": placement, **record(profile, fraction, copy, reclaim, placement)})
    return {"original": measured, "baseline_end_to_end": baseline, "case_batches": list(tapes),
            "selected_request_reads": {cid: reads for cid, reads in plan.items() if reads},
            "native_request_GET_envelope_s_from_origin": request_timing(
                flows, {key: row["start"] for key, row in flows.items()}, {key: row["end"] for key, row in flows.items()}, origin_s),
            "original_P0_177_last_GET_end_s_from_origin":
                origin_s + max(row["end"] for key, row in flows.items() if key[0] == TAIL_GET),
            "P0_177_original_GET_envelope_s": (max(row["end"] for key, row in flows.items() if key[0] == TAIL_GET)
                                               - min(row["start"] for key, row in flows.items() if key[0] == TAIL_GET)),
            "scenarios": conditions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--restore-audit", type=Path, required=True)
    parser.add_argument("--raw-p", type=Path, required=True)
    parser.add_argument("--raw-d", type=Path, required=True)
    parser.add_argument("--terminal-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.results / METHODS
    manifest, request_hash, arms = load_native_archive(directory)
    raw_events, counts, method_hash = load_method_evidence(directory, manifest)
    audit = read_json(args.restore_audit)
    raw_p = inspect_key_evidence(args.raw_p, "batch_get_into_multi_buffers")
    raw_d = inspect_key_evidence(args.raw_d, "batch_put_from_multi_buffers")
    ledger_hash, families, tips = source_prefixes(args.terminal_ledger, manifest["workload"])
    if (request_hash, method_hash, raw_p["sha256"], ledger_hash) != (
            audit["request_trace_sha256"], audit["method_trace_sha256"], audit["raw_P_trace_sha256"],
            audit["terminal_ledger_sha256"]):
        raise ValueError("Restore audit is not tied to the same native trace and Terminal-Bench ledger")
    fits = {name: meta["decode_fit"] for name, (_, meta) in calibration(args.results).items()}
    result = {case: compare_arm(arm, raw_events[case], counts[case]["P_GET_descriptor_bytes"], audit["cases"][case],
                                families, tips, fits, int(case.rsplit("-cap", 1)[1])) for case, arm in arms.items()}
    write_json(args.output, {
        "archive": METHODS, "method_trace_sha256": method_hash, "request_trace_sha256": request_hash,
        "restore_audit_sha256": sha256(args.restore_audit), "terminal_ledger_sha256": ledger_hash,
        "raw_P_GET_schema": raw_p, "raw_D_PUT_schema": raw_d,
        "physical_key_equality_observed": False, "physical_destination_layout_observed": False,
        "scope": "The five r69 P1/228 caffe-cifar-10 requests only. Per-rank Attention candidates within each original "
                 "batch, original private state objects, fixed offers, EOS, routes, group membership and dispatch. "
                 "No cross-rank sharing. D decode is replayed after the P resource change.",
        "wire_model": "Use native API intervals to anchor a service profile and infer bytes per port from MAC proportions, "
                      "flattening enough to stay below each nominal cap. Reallocate each port's reconstructed native "
                      "service among active calls while holding other observed costs; use the fitted port peak only "
                      "in native idle periods. Protocol traffic and other resource waits are not identifiable below "
                      "the polling cadence. Per-flow headroom and per-call port shares are assumptions, not packet evidence.",
        "cases": result,
    })


if __name__ == "__main__":
    main()
