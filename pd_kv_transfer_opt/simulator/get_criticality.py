"""Replay measured synchronous GETs and the downstream natural-EOS work."""

from __future__ import annotations

import argparse
from collections import defaultdict
from gzip import open as gzip_open
import json
from math import isclose
from pathlib import Path

from pd_kv_transfer_opt.common import percentile, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import get_batches, interval_union
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, phase_times, replay
from pd_kv_transfer_opt.simulator.decode_supply import simulate_decode
from pd_kv_transfer_opt.simulator.freeinference_ab import calibration
from pd_kv_transfer_opt.simulator.rdma_audit import artifact
from pd_kv_transfer_opt.simulator.rdma_replay import drained_measurement, egress_epochs, load_native_archive
from pd_kv_transfer_opt.simulator.terminal_pressure import planned_recovery_start


R69 = "2026-11-22-r69-terminal-source-restore-short-guard"
R78 = "2026-11-22-r78-terminal-source-restore-stable-tail"
METHODS = R69 + "-methods"


def worker_loads(events):
    active, batches = {}, defaultdict(list)
    for row in sorted(events, key=lambda item: item["monotonic_ns"]):
        rank = row["instance"], row["pid"]
        kind = row["type"]
        if kind == "worker_load_begin":
            if rank in active:
                raise ValueError("Nested worker load")
            active[rank] = (row["monotonic_ns"], tuple(sorted(row["request_ids"])), [])
        elif kind == "store_end" and row["operation"] == "batch_get_into_multi_buffers":
            if row["negative_returns"] or rank not in active:
                raise ValueError("Unjoined or failed synchronous batch GET")
            end = row["monotonic_ns"]
            active[rank][2].append((end - row["elapsed_ns"], end, row["payload_bytes"]))
        elif kind == "worker_load_end":
            start, members, reads = active.pop(rank)
            end = row["monotonic_ns"]
            if members != tuple(sorted(row["request_ids"])) or any(a < start or b > end for a, b, _ in reads):
                raise ValueError("Worker events escaped their same-rank load interval")
            batches[rank[0], members].append({"worker": rank[1], "start": start, "end": end, "reads": reads})
    if active or any(len({r["worker"] for r in ranks}) != 4 for ranks in batches.values()):
        raise ValueError("Missing worker rank or terminal event")
    return batches


def prefill_chain(events, requests):
    arrivals, raw_chunks, errors = get_batches(events)
    if errors or arrivals.keys() != requests.keys():
        raise ValueError("Could not match scheduler chunks and request admissions")
    chunks = defaultdict(list)
    for role in ("P0", "P1"):
        steps = sorted((step for step in raw_chunks if step["instance"] == role), key=lambda step: step["ready_ns"])
        last = 0
        for i, step in enumerate(steps):
            ids = tuple(row[0] for row in step["rows"])
            admission = max(arrivals[key]["arrival_ns"] for key in ids)
            dispatch, ready = step["dispatch_ns"], step["ready_ns"]
            start = max(dispatch, last)
            gap = start - max(last, admission)
            if gap < 0 or ready < start or any(requests[key]["P"] != role for key in ids):
                raise ValueError("Invalid worker progression or moved P route")
            chunks[role].append({"id": f"{role}/{i + 1}", "batch": step["batch"], "ids": ids,
                                 "dispatch": dispatch, "start": start, "ready": ready,
                                 "admission": admission, "gap": gap})
            last = ready
    return chunks


def join_loads(chunks, loads, exact_total):
    candidates = {}
    observed = 0
    for (role, ids), ranks in loads.items():
        first = min(r["start"] for r in ranks)
        last = max(r["end"] for r in ranks)
        byte_count = sum(n for r in ranks for _, _, n in r["reads"])
        observed += byte_count
        if not byte_count:
            continue
        possible = [c for c in chunks[role] if c["dispatch"] <= first and last <= c["ready"] and set(ids) <= set(c["ids"])]
        exposed = []
        for chunk in possible:
            timings = [interval_union((max(a, chunk["start"]), b) for a, b, _ in r["reads"] if b > chunk["start"])
                       for r in ranks]
            if any(timings):
                exposed.append((chunk, timings))
        if len(exposed) != 1 or exposed[0][0]["id"] in candidates:
            raise ValueError("GET must have a unique non-overlapped candidate in the worker frontier")
        chunk, times = exposed[0]
        candidates[chunk["id"]] = {
            "chunk": chunk["id"], "P": role, "request_ids_in_GET": list(ids), "request_ids_in_chunk": list(chunk["ids"]),
            "payload_GB": byte_count / 1e9, "scheduler_matches": [row["id"] for row in possible],
            "unique_dispatch_join": len(possible) == 1, "dispatch_to_ready_s": (chunk["ready"] - chunk["dispatch"]) / NS,
            "rank_end_ns": [row["end"] for row in ranks], "uncovered_rank_GET_ns": times,
            "slowest_rank_GET_s": max(sum(b - a for a, b, _ in r["reads"]) for r in ranks) / NS,
            "last_rank_load_end_to_ready_s": (chunk["ready"] - last) / NS,
        }
    if observed != exact_total:
        raise ValueError("Joined GET payload disagrees with the archived descriptor total")
    return candidates


def clip_seconds(group, fraction):
    rank = group["rank_end_ns"]
    duration = group["uncovered_rank_GET_ns"]
    return (max(rank) - max(t - fraction * dt for t, dt in zip(rank, duration, strict=True))) / NS


def p_chunk_readies(chunks, cuts, *, propagate, hold_dispatch=False):
    completed = {}
    for sequence in chunks.values():
        previous = 0
        for step in sequence:
            cut = cuts.get(step["id"], 0.) * NS
            service = step["ready"] - step["start"]
            if cut > service + 1:
                raise ValueError("A cut exceeds its non-overlapped worker service")
            updated = step["ready"] - cut
            if propagate:
                if hold_dispatch:
                    start = max(previous, step["dispatch"])
                else:
                    start = max(previous, step["admission"]) + step["gap"]
                updated = start + service - cut
                previous = updated
            completed[step["id"]] = updated
    return completed


def p_reply_advances(chunks, cuts, *, propagate, hold_dispatch=False):
    completed = p_chunk_readies(chunks, cuts, propagate=propagate, hold_dispatch=hold_dispatch)
    actual, shifted = {}, {}
    for sequence in chunks.values():
        for step in sequence:
            for key in step["ids"]:
                actual[key] = max(actual.get(key, 0), step["ready"])
                shifted[key] = max(shifted.get(key, 0), completed[step["id"]])
    return {key: (t - shifted[key]) / NS for key, t in actual.items()}


def scenario(requests, advance, fits, since, *, track=()):
    durations = phase_times(requests)
    p_http = {key: row["P_HTTP"] - advance.get(key, 0) for key, row in durations.items()}
    if any(seconds < -1e-8 for seconds in p_http.values()):
        raise ValueError("Earlier terminal P chunk would precede the request's observed P start")
    phases = {key: {**cost, "P_HTTP": p_http[key]} for key, cost in durations.items()}
    ready, done, chain = replay(requests, phases)
    fixed = {key: {"first_s": done[key] - cost["first_to_end"], "end_s": done[key],
                   "P_end_s": ready[key] + cost["ready_to_P"] + cost["P_HTTP"]}
             for key, cost in phases.items()}
    first_offer = min(row["offered_s"] for row in requests.values())
    q = sum(row["output_tokens"] for row in requests.values())

    def metrics(rows, finish, tail):
        continuations = [rows[key] for key in requests if requests[key]["predecessor"] is not None]
        result = {"duration_s": finish - first_offer, "output_tokens": q,
                "throughput_tokens_s": q / (finish - first_offer), "tail": tail,
                "last_P_reply_s": max(row["P_end_s"] for row in rows.values()),
                "successor_first_tokens_after_s": {str(s): sum(row["first_s"] <= since + s for row in continuations)
                                                    for s in (5, 10, 20)},
                "mean_D_running_first_10s_by_role": {
                    role: sum(max(0., min(point["end_s"], since + 10) - max(point["first_s"], since))
                              for key, point in rows.items() if requests[key]["D"] == role) / 10
                    for role in ("D0", "D1")},
                "sum_successor_first_times_s": sum(row["first_s"] for row in continuations)}
        if track:
            result["tracked_requests"] = {key: {field: rows[key][field] for field in ("P_end_s", "first_s", "end_s")}
                                          for key in track}
        return result

    result = {"fixed_observed_downstream": metrics(fixed, done[chain[-1]], chain[-1])}
    for name, fit in fits.items():
        simulated = simulate_decode(requests, fit, prefill_http_s=p_http, include_requests=True)
        result[name] = metrics(simulated["per_request"], simulated["finish_s"], simulated["tail_id"])
    return result


def against_baseline(value, base):
    for name, arm in value.items():
        start = base[name]
        arm.update(saved_finish_s=start["duration_s"] - arm["duration_s"],
                   relative_throughput_gain=arm["throughput_tokens_s"] / start["throughput_tokens_s"] - 1,
                   earlier_successor_first_sum_s=start["sum_successor_first_times_s"] - arm["sum_successor_first_times_s"],
                   additional_successor_first_tokens_after_s={s: q - start["successor_first_tokens_after_s"][s]
                                                              for s, q in arm["successor_first_tokens_after_s"].items()})
    return value


def d_pull_join(events, requests, origin_s):
    ready = defaultdict(list)
    for row in events:
        if row["type"] == "native_handle_end":
            key = row["remote_request_id"]
            if key not in requests or requests[key]["D"] != row["instance"]:
                raise ValueError("Could not match all rank callbacks to the original D instance")
            ready[key].append(row["monotonic_ns"] / NS - origin_s)
    deltas = {key: requests[key]["first_s"] - max(times) for key, times in ready.items() if len(times) == 4}
    if any(d < 0 for d in deltas.values()):
        raise ValueError("D Pull callback completion follows its client's first token")
    return {"requests_with_all_four_rank_callbacks": len(deltas),
            "last_rank_P_to_D_callback_to_first_token_s_p50_p95": [percentile(list(deltas.values()), q) for q in (.5, .95)]}


def method_replay(events, requests, fits, origin, exact_total):
    p_events = [row for row in events if row["instance"].startswith("P")]
    chunks = prefill_chain(p_events, requests)
    joined = join_loads(chunks, worker_loads(p_events), exact_total)
    since = planned_recovery_start(requests)
    zero = p_reply_advances(chunks, {}, propagate=True)
    if zero.keys() != requests.keys() or any(dt for dt in zero.values()):
        raise ValueError("Frozen P chunk chain does not reproduce the original native schedule")
    base = scenario(requests, {}, fits, since)
    observed = drained_measurement(requests)
    if not isclose(base["fixed_observed_downstream"]["duration_s"], observed["duration_s"], abs_tol=1e-6):
        raise ValueError("End-to-end fixed stage replay did not reproduce the measured endpoint")

    def reduce(cuts, propagate=True):
        advance = p_reply_advances(chunks, cuts, propagate=propagate)
        return against_baseline(scenario(requests, advance, fits, since), base)

    definite = {key: clip_seconds(g, 1) for key, g in joined.items() if g["unique_dispatch_join"]}
    all_cuts = {key: clip_seconds(g, 1) for key, g in joined.items()}
    single = {}
    for key, group in joined.items():
        cut = all_cuts[key]
        single[key] = {"group": group, "half_GET_effective_s": clip_seconds(group, .5), "all_GET_effective_s": cut,
                       "half": reduce({key: clip_seconds(group, .5)}),
                       "zero": reduce({key: cut}),
                       "all_other_definite_GET_at_zero": reduce({k: v for k, v in definite.items() if k != key})}
    return {"hardware": observed, "baseline_replay": base, "pressure_start_s": since,
            "P_chunks": sum(map(len, chunks.values())), "GET_batches": len(joined),
            "unambiguous_four_rank_GET_batches": len(definite), "D_ready_callback_evidence": d_pull_join(events, requests, origin),
            "direct_members_only_no_queue_propagation": reduce(definite, propagate=False),
            "half_all_unambiguous_propagated": reduce({key: clip_seconds(joined[key], .5) for key in definite}),
            "zero_all_unambiguous_propagated": reduce(definite),
            "zero_all_including_ambiguous_frontier_sensitivity": reduce(all_cuts), "by_GET_batch": single}


def proxy_replay(arms, fits):
    result = {}
    for order in ("b1", "b2"):
        low, high = (arms[f"{order}-cap{cap}"] for cap in (100, 200))
        rows, fast = low[1], high[1]
        since = planned_recovery_start(rows)
        same_route = {key for key, row in rows.items() if row["predecessor"] is not None and row["P"] == fast[key]["P"]}
        all_successors = {key for key, row in rows.items() if row["predecessor"] is not None}
        advance = {key: max(0, rows[key]["phases_s"]["P_HTTP"] - fast[key]["phases_s"]["P_HTTP"])
                   for key in same_route}
        complete = {key: max(0, rows[key]["phases_s"]["P_HTTP"] - fast[key]["phases_s"]["P_HTTP"])
                    for key in all_successors}
        base = scenario(rows, {}, fits, since)
        actual = {str(cap): {**drained_measurement(arm[1]), "D_egress_GB_during_P_HTTP":
                             sum((b - a) * sum(rates) / 8 for a, b, rates in egress_epochs(arm))}
                  for cap, arm in ((100, low), (200, high))}
        result[order] = {"hardware": actual, "baseline_replay_at_100": base, "same_P_successors": len(same_route),
                         "successors": len(all_successors),
                         "match_faster_P_HTTP_where_same_P_at_fixed_low_EOS": against_baseline(
                             scenario(rows, advance, fits, since), base),
                         "match_faster_P_HTTP_for_all_successors_including_changed_routes": against_baseline(
                             scenario(rows, complete, fits, since), base),
                         "largest_same_P_HTTP_advances": sorted(advance.items(), key=lambda item: -item[1])[:5]}
    return result


def load_method_evidence(path, manifest):
    method_path = path / "native-method-events.jsonl.gz"
    method_hash = sha256(method_path)
    if method_hash != manifest["artifacts"][method_path.name]["sha256"]:
        raise ValueError("Method-event digest changed")
    events = defaultdict(list)
    with gzip_open(method_path, "rt", encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            events[row["case"]].append(row["data"])
    evidence, _ = artifact(path / "native-method-summary.json", manifest)
    return events, evidence["cases"], method_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fits = {name: meta["decode_fit"] for name, (_, meta) in calibration(args.results).items()}
    md = args.results / METHODS
    manifest, digest, arms = load_native_archive(md)
    events, evidence, method_hash = load_method_evidence(md, manifest)
    methods = {case: method_replay(events[case], requests, fits, origin, evidence[case]["P_GET_descriptor_bytes"])
               for case, (_, requests, origin) in arms.items()}
    report = {"method_trace_sha256": method_hash, "method_request_trace_sha256": digest,
              "control_fit": fits, "r69_methods": methods,
              "scope": "The method replay fixes chunk contents, observed P admissions, P and D routes, natural output "
                       "counts, and application dependencies. The serial P frontier uses dispatch/ready overlaps on one "
                       "host. D models either retain observed downstream phases or replay calibrated batch occupancy. "
                       "Zero-GET is a conditional removal of exposed same-rank synchronous service. Formal cap replays "
                       "retime measured P HTTP with a 200 Gbps anchor, which includes other P costs and does not identify GET batches."}
    for name, directory in (("r69", R69), ("r78", R78)):
        _, digest, arms = load_native_archive(args.results / directory)
        report[f"{name}_formal"] = {"trace_sha256": digest, "orders": proxy_replay(arms, fits)}
    write_json(args.output, report)


if __name__ == "__main__":
    main()
