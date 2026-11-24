"""Replay r69 peer-source sensitivity with hard D egress budgets and measured GET tapes."""

import argparse
from bisect import bisect_right
from collections import defaultdict
from heapq import heappop, heappush
from itertools import pairwise
from math import floor, inf, isclose
from pathlib import Path
from statistics import median_low

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import ATTENTION_OBJECT_BYTES
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS
from pd_kv_transfer_opt.simulator.freeinference_ab import calibration
from pd_kv_transfer_opt.simulator.get_criticality import METHODS, against_baseline, load_method_evidence, scenario
from pd_kv_transfer_opt.simulator.get_overlap_screen import batches
from pd_kv_transfer_opt.simulator.get_port_replay import mac_weights
from pd_kv_transfer_opt.simulator.get_restore_audit import shared_prefix_blocks, source_prefixes
from pd_kv_transfer_opt.simulator.rdma_replay import PORTS, drained_measurement, load_native_archive
from pd_kv_transfer_opt.simulator.terminal_pressure import planned_recovery_start


EPS = 1e-9
BYTE_EPS = .1
SOURCE_CLOCK_GUARD_S = .042
SETUP_S = .010


def hard_port_split(rate, weights, port_ceiling):
    """Apply the reference port skew wherever its byte budget fits."""
    flat = rate / PORTS
    if flat > port_ceiling * (1 + EPS):
        raise ValueError("Service exceeds the configured aggregate D egress limit")
    high = rate * max(weights)
    fraction = min(1., max(0., (port_ceiling - flat) / (high - flat))) if high > flat else 1.
    return tuple(1 / PORTS + fraction * (w - 1 / PORTS) for w in weights)


def native_service(flows, cap, weights):
    """Fit feasible within-call byte placement while retaining all observed call boundaries."""
    ceiling = cap * NS / 8
    edges = sorted({flow[edge] for flow in flows.values() for edge in ("start", "end")})
    remaining = {key: float(row["bytes"]) for key, row in flows.items()}
    source_rate = {key: row["bytes"] / (row["end"] - row["start"]) for key, row in flows.items()}
    cells = []
    relocated = 0.
    for a, b in pairwise(edges):
        width = b - a
        balance = {key: rate * width for key, rate in source_rate.items()
                   if flows[key]["start"] <= a < flows[key]["end"]}
        wanted = sum(balance.values())
        limit = width * ceiling
        gain = min(1., limit / wanted) if wanted else 1.
        relocated += max(0., wanted - limit)
        balance = {key: amount * gain for key, amount in balance.items()}
        for key, amount in balance.items():
            remaining[key] -= amount
        cells.append((a, b, balance, limit - sum(balance.values())))

    schedules = defaultdict(list)
    demands = {key: [0.] * PORTS for key in flows}
    for a, b, balance, spare in cells:
        # Reinsert just the bytes moved by the nominal cap, prioritizing earlier observed deadlines.
        for key in sorted(balance, key=lambda item: flows[item]["end"]):
            restored = max(0., min(spare, remaining[key]))
            balance[key] += restored
            remaining[key] -= restored
            spare -= restored
        share = hard_port_split(sum(balance.values()) / (b - a), weights, ceiling / PORTS)
        for key, amount in balance.items():
            sizes = tuple(amount * ratio for ratio in share)
            for port, size in enumerate(sizes):
                demands[key][port] += size
            schedule = schedules[key]
            rates = tuple(size / (b - a) for size in sizes)
            schedule.append((a - flows[key]["start"], rates))
    if any(abs(left) > 1 for left in remaining.values()):
        raise ValueError("Observed bytes could not be fitted inside the original GET envelopes")
    for key, plan in schedules.items():
        if not isclose(sum(demands[key]), flows[key]["bytes"], abs_tol=1.):
            raise ValueError("Lost bytes while fitting the per-port native schedule")
        end = flows[key]["end"] - flows[key]["start"]
        plan.append((end, tuple(size / end for size in demands[key])))
    return {key: (tuple(t for t, _ in plan), tuple(r for _, r in plan), demands[key])
            for key, plan in schedules.items()}, relocated


def replay_transfers(flows, calls, copy, placement, trial_starts, candidate, cap, profile):
    """Replay simultaneous RoCE work across all P workers and include device-copy costs."""
    ceiling = cap * NS / (8 * PORTS)
    arrivals, on_wire, starts, ends = [], {}, {}, {}
    projected = {key: float(flow["bytes"] - candidate[key]) / flow["bytes"] for key, flow in flows.items()}
    for tape in calls:
        first = tape[0]
        heappush(arrivals, (trial_starts[first[0]] + flows[first]["offset"], first))
    following = {a: b for tape in calls for a, b in pairwise(tape)}
    sent = [0.] * PORTS
    time = arrivals[0][0]
    while arrivals or on_wire:
        if not on_wire:
            time = max(time, arrivals[0][0])
        while arrivals and arrivals[0][0] <= time + EPS:
            release, key = heappop(arrivals)
            starts[key] = release
            on_wire[key] = release, [size * projected[key] for size in profile[key][2]]
        next_time = arrivals[0][0] if arrivals else inf
        offered = {}
        for key, (release, outstanding) in on_wire.items():
            points, rates, _ = profile[key]
            index = bisect_right(points, time - release + EPS) - 1
            if index + 1 < len(points):
                next_time = min(next_time, release + points[index + 1])
            offered[key] = [pace if left > BYTE_EPS else 0. for pace, left in zip(rates[index], outstanding, strict=True)]
        scales = []
        for port in range(PORTS):
            rate = sum(v[port] for v in offered.values())
            scales.append(min(1., ceiling / rate) if rate else 0.)
        rates = {key: [x * y for x, y in zip(raw, scales, strict=True)] for key, raw in offered.items()}
        for key, (_, outstanding) in on_wire.items():
            completion = min((time + b / v for b, v in zip(outstanding, rates[key], strict=True) if v), default=inf)
            next_time = min(next_time, completion)
        if not time < next_time < inf:
            raise ValueError("The peer replay stalled before a finite service event")
        for key, (_, outstanding) in on_wire.items():
            for port, rate in enumerate(rates[key]):
                amount = min(outstanding[port], rate * (next_time - time))
                outstanding[port] -= amount
                sent[port] += amount
        time = next_time
        for key in [key for key, (_, left) in on_wire.items() if max(left) <= BYTE_EPS]:
            begin, _ = on_wire.pop(key)
            local = SETUP_S + candidate[key] / (copy * NS) if candidate[key] else 0.
            done = time + local if placement == "serial" else max(time, begin + local)
            ends[key] = done
            if key in following:
                nxt = following[key]
                heappush(arrivals, (done + flows[nxt]["start"] - flows[key]["end"], nxt))
    if not isclose(sum(sent) + sum(candidate.values()), sum(r["bytes"] for r in flows.values()), abs_tol=200.):
        raise ValueError("Peer and RoCE payloads did not reconcile")
    return starts, ends, sent


def peer_tapes(chunks, loads, joined, hypothesis):
    by_id = {row["id"]: row for steps in chunks.values() for row in steps}
    origin = min(a for ranks in loads.values() for rank in ranks for a, _, _ in rank["reads"])
    flows, tapes, fixed = {}, {}, set()
    for key, match in joined.items():
        rows = loads[match["P"], tuple(match["request_ids_in_GET"])]
        candidate = hypothesis[key]
        if len(rows) != 4 or sorted(x["id"] for x in candidate["requests"]) != match["request_ids_in_GET"]:
            raise ValueError("The peer-source plan differs from the profiled worker statement")
        step = by_id[key]
        if not candidate["unambiguous_load"] or any(a < step["start"] for r in rows for a, _, _ in r["reads"]):
            fixed.add(key)
        worker_tapes = {}
        for rank in rows:
            tape = []
            for idx, ((a, b, size), item) in enumerate(zip(rank["reads"], candidate["requests"], strict=True)):
                if size * len(rows) != item["GET_bytes"]:
                    raise ValueError("Peer object plan lost a rank's read payload")
                handle = key, rank["worker"], idx
                flows[handle] = {"start": (a - origin) / NS, "end": (b - origin) / NS, "bytes": size,
                                 "offset": (a - step["start"]) / NS, "request": item["id"],
                                 "eligible_blocks": item["peer_attention_blocks_upper"]}
                tape.append(handle)
            worker_tapes[rank["worker"]] = (tape, (rank["end"] - rank["reads"][-1][1]) / NS)
        tapes[key] = {"ranks": worker_tapes, "result_after_workers_s": (step["ready"] - max(r["end"] for r in rows)) / NS}
    return flows, tapes, origin, fixed


def retime_P(chunks, service, *, held_dispatch, frozen, origin):
    starts, native_finish, moved_finish = {}, {}, {}
    for steps in chunks.values():
        previous = -inf
        for step in steps:
            key = step["id"]
            original = (step["start"] - origin) / NS
            if held_dispatch:
                start = max(previous, (step["dispatch"] - origin) / NS)
            else:
                start = max(previous, (step["admission"] - origin) / NS) + step["gap"] / NS
            if key in frozen:
                start = max(start, original)
            starts[key] = start
            previous = start + service[key]
            for req in step["ids"]:
                native_finish[req] = max(native_finish.get(req, -inf), (step["ready"] - origin) / NS)
                moved_finish[req] = max(moved_finish.get(req, -inf), previous)
    return starts, {key: end - moved_finish[key] for key, end in native_finish.items()}


def replay_P(chunks, flows, tapes, origin, fixed, cap, profile, removed, *, local_rate, placement, hold_dispatch,
             feedback):
    timestamps = {s["id"]: (s["start"] - origin) / NS for seq in chunks.values() for s in seq}
    native_service_s = {s["id"]: (s["ready"] - s["start"]) / NS for seq in chunks.values() for s in seq}
    tape_order = [keys for tape in tapes.values() for keys, _ in tape["ranks"].values()]
    trial = timestamps.copy()
    stable = not feedback
    residual = 0.
    for iteration in range(1, (61 if feedback else 2)):
        released, ends, sent = replay_transfers(flows, tape_order, local_rate, placement, trial, removed, cap, profile)
        service = native_service_s.copy()
        for key, tape in tapes.items():
            done = max(ends[seq[-1]] + end for seq, end in tape["ranks"].values())
            service[key] = max(0., done + tape["result_after_workers_s"] - trial[key])
        starts, adv = retime_P(chunks, service, held_dispatch=hold_dispatch, frozen=fixed, origin=origin)
        # These native calls preceded the observable worker frontier, so anchor them.
        projected = {key: timestamps[key] if key in fixed else starts[key] for key in tapes}
        residual = max(abs(projected[k] - trial[k]) for k in projected)
        if not feedback or residual < 1e-5:
            stable = True
            break
        for key in projected:
            trial[key] = .5 * (projected[key] + trial[key])
    ends_error = max(abs(ends[key] - row["end"]) for key, row in flows.items())
    if not any(removed.values()) and (ends_error > 1e-6 or max(map(abs, adv.values())) > 1e-6):
        raise ValueError("Zero-change replay deviates from recorded worker timing")
    return adv, released, {"wire_bytes_per_port": sent, "iterations": iteration, "start_residual_s": residual,
                 "feedback_converged": stable, "max_native_GET_error_s": ends_error if not any(removed.values()) else None,
                 "last_P_batch_advance_s": {r: max(timestamps[s["id"]] + native_service_s[s["id"]] for s in seq)
                        - max(starts[s["id"]] + service[s["id"]] for s in seq) for r, seq in chunks.items()},
                 "GET_batch_service_saved_s": {key: native_service_s[key] - service[key] for key in tapes}}


def alternate_sources(requests, flows, prefix_index, reads, origin, age_limit, publication_lead):
    """Find logically matching pages already produced on the other P in this same capture."""
    families, tips = prefix_index
    times = defaultdict(list)
    for row in flows.values():
        times[row["request"]].append(row["start"] + origin)
    bounds, donors = {}, {}
    for req, span in times.items():
        bound = reads[req]
        if bound["eligible"] <= 0:
            continue
        old = requests[req]["predecessor"]
        candidates = []
        for key, candidate in requests.items():
            if (candidate["P"] == requests[old]["P"] and min(span) - candidate["P_end_s"] >= publication_lead
                    and max(span) - candidate["P_end_s"] + SOURCE_CLOCK_GUARD_S <= age_limit):
                lcp = shared_prefix_blocks(req, key, families, tips)
                covered = max(0, min(bound["eligible"], lcp - bound["first"]))
                if covered:
                    candidates.append((covered, candidate["P_end_s"], key))
        if candidates:
            bounds[req], _, donors[req] = max(candidates)
    return bounds, donors


def measure_condition(chunks, flows, tapes, origin, unified_origin_s, fixed, cap, profile, removed, requests, *,
                      local_rate, placement, held, feedback, donors=None, age_limit=inf):
    moved = removed.copy()
    rejected = set()
    while True:
        advance, releases, wire = replay_P(chunks, flows, tapes, origin, fixed, cap, profile, moved,
                                          local_rate=local_rate, placement=placement, hold_dispatch=held,
                                          feedback=feedback)
        invalid = set()
        for key, n in moved.items():
            if not n or donors is None:
                continue
            req = flows[key]["request"]
            source = donors[req]
            published = requests[source]["P_end_s"] - advance[source]
            read_age = releases[key] + unified_origin_s - published
            if not SOURCE_CLOCK_GUARD_S <= read_age <= age_limit - SOURCE_CLOCK_GUARD_S:
                invalid.add(req)
        if not invalid:
            return advance, wire, sum(moved.values()) / NS, sorted(rejected)
        # Keep the native fallback if the replay no longer proves publication before GET.
        rejected.update(invalid)
        moved = {key: 0 if flows[key]["request"] in invalid else n for key, n in moved.items()}


def simulate_case(arm, events, measured_bytes, source, fits, cap, prefix_index=None, observed_plans=None):
    raw, requests, origin_s = arm
    p_events, chunks, loads, joined = batches(events, requests, measured_bytes)
    flows, tapes, origin, fixed = peer_tapes(chunks, loads, joined, source["candidates_by_batch"])
    d_clock, = (row["anchor"] for row in raw["client"] if row["type"] == "client_origin")
    p_to_d = median_low(row["unix_ns"] - row["monotonic_ns"] for row in p_events) - d_clock["unix_ns"] + d_clock["monotonic_ns"]
    unified_origin_s = (origin + p_to_d) / NS - origin_s
    ages = {}
    for row in flows.values():
        if row["eligible_blocks"]:
            req = row["request"]
            donor = requests[requests[req]["predecessor"]]
            age = row["start"] + unified_origin_s - donor["P_end_s"]
            ages[req] = max(ages.get(req, 0.), age)
    g = list(mac_weights(raw["telemetry"], cap, p_to_d, origin, flows, "pooled_MAC")["P0"])
    configs = {
        "quarter_parallel": (.25, inf, 10., "parallel", None),
        "half_serial": (.5, inf, 10., "serial", None),
        "half_parallel": (.5, inf, 10., "parallel", None),
        "half_parallel_5GBs": (.5, inf, 5., "parallel", None),
        "half_parallel_8_5GBs": (.5, inf, 8.5, "parallel", None),
        "half_parallel_only_P0": (.5, inf, 10., "parallel", "P0"),
        "half_parallel_only_P1": (.5, inf, 10., "parallel", "P1"),
        "all_if_30s": (1., 30., 10., "parallel", None),
        "all_if_40s": (1., 40., 10., "parallel", None),
        "all_if_50s": (1., 50., 10., "parallel", None),
        "all_if_60s": (1., 60., 10., "parallel", None),
        "all_if_70s": (1., 70., 10., "parallel", None),
        "threequarters_parallel": (.75, inf, 10., "parallel", None),
    }
    newer = {}
    if prefix_index is not None:
        reads = {row["request"]: {"first": row["attention_span_in_blocks"][0]}
                 for batch in observed_plans.values() for row in batch["read_plan"]}
        for row in flows.values():
            reads[row["request"]]["eligible"] = row["eligible_blocks"]
        for age, lead in ((40., SOURCE_CLOCK_GUARD_S), (50., SOURCE_CLOCK_GUARD_S), (50., 6.)):
            name = f"newer_peer_{age:g}s_lead_{lead:g}s"
            newer[name] = alternate_sources(requests, flows, prefix_index, reads, unified_origin_s, age, lead)
            configs[name] = (1., age, 10., "parallel", None)
    since = planned_recovery_start(requests)
    baseline = scenario(requests, {}, fits, since)
    model = {}
    for shape, weights in (("balanced", [1 / PORTS] * PORTS), ("mac", g)):
        profile, relocated = native_service(flows, cap, weights)
        empty = dict.fromkeys(flows, 0)
        _, _, control = replay_P(chunks, flows, tapes, origin, fixed, cap, profile, empty, local_rate=10.,
                                 placement="serial", hold_dispatch=False, feedback=True)
        conditions = {}
        for name, (fraction, age_limit, local_rate, placement, role) in configs.items():
            removed = {}
            alternative, donors = newer.get(name, (None, None))
            for key, row in flows.items():
                use = key[0] not in fixed and row["eligible_blocks"] and (role is None or key[0].startswith(role))
                req = row["request"]
                if alternative is not None:
                    blocks = alternative.get(req, 0)
                else:
                    blocks = row["eligible_blocks"] if ages.get(req, inf) + SOURCE_CLOCK_GUARD_S <= age_limit else 0
                if use and blocks:
                    moved = floor(blocks * fraction) * ATTENTION_OBJECT_BYTES
                    solo = row["end"] - row["start"]
                    local = SETUP_S + moved / (local_rate * NS)
                    remote = solo * (1 - moved / row["bytes"])
                    cost = local + remote if placement == "serial" else max(local, remote)
                    removed[key] = moved if cost < solo else 0
                else:
                    removed[key] = 0
            inputs = {"moved_GB": sum(removed.values()) / NS, "eligible_fraction": fraction,
                      "hypothetical_source_lifetime_s": age_limit if age_limit < inf else None,
                      "net_peer_copy_GBps_per_rank": local_rate, "copy_order": placement, "only_receiver": role}
            for label, held, feedback in (("anchored_calls_follow_P_batches", False, False),
                                          ("retimed_calls_follow_P_batches", False, True),
                                          ("retimed_calls_hold_dispatch", True, True)):
                adv, wire, realized, rejected = measure_condition(
                    chunks, flows, tapes, origin, unified_origin_s, fixed, cap, profile, removed, requests,
                    local_rate=local_rate, placement=placement, held=held, feedback=feedback,
                    donors=donors, age_limit=age_limit)
                values = against_baseline(scenario(requests, adv, fits, since), baseline) if wire["feedback_converged"] else None
                inputs[label] = {"wire": wire, "throughput": values, "realized_moved_GB": realized,
                                 "source_publication_or_age_fallback_requests": rejected}
            if donors is not None:
                inputs["alternate_donor_by_request"] = donors
            conditions[name] = inputs
        model[shape] = {"weights": weights, "bytes_relocated_to_keep_native_cap": relocated,
                        "zero_change_control": control, "cases": conditions}
    return {"observed": drained_measurement(requests), "baseline": baseline, "unmodified_early_batches": sorted(fixed),
            "candidate_source_age_s_at_last_observed_rank_GET_start": ages,
            "models": model}


def analyze(results, ledger=None):
    folder = results / METHODS
    manifest, digest, arms = load_native_archive(folder)
    method, receipts, method_sha = load_method_evidence(folder, manifest)
    source = results / "2026-11-23-r69-peer-source-screen" / "summary.json"
    hypothesis = read_json(source)
    if hypothesis["source_hashes"]["request"] != digest or hypothesis["source_hashes"]["method"] != method_sha:
        raise ValueError("Conditional source eligibility was computed for a different trace")
    ledger_sha, prefix_index, observed = None, None, None
    if ledger is not None:
        ledger_sha, families, tips = source_prefixes(ledger, manifest["workload"])
        prefix_index = families, tips
        path = results / "2026-11-22-cpu-get-criticality" / "restore-audit.json"
        if sha256(path) != hypothesis["source_hashes"]["prior_read_plan"]:
            raise ValueError("Peer replay did not use the original per-request page ranges")
        observed = read_json(path)["cases"]
    fits = {name: row["decode_fit"] for name, (_, row) in calibration(results).items()}
    return {"source_sha256": sha256(source), "request_trace_sha256": digest, "method_trace_sha256": method_sha,
            "terminal_ledger_sha256": ledger_sha,
            "scope": "Observed logical prefix intersection and fixed per-arm EOS/routes. The source-lifetime threshold "
                     "is a sensitivity mask on observed GET ages, not a measurement of HBM residency. Per-port curves "
                     "preserve measured descriptor bytes within the observed API envelopes. The MAC-weighted assignment "
                     "is a conditional footprint. Sources that start before the measured worker frontier remain anchored. "
                     "Local copy rates are net assumed rates, and device or donor interference beyond those rates is unmeasured. "
                     "A newer donor is also checked after retiming, falling back to native reads if its publication or age "
                     "is no longer compatible with any rank's new GET start.",
            "arms": {case: simulate_case(arm, method[case], receipts[case]["P_GET_descriptor_bytes"],
                                        hypothesis["cases"][case], fits, int(case[-3:]), prefix_index,
                                        observed[case]["batches"] if observed else None) for case, arm in arms.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--terminal-ledger", type=Path, help="Hashed Terminal-Bench ledger for alternate-source screening")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, analyze(args.results, args.terminal_ledger))


if __name__ == "__main__":
    main()
