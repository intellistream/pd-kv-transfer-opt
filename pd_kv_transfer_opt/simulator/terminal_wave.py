"""Replay a ready successor wave on two synchronous prefill workers and two TPs for decode."""

from __future__ import annotations

from collections import defaultdict, deque
from math import ceil, isclose

from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union
from pd_kv_transfer_opt.simulator.decode_supply import DECODE_ROLES, simulate_decode
from pd_kv_transfer_opt.simulator.rdma_replay import PORTS
from pd_kv_transfer_opt.simulator.terminal_pressure import peak_window


P_ROLES = ("P0", "P1")


def prefill_wave(requests, cap_gbps, shares, service):
    """Advance sync GETs with the most loaded rail governing two equal-share readers."""
    if len(shares) != PORTS or not isclose(sum(shares), 1) or min(shares) < 0:
        raise ValueError("Expected measured nonnegative shares of all D egress ports")
    first_width = service.get("first_GET_batch_width", service["GET_batch_width"])
    if cap_gbps <= 0 or not 0 < service["rail_efficiency"] <= 1 or min(first_width, service["GET_batch_width"]) < 1:
        raise ValueError("Invalid service or cap")
    rate_GB_s = cap_gbps * service["rail_efficiency"] / (8 * PORTS * max(shares))
    queued = {role: deque(sorted((k for k, r in requests.items() if r["P"] == role),
                                 key=lambda k: (requests[k]["P_admit_s"], k))) for role in P_ROLES}
    if not requests or any(row["P"] not in P_ROLES or row["GET_GB"] < 0 or row["suffix_tokens"] < 0 for row in requests.values()):
        raise ValueError("Expected a nonempty request set on the two P instances with nonnegative work")
    # Each model worker's first issue includes API-to-engine launch time.
    available = {role: min(requests[k]["P_admit_s"] for k in queue) + service["first_GET_launch_s"]
                 for role, queue in queued.items() if queue}
    clock = min(available.values())
    pending, completed, intervals, calls, issued = {}, {}, [], [], set()

    def retire(role, keys, get_started, bytes_GB):
        post = service["batch_nonwire_s"] + sum(requests[key]["suffix_tokens"] for key in keys) / service["warm_tokens_s"]
        available[role] = clock + post
        for key in keys:
            completed[key] = available[role]
        calls.append({"P": role, "requests": keys, "start_s": get_started, "GET_end_s": clock,
                      "P_reply_s": available[role], "GET_GB": bytes_GB})

    while len(completed) < len(requests):
        for role, queue in queued.items():
            if not queue or role in pending or available[role] > clock or requests[queue[0]]["P_admit_s"] > clock:
                continue
            batch = []
            width = service["GET_batch_width"] if role in issued else first_width
            for _ in range(width):
                if not queue or requests[queue[0]]["P_admit_s"] > clock:
                    break
                batch.append(queue.popleft())
            issued.add(role)
            volume = sum(requests[k]["GET_GB"] for k in batch)
            if volume:
                pending[role] = [batch, clock, volume, volume]
            else:
                retire(role, batch, clock, 0.0)
        next_at = [max(available[role], requests[queue[0]]["P_admit_s"]) for role, queue in queued.items()
                   if queue and role not in pending]
        if pending:
            share_rate = rate_GB_s / len(pending)
            next_at.append(clock + min(batch[-1] for batch in pending.values()) / share_rate)
        if not next_at:
            break
        following = min(next_at)
        rates = tuple(rate_GB_s * share * 8 if pending else 0.0 for share in shares)
        if following > clock:
            intervals.append((clock, following, rates))
        elapsed, clock = following - clock, following
        for role, batch in list(pending.items()):
            batch[-1] -= elapsed * share_rate
            if batch[-1] <= 1e-8:
                keys, get_started, bytes_GB, _ = pending.pop(role)
                retire(role, keys, get_started, bytes_GB)

    offered_GB = sum(row["GET_GB"] for row in requests.values())
    moved = [sum((right - left) * rate[i] / 8 for left, right, rate in intervals) for i in range(PORTS)]
    if len(completed) != len(requests) or not all(isclose(gb, offered_GB * share, rel_tol=1e-9, abs_tol=1e-8)
                                                 for gb, share in zip(moved, shares, strict=True)):
        raise RuntimeError("The simulated P wave did not conserve its requests or per-port bytes")
    return completed, intervals, calls, moved


def decode_deliveries(jobs, P_reply, service, slots, handoff_s, placements=None):
    """Drive the decode simulator from each actual P handoff and a distinct request's natural output."""
    rows = {}
    scores = {}
    for key in sorted(jobs, key=lambda name: (P_reply[name], jobs[name]["P_admit_s"], name)):
        job = jobs[key]
        p_end = P_reply[key]
        first_d = p_end + job.get("handoff_s", handoff_s)
        rows[key] = {"offered_s": p_end, "predecessor": None, "tool_gap_s": 0, "ready_s": p_end,
                     "first_s": first_d, "end_s": first_d,
                     "output_tokens": job["output_tokens"],
                     "phases_s": {"ready_to_P": 0.0, "P_HTTP": 0.0, "P_return_to_D": 0.0}}
        if placements is not None:
            rows[key]["D"] = placements[key]
        scores[key] = job["decode_score"]
    simulation = simulate_decode(rows, service, slots=slots, include_requests=True,
                                 native_scores=scores if placements is None else None)
    if placements is not None:
        for key, row in simulation["per_request"].items():
            row["D"] = placements[key]
    return simulation


def profile_supply(jobs, P_reply, decode, epochs, cap_gbps, start_s):
    def clipped_span(left, right, begin, end):
        return max(0.0, min(right, end) - max(left, begin))

    per_request = decode["per_request"]
    hot = [(a, b) for a, b, rates in epochs if max(rates) >= .85 * cap_gbps / PORTS]
    by_D = {}
    for role in DECODE_ROLES:
        keys = [key for key, row in per_request.items() if row["D"] == role]
        inflight = [(jobs[key]["P_admit_s"], P_reply[key]) for key in keys if key in jobs]
        generating = [(per_request[key]["first_s"], per_request[key]["end_s"]) for key in keys]
        changes = defaultdict(lambda: [0, 0, 0])
        for dimension, spans in enumerate((inflight, generating, hot)):
            for begin, end in spans:
                changes[begin][dimension] += 1
                changes[end][dimension] -= 1
        boundaries = sorted(changes)
        idle = width_hot = hot_wall = 0.0
        idle_hot = []
        P_pending = width = hot_ports = 0
        for begin, end in zip(boundaries, boundaries[1:]):
            delta_P, delta_D, delta_hot = changes[begin]
            P_pending += delta_P
            width += delta_D
            hot_ports += delta_hot
            if not P_pending:
                continue
            dt = end - begin
            idle += dt if width == 0 else 0.0
            if hot_ports:
                width_hot += width * dt
                hot_wall += dt
                if width == 0:
                    if idle_hot and isclose(idle_hot[-1][1], begin, rel_tol=0.0, abs_tol=1e-8):
                        idle_hot[-1] = (idle_hot[-1][0], end)
                    else:
                        idle_hot.append((begin, end))
        by_D[role] = {"idle_with_eligible_P_pending_s": idle, "idle_with_pending_P_and_hot_D_port_s": interval_union(idle_hot),
                      "longest_idle_with_pending_P_and_hot_D_port_s": max((b-a for a,b in idle_hot), default=0.0),
                      "mean_active_sequences_while_P_pending_and_port_hot": width_hot / hot_wall if hot_wall else 0.0,
                      "live_sequence_s_first_10": sum(clipped_span(a, b, start_s, start_s + 10) for a, b in generating)}
    return {"D_first_tokens_in_first_10_s": sum(per_request[key]["first_s"] <= start_s + 10 for key in jobs),
            "D_ready_within_first_10_s": sum(per_request[key]["D_eligible_s"] <= start_s + 10 for key in jobs),
            "P_replies_in_first_10_s": sum(t <= start_s + 10 for t in P_reply.values()),
            "P_last_reply_s": max(P_reply.values()), "first_D_token_s": min(per_request[key]["first_s"] for key in jobs),
            "after_last_P_reply_s": decode["finish_s"] - max(P_reply.values()),
            "port_hot_s": interval_union(hot),
            "longest_aggregate_at_least_100_Gbps_s": longest_aggregate_window(epochs),
            "peak_20_s_D_TX_Gbps": peak_window(epochs, 20.0) if epochs else 0.0,
            "D": by_D}


def sample_ports(epochs, start_s, width_s):
    """Convert variable service events into zero-padded, byte-conserving telemetry intervals."""
    if not epochs:
        return []
    if width_s <= 0 or start_s > epochs[0][0]:
        raise ValueError("Sample from no later than the first modeled wire event")
    result, index = [], 0
    for sample in range(ceil((epochs[-1][1] - start_s) / width_s)):
        left = start_s + sample * width_s
        right = left + width_s
        totals = [0.0] * PORTS
        while index < len(epochs):
            begin, end, rates = epochs[index]
            if begin >= right:
                break
            overlap = max(0.0, min(end, right) - max(begin, left))
            for port in range(PORTS):
                totals[port] += rates[port] * overlap
            if end > right:
                break
            index += 1
        result.append((left, right, tuple(value / width_s for value in totals)))
    return result


def longest_aggregate_window(epochs):
    current, best = 0.0, 0.0
    for start, stop, rates in epochs:
        current = current + stop - start if sum(rates) >= 100.0 else 0.0
        best = max(best, current)
    return best


def ongoing_supply_gap(low, high):
    """Compare ready stock while both decode instances have started and the low cap is hot."""
    runs = (low, high)
    delivery = [{key: run["sim"]["per_request"][key] for key in run["P_reply_s"]} for run in runs]
    if any({row["D"] for row in wave.values()} != set(DECODE_ROLES) for wave in delivery):
        return {"compared_hot_s": 0.0, "100_fewer_ready_s": 0.0, "longest_100_fewer_ready_s": 0.0,
                "mean_ready_sequences": {"100": None, "200": None}}
    begin = max(min(row["first_s"] for row in wave.values() if row["D"] == role)
                for wave in delivery for role in DECODE_ROLES)
    end = min(max(run["P_reply_s"].values()) for run in runs)
    changes = defaultdict(lambda: [0, 0, 0])
    for cap_index, run in enumerate(runs):
        for row in run["sim"]["per_request"].values():
            changes[row["D_eligible_s"]][cap_index] += 1
            changes[row["end_s"]][cap_index] -= 1
    for left, right, rates in low["epochs"]:
        if max(rates) >= .85 * 100 / PORTS:
            changes[left][2] += 1
            changes[right][2] -= 1
    edges = sorted(changes)
    counts = [0, 0, 0]
    compared = deficit = longest = consecutive = 0.0
    ready_s = [0.0, 0.0]
    for left, right in zip(edges, edges[1:]):
        counts = [n + d for n, d in zip(counts, changes[left], strict=True)]
        width = max(0.0, min(right, end) - max(left, begin))
        if width and counts[2]:
            compared += width
            for i in (0, 1):
                ready_s[i] += width * counts[i]
            if counts[0] < counts[1] and counts[0] < 32:
                consecutive += width
                deficit += width
                longest = max(longest, consecutive)
                continue
        consecutive = 0.0
    return {"compared_hot_s": compared, "100_fewer_ready_s": deficit, "longest_100_fewer_ready_s": longest,
            "mean_ready_sequences": {cap: n / compared if compared else None for cap, n in zip(("100", "200"), ready_s, strict=True)}}
