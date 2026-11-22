"""Shadow-test downstream admission signals before P work starts, on fixed-work traces."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from heapq import heapify, heapreplace
from itertools import pairwise
from pathlib import Path
from statistics import median

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import normalized_proxy_id, read_bundle


NS = 1_000_000_000
DECISION_POINTS = ("P_dispatch", "D_ingress")
SLACK_SECONDS = (1.0, 5.0, 15.0)


def first_dispatches(prefill):
    stamps = {}
    for row in prefill["native"]:
        if row["type"] != "p_chunk_dispatch":
            continue
        unix = row["unix_ns"] + row["dispatch_ns"] - row["monotonic_ns"]
        for rid, _, tokens in row["rows"]:
            if tokens > 0:
                key = normalized_proxy_id(rid)
                stamps[key] = min(unix, stamps.get(key, unix))
    return stamps


def rate_before(observations, index, window_s):
    latest = observations[index]
    start = latest["time_s"] - window_s
    state = observations[:index + 1]
    window = state[bisect_left([row["time_s"] for row in state], start):]
    if len(window) < 2:
        return None
    active_seconds = sum((right["time_s"] - left["time_s"]) * left["running"]
                         for left, right in pairwise(window))
    tokens = window[-1]["generation_tokens"] - window[0]["generation_tokens"]
    return tokens / active_seconds if active_seconds >= 2 and tokens >= 16 else None


def estimated_slot(inflight, now, slots, token_rate):
    """Estimate the next slot using observed streaming age and configured budgets."""
    active, waiting = [], []
    tick = 1 / token_rate
    for row in inflight:
        first = row["D_first_chunk_s"]
        if first < now:
            active.append(max(now + tick, first + (row["budget"] - 1) * tick))
        else:
            waiting.append(row)
    if len(active) > slots:
        return None
    release = active + [now] * (slots - len(active))
    heapify(release)
    for row in sorted(waiting, key=lambda item: item["D_begin_s"]):
        heapreplace(release, release[0] + row["budget"] * tick)
    return release[0]


def evaluate(requests, samples, slots, decision, guard_s, window_s):
    observations, schedule = defaultdict(list), defaultdict(list)
    for row in requests.values():
        schedule[row["decode"]].append(row)
    for row in sorted(samples, key=lambda sample: sample["time_s"]):
        observations[row["route"]].append(row)
    timestamps = {role: [row["time_s"] for row in rows] for role, rows in observations.items()}
    evaluations = []
    for key, request in requests.items():
        stamp = request["P_dispatch_s"] if decision == "P_dispatch" else request["D_begin_s"]
        cutoff = stamp - guard_s if decision == "P_dispatch" else stamp
        role = request["decode"]
        index = bisect_left(timestamps[role], cutoff) - 1
        current = observations[role][index] if index >= 0 else None
        fresh = current is not None and cutoff - current["time_s"] <= 2
        earlier = [row for row in schedule[role] if row["D_begin_s"] < cutoff < row["D_end_s"]]
        expected_slack = None
        if fresh:
            speed = rate_before(observations[role], index, window_s)
            if decision == "P_dispatch":
                completed = sorted((row for row in requests.values()
                                    if row["P_HTTP_return_s"] < cutoff and row["prefill"] == request["prefill"]),
                                   key=lambda row: row["P_HTTP_return_s"])
                producer = median(row["P_HTTP_return_s"] - row["P_dispatch_s"] for row in completed[-16:]) if completed else None
            else:
                producer = 0
            if speed is not None and producer is not None:
                predicted = estimated_slot(earlier, cutoff, slots, speed)
                if predicted is not None:
                    expected_slack = max(0.0, predicted - stamp - producer)

        evaluations.append({"id": key, "decision_s": stamp, "D": role, "running": current["running"] if fresh else None,
                            "forwarded_unfinished": len(earlier), "predicted_slack_s": expected_slack,
                            "observed_D_ingress_to_enqueue_s": request["D_transfer_enqueued_s"] - request["D_begin_s"]})
    return evaluations


def score(evaluations, slots, threshold):
    outcomes = defaultdict(Counter)
    for row in evaluations:
        observed = row["observed_D_ingress_to_enqueue_s"] >= threshold
        choices = {
            "running_slots_full": row["running"] is not None and row["running"] >= slots,
            "forwarded_unfinished_at_least_slots": row["forwarded_unfinished"] >= slots,
            "predicted_slack": row["predicted_slack_s"] is not None and row["predicted_slack_s"] >= threshold,
        }
        for policy, decision in choices.items():
            category = ("true_action" if observed else "false_action") if decision else ("miss" if observed else "refrain")
            outcomes[policy][category] += 1
    return {name: {key: counts[key] for key in ("true_action", "false_action", "miss", "refrain")}
            for name, counts in outcomes.items()}


def analyze(summary_path, prefill_path, decode_path, slots, guard_s, window_s):
    summary = read_json(summary_path)
    sources = {"P": sha256(prefill_path), "D": sha256(decode_path)}
    if any(summary["source_sha256"][role] != digest for role, digest in sources.items()):
        raise ValueError("Method traces do not match summary sources")
    prefill, decode = (read_bundle(path) for path in (prefill_path, decode_path))
    p_first = first_dispatches(prefill)
    proxy = defaultdict(dict)
    for row in decode["proxy"]:
        if row["type"] in ("prefill_begin", "decode_begin"):
            proxy[row["id"]][row["type"]] = row
    cases = {}
    for case, arm in summary["cases"].items():
        origin, = (row for row in decode["client"] if row["case"] == case and row["type"] == "client_origin")
        manifest, = (row for row in decode["client_manifest"] if row["case"] == case)
        if not manifest["ignore_eos"]:
            raise ValueError("This forecast requires externally fixed output work")
        bridge_offset = origin["anchor"]["unix_ns"] - origin["anchor"]["monotonic_ns"]
        start_ns = round(origin["monotonic_s"] * NS)
        requests = {}
        for key, row in arm["per_request"].items():
            events = proxy[f"{case}~{key}"]
            stamp = (p_first[events["prefill_begin"]["proxy_request_id"]] - bridge_offset - start_ns) / NS
            outgoing = events["decode_begin"]
            if outgoing["max_tokens"] != row["output_tokens"] or outgoing["route"] != row["decode"]:
                raise ValueError(f"Configured fixed decode work differs for {case}/{key}")
            if not row["P_begin_s"] - guard_s <= stamp <= row["P_HTTP_return_s"] + guard_s:
                raise ValueError(f"P dispatch outside clock tolerance for {case}/{key}")
            requests[key] = {**row, "P_dispatch_s": stamp, "budget": outgoing["max_tokens"]}
        samples = [dict(row, time_s=(row["monotonic_ns"] - start_ns) / NS) for row in decode["decode_supply_sample"]
                   if row["case"] == case]
        cases[case] = {}
        for decision in DECISION_POINTS:
            rows = evaluate(requests, samples, slots, decision, guard_s, window_s)
            cases[case][decision] = {
                "requests": len(rows), "covered_slot_samples": sum(row["running"] is not None for row in rows),
                "predicted_requests": sum(row["predicted_slack_s"] is not None for row in rows),
                "thresholds": {f"{seconds:g}s": score(rows, slots, seconds) for seconds in SLACK_SECONDS},
                "at_5s_witnesses": {
                    "predicted_too_much": [r for r in rows if r["predicted_slack_s"] is not None
                                           and r["predicted_slack_s"] >= 5 > r["observed_D_ingress_to_enqueue_s"]][:5],
                    "forecast_missed": [r for r in rows if r["predicted_slack_s"] is not None
                                        and r["predicted_slack_s"] < 5 <= r["observed_D_ingress_to_enqueue_s"]][:5],
                },
            }
    return {"summary_sha256": sha256(summary_path), "trace_sha256": sources, "clock_guard_s": guard_s,
            "rate_history_s": window_s, "D_slots": slots, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "prefill", "decode", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--d-slots", type=int, required=True)
    parser.add_argument("--clock-guard-s", type=float, default=.15)
    parser.add_argument("--rate-history-s", type=float, default=15)
    args = parser.parse_args()
    if args.d_slots <= 0 or args.clock_guard_s < 0 or args.rate_history_s <= 0:
        parser.error("Expected positive slot count and rate window, and a nonnegative clock guard")
    write_json(args.output, analyze(args.summary, args.prefill, args.decode, args.d_slots,
                                    args.clock_guard_s, args.rate_history_s))


if __name__ == "__main__":
    main()
