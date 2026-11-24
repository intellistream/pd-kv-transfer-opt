"""Replay a bounded decode batch from observed prefill delivery and fixed output work."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from heapq import heappop, heappush
from itertools import count, pairwise
from pathlib import Path
from statistics import linear_regression, median

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.profile_closed_sessions import gauge, read_trace, requests_for


DECODE_ROLES = ("D0", "D1")


def native_arms(archive):
    manifest = read_json(archive / "manifest.json")
    trace = archive / manifest["event_trace"]
    digest = sha256(trace)
    if digest != manifest["artifacts"][trace.name]["sha256"]:
        raise ValueError(f"Trace digest mismatch in {archive.name}")
    requests = manifest["workload"]["request_sources"]
    arms = {}
    for name, arm in read_trace(trace).items():
        origin, = (row["monotonic_s"] for row in arm["client"] if row["type"] == "client_origin")
        arms[name] = (arm, requests_for(arm, requests, origin), origin)
    return digest, arms


def decode_samples(trace, requests, origin):
    samples, by_role = defaultdict(list), defaultdict(list)
    for t in trace["telemetry"]:
        for row in t.get("engines", ()):
            role = row["role"]
            output = gauge(row.get("values", {}), "generation_tokens_total")
            if role in DECODE_ROLES and output is not None:
                samples[role].append((row["monotonic_ns"] / 1e9 - origin, output))
    for row in requests.values():
        by_role[row["D"]].append((row["first_s"], row["end_s"]))

    for role in DECODE_ROLES:
        for (begin, previous), (end, current) in pairwise(samples[role]):
            duration, generated = end - begin, current - previous
            if begin < 0 or not 0.1 <= duration <= 6 or generated < 20:
                continue
            active_area = sum(max(0, min(end, last) - max(begin, first)) for first, last in by_role[role])
            active_width = active_area / duration
            if 0.9 <= active_width <= 16:
                yield active_width, active_area / generated


def fit_decode(observations):
    widths, steps = zip(*observations, strict=True)
    slope, intercept = linear_regression([width - 1 for width in widths], steps)
    if intercept <= 0 or slope < 0:
        raise ValueError("The observations do not identify a positive decode period")
    return {"first_sequence_period_s": intercept, "additional_sequence_period_s": slope,
            "samples": len(widths), "observed_mean_width": [min(widths), max(widths)],
            "median_fit_error_ms": 1000 * median(abs(s - intercept - slope * (n - 1)) for n, s in observations)}


def simulate_decode(requests, service, *, prefill_http_s=None, slots=16, include_requests=False, native_scores=None,
                    step_savings_s=None):
    if not requests or slots < 1:
        raise ValueError("Decode replay needs a nonempty request trace and at least one slot per D")
    if native_scores is not None and (native_scores.keys() != requests.keys() or any(n <= 0 for n in native_scores.values())):
        raise ValueError("Native decode routing needs a positive proxy score for each request")
    step_periods = {width: service["first_sequence_period_s"] + service["additional_sequence_period_s"] * (width - 1)
                    for width in range(1, slots + 1)}
    for width, saving in (step_savings_s or {}).items():
        if width not in step_periods or not 0 <= saving < step_periods[width]:
            raise ValueError("Decode step savings must leave a positive period at a supported width")
        step_periods[width] -= saving
    placements = {name: row["D"] for name, row in requests.items()} if native_scores is None else {}
    decode_load = dict.fromkeys(DECODE_ROLES, 0)
    records, children = {}, defaultdict(list)
    upstream = {}
    timed, ordinal = [], count()
    active = {role: {} for role in DECODE_ROLES}
    waiting = {role: deque() for role in DECODE_ROLES}

    def ready(key, predecessor_done=None):
        row = requests[key]
        offered = row["offered_s"]
        release = offered if predecessor_done is None else predecessor_done + row["tool_gap_s"]
        eligible = max(offered, release)
        delay = row["first_s"] - row["ready_s"]
        if prefill_http_s is not None:
            delay += prefill_http_s[key] - row["phases_s"]["P_HTTP"]
        decode_eligible = eligible + max(0.0, delay)
        if include_requests or native_scores is not None:
            phases = row["phases_s"]
            prefill_http = phases["P_HTTP"] if prefill_http_s is None else prefill_http_s[key]
            prefill_begin = eligible + phases["ready_to_P"]
        if native_scores is None:
            heappush(timed, (decode_eligible, next(ordinal), key, None))
        else:
            decode_reserve = prefill_begin + prefill_http + phases["P_return_to_D"]
            if decode_reserve > decode_eligible + 1e-8:
                raise ValueError(f"Decode reservation follows eligibility for {key}")
            heappush(timed, (decode_reserve, next(ordinal), key, decode_eligible))
        if include_requests:
            upstream[key] = {"ready_s": eligible, "P_start_s": prefill_begin,
                             "P_end_s": prefill_begin + prefill_http, "D_eligible_s": decode_eligible}

    for key, row in requests.items():
        if row["predecessor"] is None:
            ready(key)
        else:
            children[row["predecessor"]].append(key)
    if not timed:
        raise ValueError("The supplied trace has no root request")

    def complete(key, at):
        records[key]["end_s"] = at
        if native_scores is not None:
            decode_load[placements[key]] -= native_scores[key]
        for child in children[key]:
            ready(child, at)

    clock = timed[0][0]
    while len(records) < len(requests) or any(active.values()):
        while timed and timed[0][0] <= clock:
            _, _, key, decode_eligible = heappop(timed)
            if decode_eligible is None:
                waiting[placements[key]].append(key)
            else:
                role = min(DECODE_ROLES, key=decode_load.__getitem__)
                placements[key] = role
                decode_load[role] += native_scores[key]
                heappush(timed, (decode_eligible, next(ordinal), key, None))
        for role, remaining in active.items():
            queue = waiting[role]
            while queue and len(remaining) < slots:
                key = queue.popleft()
                records[key] = {"first_s": clock}
                tokens = requests[key]["output_tokens"] - 1
                if tokens:
                    remaining[key] = float(tokens)
                else:
                    complete(key, clock)
        periods = {role: step_periods[len(rows)] for role, rows in active.items() if rows}
        candidates = [clock + min(active[role].values()) * period for role, period in periods.items()]
        if timed:
            candidates.append(timed[0][0])
        if not candidates:
            if len(records) == len(requests):
                break
            raise RuntimeError("No native dependency can release the remaining decode requests")
        end = min(candidates)
        for role, period in periods.items():
            progress = (end - clock) / period
            remaining = active[role]
            for key in tuple(remaining):
                remaining[key] -= progress
                if remaining[key] <= 1e-8:
                    complete(key, end)
                    del remaining[key]
        clock = end

    tail = max(records, key=lambda key: records[key]["end_s"])
    finish = records[tail]["end_s"]
    outputs = sum(row["output_tokens"] for row in requests.values())
    live_first = min(row["offered_s"] for row in requests.values())
    result = {"finish_s": finish, "tail_id": tail, "output_tokens": outputs,
              "drained_throughput_tokens_s": outputs / (finish - live_first),
              "median_finish_delta_from_observation_s": median(records[key]["end_s"] - row["end_s"] for key, row in requests.items())}
    if include_requests:
        result["per_request"] = {key: {**upstream[key], **row} for key, row in records.items()}
        if native_scores is not None:
            for key, row in result["per_request"].items():
                row["D"] = placements[key]
    return result


def analyze(calibration, holdout):
    fits, fit_traces = [], {}
    for path in calibration:
        digest, arms = native_arms(path)
        fit_traces[path.name] = digest
        for arm in arms.values():
            fits.extend(decode_samples(*arm))
    service = fit_decode(fits)
    trace, arms = native_arms(holdout)
    if trace in fit_traces.values():
        raise ValueError("Keep the validation run outside the decode cost fit")
    verification = {}
    for name, (_, requests, _) in arms.items():
        result = simulate_decode(requests, service)
        measured = max(row["end_s"] for row in requests.values())
        verification[name] = {**result, "observed_finish_s": measured, "finish_error_s": result["finish_s"] - measured}
    comparisons = {}
    for order in ("b1", "b2"):
        low, high = f"{order}-cap100", f"{order}-cap200"
        if low not in arms or high not in arms:
            continue
        requests, faster = arms[low][1], arms[high][1]
        hypothetical = simulate_decode(requests, service, prefill_http_s={key: row["phases_s"]["P_HTTP"]
                                                                         for key, row in faster.items()})
        comparisons[order] = {"fixed_low_cap_output_and_D_routes": hypothetical,
                              "conditional_advance_s": verification[low]["finish_s"] - hypothetical["finish_s"]}
    return {"decode_service_fit": service, "training_trace_sha256": fit_traces, "holdout_trace_sha256": trace,
            "heldout_validation": verification, "prefill_delivery_sensitivity": comparisons,
            "assumptions": "Fixed natural output and selected D. Hold observed time to D first token before emitting one output token. Gate each successor on its modeled predecessor. Calibrate a shared decode step period from engine counters and streaming lifetimes. The P delivery substitution can include route and cache effects."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("holdout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--calibration", type=Path, action="append", required=True)
    args = parser.parse_args()
    write_json(args.output, analyze(args.calibration, args.holdout))


if __name__ == "__main__":
    main()
