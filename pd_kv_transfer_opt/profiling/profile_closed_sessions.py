"""Profile native cap-pair events and replay the observed session dependencies."""

import argparse
from collections import Counter, defaultdict
from gzip import open as gzip_open
from itertools import combinations, pairwise
import json
from math import ceil, comb, floor, isclose, log
from pathlib import Path

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_multiwave import aligned_tx_intervals
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union


PHASES = ("ready_to_P", "P_HTTP", "P_return_to_D", "D_to_first", "first_to_end")
PROXY_PHASES = ("prefill_begin", "prefill_end", "decode_begin")
ROLES = ("P0", "P1", "D0", "D1")
NS = 1_000_000_000


def read_trace(path):
    arms = defaultdict(lambda: defaultdict(list))
    with gzip_open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            arms[record["case"]][record["kind"]].append(record["data"])
    return arms


def requests_for(arm, workload, origin_s):
    ends = {row["id"]: row for row in arm["client"] if row["type"] == "request_end"}
    starts = defaultdict(dict)
    for event in arm["proxy"]:
        kind = event["type"]
        if kind not in PROXY_PHASES:
            continue
        row = starts[event["id"]]
        if kind in row:
            raise ValueError(f"Duplicate {kind} for {event['id']}")
        row[kind] = event
    if ends.keys() != starts.keys() or ends.keys() != {row["id"] for row in workload}:
        raise ValueError("Incomplete client or proxy coverage")

    profiles = {}
    for spec in workload:
        key = spec["id"]
        end = ends[key]
        if (end["status"] != "ok" or not end["done_signal"] or not end["all_content_chunks_have_token_ids"]
                or not 0 < end["completion_tokens"] == end["streamed_token_ids"] <= spec["max_tokens"]
                or any(end[field] != spec[field] for field in ("offered_s", "session_id", "predecessor", "prompt_tokens"))):
            raise ValueError(f"Invalid replay receipt for {key}")
        proxy = starts[key]
        if proxy["prefill_begin"]["input_tokens"] != spec["prompt_tokens"]:
            raise ValueError(f"Prefill prompt length changed for {key}")
        p_start, p_end, d_start = (proxy[name]["monotonic_ns"] / NS - origin_s for name in PROXY_PHASES)
        times = (end["ready_s"], p_start, p_end, d_start, end["first_s"], end["end_s"])
        if not times[0] <= end["dispatch_s"] <= p_start or any(right < left for left, right in pairwise(times)):
            raise ValueError(f"Invalid same-host event order for {key}")
        profiles[key] = {
            "session_id": spec["session_id"], "predecessor": spec["predecessor"],
            "offered_s": spec["offered_s"], "tool_gap_s": spec["tool_gap_s"],
            "ready_s": end["ready_s"], "dispatch_s": end["dispatch_s"],
            "P_start_s": p_start, "P_end_s": p_end, "D_start_s": d_start,
            "first_s": end["first_s"], "end_s": end["end_s"],
            "P": proxy["prefill_begin"]["role"], "D": proxy["decode_begin"]["role"],
            "output_tokens": end["completion_tokens"], "finish_reason": end["finish_reason"],
            "effective_TPOT_ms": 1e3 * (end["end_s"] - end["first_s"]) / (end["completion_tokens"] - 1)
            if end["completion_tokens"] > 1 else None,
            "phases_s": dict(zip(PHASES, (right - left for left, right in pairwise(times)), strict=True)),
        }
    return profiles


def replay(profiles, durations):
    """Hold residence times fixed and propagate releases within each session."""
    ready, done, previous = {}, {}, {}
    for key, row in profiles.items():
        prior, offered = row["predecessor"], row["offered_s"]
        release = done[prior] + row["tool_gap_s"] if prior is not None else offered
        ready[key] = max(offered, release)
        done[key] = ready[key] + sum(durations[key].values())
        previous[key] = prior if release >= offered else None
    tail = max(done, key=done.get)
    chain = []
    current = tail
    while current is not None:
        chain.append(current)
        current = previous[current]
    return ready, done, chain[::-1]


def phase_times(profiles):
    return {key: row["phases_s"] for key, row in profiles.items()}


def describe(values):
    values = sorted(values)
    return {"p50": percentile(values, .5), "p95": percentile(values, .95), "max": max(values)}


def check_observed(profiles):
    costs = phase_times(profiles)
    ready, done, chain = replay(profiles, costs)
    error = max(abs(predicted[key] - row[observed])
                for predicted, observed in ((ready, "ready_s"), (done, "end_s"))
                for key, row in profiles.items())
    if error > 1e-6:
        raise ValueError(f"Observed stage/release round-trip error {error:.6f} s")
    drain = done[chain[-1]]
    sensitivity = {}
    for scale in (0.9, 0.5, 0.0):
        adjusted = {key: {**cost, "P_HTTP": cost["P_HTTP"] * scale} for key, cost in costs.items()}
        _, predicted, _ = replay(profiles, adjusted)
        finish = max(predicted.values())
        sensitivity[f"{scale:g}x"] = {"finish_s": finish, "save_s": drain - finish}
    return {
        "max_error_s": error, "tail": chain[-1], "tail_chain": chain,
        "tail_chain_phase_sum_s": {phase: sum(costs[key][phase] for key in chain) for phase in PHASES},
        "tail_chain_root_offer_s": profiles[chain[0]]["offered_s"],
        "tail_chain_tool_wait_s": sum(profiles[key]["tool_gap_s"] for key in chain[1:]),
        "next_session_end_s": max(row["end_s"] for row in profiles.values()
                                  if row["session_id"] != profiles[chain[-1]]["session_id"]),
        "P_HTTP_sensitivity": sensitivity,
    }


def gauge(values, name):
    readings = [value for key, value in values.items() if key.split("{", 1)[0] == "vllm:" + name]
    return sum(readings) if readings else None


def measure_ports(samples, origin_ns, stop_s, cap):
    intervals = [(max(left, 0), min(right, stop_s), rate, per_port)
                 for left, right, rate, per_port in aligned_tx_intervals(samples, origin_ns)
                 if max(left, 0) < min(right, stop_s)]
    aggregate_hot, any_hot = [], []
    by_second = defaultdict(float)
    for start, end, rate, rates in intervals:
        if rate >= .85 * cap:
            aggregate_hot.append((start, end))
        if max(rates) >= .85 * cap / len(rates):
            any_hot.append((start, end))
        for second in range(floor(start), min(ceil(end), floor(stop_s))):
            by_second[second] += rate * max(0, min(end, second + 1) - max(start, second))
    return intervals, aggregate_hot, any_hot, max(by_second.values())


def measure_queue(samples, profiles, origin_ns, stop_s, aggregate_hot, any_hot, slots):
    observations = defaultdict(list)
    for sample in samples:
        for row in sample.get("engines", []):
            values = row.get("values", {})
            running = gauge(values, "num_requests_running")
            waiting = gauge(values, "num_requests_waiting")
            occupancy = gauge(values, "kv_cache_usage_perc")
            if running is not None and waiting is not None:
                observations[row["role"]].append(((row["monotonic_ns"] - origin_ns) / NS, running, waiting, occupancy))

    result = {}
    for role in ROLES:
        points = observations[role]
        pending = [(row["P_start_s"], row["P_end_s"]) for row in profiles.values() if row["D"] == role]
        empty_with_P = []
        summary = {"covered_s": 0.0, "running_request_s": 0.0, "waiting_request_s": 0.0,
                   "full_slots_s": 0.0, "peak_kv_fraction": max(
                       (kv for _, _, _, kv in points if kv is not None), default=None),
                   "peak_waiting": max(wait for _, _, wait, _ in points)}
        for (left, running, wait, _), (right, _, _, _) in pairwise(points):
            if right - left > 6:
                continue
            start, stop = max(0, left), min(stop_s, right)
            if start >= stop:
                continue
            width = stop - start
            summary["covered_s"] += width
            summary["running_request_s"] += width * running
            summary["waiting_request_s"] += width * wait
            if running >= slots[role[0]]:
                summary["full_slots_s"] += width
            if running == wait == 0:
                empty_with_P.extend((max(start, begin), min(stop, end)) for begin, end in pending
                                    if max(start, begin) < min(stop, end))

        def coincident(hot):
            return interval_union((max(l1, l2), min(r1, r2))
                                  for l1, r1 in empty_with_P for l2, r2 in hot if l1 < r2 and l2 < r1)

        summary["mean_running"] = summary["running_request_s"] / summary["covered_s"]
        if role.startswith("D"):
            summary.update(no_D_work_with_P_outstanding_s=interval_union(empty_with_P),
                           overlap_with_aggregate_hot_s=coincident(aggregate_hot),
                           overlap_with_any_hot_port_s=coincident(any_hot))
        result[role] = summary
    return result


def measure_arms(arm, profiles, origin_s, cap, slots):
    stop = max(row["end_s"] for row in profiles.values())
    t0 = min(row["offered_s"] for row in profiles.values())
    output = sum(row["output_tokens"] for row in profiles.values())
    bins, = arm["client/output-bins.json"]
    if sum(row["n_tokens"] for row in bins) != output:
        raise ValueError("Stream counts do not match completed output usage")
    origin_ns = round(origin_s * NS)
    samples = arm["telemetry"]
    intervals, aggregate_hot, any_hot, peak = measure_ports(samples, origin_ns, stop, cap)
    last_P = max(row["P_end_s"] for row in profiles.values())
    windows = {}
    for begin, end in ((0, 10), (10, 30), (30, 90), (90, floor(last_P)), (ceil(last_P), floor(stop))):
        if end <= begin:
            continue
        seconds = end - begin
        windows[f"{begin}-{end}"] = {
            "D_MAC_TX_Gbps": sum(max(0.0, min(end, right) - max(begin, left)) * rate
                                  for left, right, rate, _ in intervals) / seconds,
            "P_returns": sum(begin <= row["P_end_s"] < end for row in profiles.values()),
            "client_output_tokens_s": sum(row["n_tokens"] for row in bins if begin <= row["at_s"] < end) / seconds,
        }
    route_outcomes = {}
    for role in ("D0", "D1"):
        group = [row for row in profiles.values() if row["D"] == role]
        route_outcomes[role] = {"requests": len(group), "tokens": sum(row["output_tokens"] for row in group),
                                "last_end_s": max(row["end_s"] for row in group)}
    phase_rows = {"roots": [], "successors": []}
    for row in profiles.values():
        label = "roots" if row["predecessor"] is None else "successors"
        phase_rows[label].append(row["phases_s"])
    return {
        "request_count": len(profiles), "output_tokens": output, "duration_s": stop - t0,
        "throughput_tokens_s": output / (stop - t0),
        "last_P_reply_s": last_P, "last_D_start_s": max(row["D_start_s"] for row in profiles.values()),
        "after_last_P_reply_s": stop - last_P, "next_request_end_s": sorted(row["end_s"] for row in profiles.values())[-2],
        "P_D_resource_samples": measure_queue(samples, profiles, origin_ns, stop, aggregate_hot, any_hot, slots),
        "D_TX_GB": sum((end - begin) * rate / 8 for begin, end, rate, _ in intervals),
        "D_TX_peak_1s_Gbps": peak, "D_aggregate_above_85pct_s": interval_union(aggregate_hot),
        "D_any_port_above_85pct_s": interval_union(any_hot),
        "route_outcomes": route_outcomes,
        "windows": windows,
        "phases": {label: {name: describe([row[name] for row in rows]) for name in PHASES} if rows else {}
                   for label, rows in phase_rows.items()},
        "observed_dependencies": check_observed(profiles),
    }


def compare_arms(low, high, low_report, high_report):
    """Symmetric stage swaps on the observed precedence graph."""
    mixed_s = {}
    contributions = Counter()
    phase_count = len(PHASES)
    for size in range(phase_count + 1):
        for stages in combinations(PHASES, size):
            mask = frozenset(stages)
            costs = {key: {phase: (high if phase in mask else low)[key]["phases_s"][phase] for phase in PHASES}
                     for key in low}
            _, ends, chain = replay(low, costs)
            mixed_s[mask] = (ends[chain[-1]], chain[-1])
            for phase in mask:
                weight = 1 / (phase_count * comb(phase_count - 1, size - 1))
                contributions[phase] += weight * (mixed_s[mask - {phase}][0] - ends[chain[-1]])

    time_low, time_high = low_report["duration_s"], high_report["duration_s"]
    volume_log = log(high_report["output_tokens"] / low_report["output_tokens"])
    time_log = log(time_low / time_high)
    if not isclose(sum(contributions.values()), time_low - time_high, abs_tol=1e-6):
        raise ValueError("Phase swaps do not reconcile with measured elapsed time")

    same_route = [key for key in low if all(low[key][role] == high[key][role] for role in ("P", "D"))]
    same_length = sum(low[key]["output_tokens"] == high[key]["output_tokens"] for key in low)
    return {
        "log_throughput_ratio": volume_log + time_log, "log_volume_term": volume_log, "log_elapsed_term": time_log,
        "elapsed_100_minus_200_s": time_low - time_high, "stage_swap_attribution_s": dict(contributions),
        "only_P_HTTP_swapped_from_200_into_100": dict(zip(("finish_s", "tail"), mixed_s[frozenset({"P_HTTP"})])),
        "only_decode_span_swapped_from_200_into_100": dict(zip(("finish_s", "tail"), mixed_s[frozenset({"first_to_end"})])),
        "matched_route_requests": len(same_route), "matched_output_lengths": same_length,
        "matched_route_but_different_output_length": sum(low[key]["output_tokens"] != high[key]["output_tokens"] for key in same_route),
        "largest_token_volume_changes": [
            {"id": key, "tokens_100": low[key]["output_tokens"], "tokens_200": high[key]["output_tokens"]}
            for key in sorted(low, key=lambda key: abs(high[key]["output_tokens"] - low[key]["output_tokens"]), reverse=True)[:5]
        ],
    }


def handoff_windows(summary):
    """Measure published KV awaiting D enqueue from the historical method traces."""
    cases = {}
    for name, arm in summary["cases"].items():
        requests = arm["per_request"]
        waits = [row["D_transfer_enqueued_s"] - row["D_begin_s"] for row in requests.values()]
        if min(waits) < 0:
            raise ValueError(f"{name} contains invalid handoff timestamps")
        byte_wait = sum(row["P_to_D_bytes"] * wait for row, wait in zip(requests.values(), waits, strict=True)) / 1e9
        removable = {
            f"{lead:g}s_reserve": sum(row["P_to_D_bytes"] * max(0.0, wait - lead)
                                     for row, wait in zip(requests.values(), waits, strict=True)) / 1e9
            for lead in (1.0, 5.0, 30.0)
        }
        tail_id = max(requests, key=lambda key: requests[key]["client_end_s"])
        tail = requests[tail_id]
        latest = {
            "id": tail_id, "P_HTTP_return_s": tail["P_HTTP_return_s"], "D_begin_s": tail["D_begin_s"],
            "D_transfer_enqueued_s": tail["D_transfer_enqueued_s"], "D_KV_ready_s": tail["D_KV_ready_s"],
            "first_s": tail["client_first_s"], "end_s": tail["client_end_s"], "output_tokens": tail["output_tokens"],
        }
        cases[name] = {"requests": len(requests), "D_begin_to_enqueue_s": describe(waits),
                       "requests_with_wait_above_1s": sum(wait > 1 for wait in waits),
                       "requests_with_wait_above_5s": sum(wait > 5 for wait in waits),
                       "payload_weighted_D_admission_wait_GB_s": byte_wait,
                       "post_hoc_payload_wait_slack_GB_s": removable, "last_request": latest}
    return {"case_observations": cases, "method_source_sha256": summary["source_sha256"]}


def analyze(directory, handoff_summaries=()):
    manifest = read_json(directory / "manifest.json")
    trace_name = manifest.get("event_trace", "r28-events.jsonl.gz")
    if Path(trace_name).name != trace_name:
        raise ValueError("The event trace must be a file in the archive directory")
    trace = directory / trace_name
    digest = sha256(trace)
    if digest != manifest["artifacts"][trace.name]["sha256"]:
        raise ValueError("The trace does not match its archive manifest")
    baseline = read_json(directory / "summary.json")["arms"]
    slots = {role: spec["max_num_seqs"] for role, spec in manifest["settings"]["engines"].items()}
    workload = manifest["workload"]["request_sources"]
    raw_arms = read_trace(trace)
    profiles, measured = {}, {}
    for name in manifest["case_order"]:
        arm = raw_arms[name]
        origin, = (row["monotonic_s"] for row in arm["client"] if row["type"] == "client_origin")
        profiles[name] = requests_for(arm, workload, origin)
        measured[name] = measure_arms(arm, profiles[name], origin, baseline[name]["cap_Gbps"], slots)
        observed = measured[name]
        prior = baseline[name]["total"]
        if not (observed["output_tokens"] == prior["output_tokens"]
                and isclose(observed["throughput_tokens_s"], prior["output_tokens_per_s"], rel_tol=1e-12)):
            raise ValueError(f"{name} disagrees with the recorded drained throughput")

    pairs = {}
    for block in ("b1", "b2"):
        low, high = (f"{block}-cap{cap}" for cap in (100, 200))
        if low in profiles and high in profiles:
            pairs[block] = compare_arms(profiles[low], profiles[high], measured[low], measured[high])
    tail_ids = {name: row["observed_dependencies"]["tail"] for name, row in measured.items()}
    for name, finals in profiles.items():
        selected = set(tail_ids.values()) | set(measured[name]["observed_dependencies"]["tail_chain"])
        measured[name]["selected_request_profiles"] = {key: row for key, row in finals.items() if key in selected}
    report = {"trace_sha256": digest, "case_order": manifest["case_order"], "arms": measured, "pairs": pairs}
    if handoff_summaries:
        reports = [read_json(path) for path in handoff_summaries]
        report["historical_handoff"] = {value["run_id"]: handoff_windows(value) for value in reports}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Native cap-pair archive directory")
    parser.add_argument("output", type=Path, help="Structured analysis JSON")
    parser.add_argument("--handoff-summary", action="append", type=Path, default=[],
                        help="Optional analyze_multiwave summary with complete D transfer method events")
    args = parser.parse_args()
    write_json(args.output, analyze(args.directory, args.handoff_summary))


if __name__ == "__main__":
    main()
