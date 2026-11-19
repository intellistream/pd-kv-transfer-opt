"""Attribute the pre-admission wait without assuming a causal engine stall."""

import argparse
from collections import defaultdict
from pathlib import Path
import re
from statistics import median

from pd_kv_transfer_opt.common import sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union, normalized_proxy_id, read_bundle
from pd_kv_transfer_opt.profiling.analyze_packed_readiness import GET, client_ids
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import get_batches


MS = 1_000_000
CLOCK_TRIM_NS = 42 * MS
LONG_WAIT_MS = 300


def clip_ms(intervals, left, right):
    return interval_union((max(start, left), min(end, right)) for start, end in intervals
                          if max(start, left) < min(end, right)) / MS


def observed_gets(events):
    starts, by_instance = {}, defaultdict(list)
    for row in sorted(events, key=lambda r: r["monotonic_ns"]):
        if row.get("operation") != GET:
            continue
        key = row["instance"], row["call_id"]
        if row["type"] == "store_begin":
            starts[key] = row["monotonic_ns"]
        elif row["type"] == "store_end" and key in starts:
            by_instance[row["instance"]].append((starts.pop(key), row["monotonic_ns"]))
    return by_instance


def request_path(end, proxy, arrival, chunks, prior_batches, gets):
    first, last = chunks[0], chunks[-1]
    begin, response = (proxy[k] for k in ("prefill_begin", "prefill_end"))
    send_estimate = arrival["monotonic_ns"] + begin["unix_ns"] - arrival["unix_ns"]
    admitted = arrival["arrival_ns"]
    last_result_unix = last["unix_ns"] + last["ready_ns"] - last["monotonic_ns"]
    prior = max((right for _, right in prior_batches if right <= admitted), default=None)
    lo, hi = send_estimate + CLOCK_TRIM_NS, admitted - CLOCK_TRIM_NS
    p_http = (response["monotonic_ns"] - begin["monotonic_ns"]) / MS
    queue = (first["dispatch_ns"] - admitted) / MS
    execution = (last["ready_ns"] - first["dispatch_ns"]) / MS
    before = (admitted - send_estimate) / MS
    after = (response["unix_ns"] - last_result_unix) / MS
    residual = p_http - queue - execution
    if abs(residual - before - after) > 0.1:
        raise ValueError(f"Proxy and engine duration accounting disagrees for {end['id']}")
    return {
        "instance": arrival["instance"], "TTFT_ms": (end["first_s"] - end["ready_s"]) * 1000,
        "D_output_tail_ms": (end["end_s"] - end["first_s"]) * 1000,
        "P_HTTP_ms": p_http, "P_queue_ms": queue, "P_dispatch_to_result_ms": execution,
        "P_HTTP_outside_engine_ms": residual, "P_send_to_admission_estimated_ms": before,
        "P_result_to_reply_estimated_ms": after, "P_admission_window_trimmed_ms": max(0, hi - lo) / MS,
        "same_instance_batch_overlap_trimmed_ms": clip_ms(prior_batches, lo, hi),
        "same_instance_GET_overlap_ms": clip_ms(gets, send_estimate, admitted),
        "same_instance_GET_overlap_trimmed_ms": clip_ms(gets, lo, hi),
        "last_same_instance_result_to_admission_ms": (admitted - prior) / MS if prior is not None else None,
    }


def audit(prefill, decode, pattern):
    if prefill["scope"][0]["run_id"] != decode["scope"][0]["run_id"]:
        raise ValueError("Expected paired capture roles")
    ends = {e["id"]: e for e in decode["client"] if e["type"] == "request_end" and pattern.fullmatch(e["case"])}
    if not ends or any(e["status"] != "ok" for e in ends.values()):
        raise ValueError("Expected completed matching formal cases")
    names = client_ids(decode)
    proxies = defaultdict(dict)
    for row in decode["proxy"]:
        if row["id"] in ends:
            proxies[row["id"]][row["type"]] = row
    events = prefill["native"]
    arrivals = {normalized_proxy_id(row["request_id"]): row for row in events if row["type"] == "p_chunk_arrival"}
    _, batches, errors = get_batches(events)
    if errors:
        raise ValueError(f"Incomplete scheduler events: {len(errors)}")
    by_request, case_intervals = defaultdict(list), defaultdict(list)
    for batch in sorted(batches, key=lambda r: r["dispatch_ns"]):
        rids = [normalized_proxy_id(row[0]) for row in batch["rows"]]
        cases = {ends[names[rid]]["case"] for rid in rids if names.get(rid) in ends}
        if not cases:
            continue
        case, = cases
        case_intervals[batch["instance"], case].append((batch["dispatch_ns"], batch["ready_ns"]))
        for rid in rids:
            by_request[rid].append(batch)
    gets = observed_gets(events)
    rows = defaultdict(dict)
    for name, end in sorted(ends.items()):
        proxy = proxies[name]
        rid = proxy["prefill_begin"]["proxy_request_id"]
        arrival = arrivals[rid]
        instance, case = arrival["instance"], end["case"]
        rows[case][name.split("~", 1)[1]] = request_path(
            end, proxy, arrival, by_request[rid], case_intervals[instance, case], gets[instance])

    long_waits = [row for requests in rows.values() for row in requests.values()
                  if row["P_send_to_admission_estimated_ms"] > LONG_WAIT_MS]
    since_result = [row["last_same_instance_result_to_admission_ms"] for row in long_waits
                    if row["last_same_instance_result_to_admission_ms"] is not None]
    return {
        "timed_request_count": len(ends), "cases": dict(rows),
        "estimated_admission_wait_above_300_ms": {
            "requests": len(long_waits),
            "within_10_ms_after_same_instance_batch_result": sum(wait < 10 for wait in since_result),
            "median_ms_after_same_instance_batch_result": median(since_result) if since_result else None,
            "median_trimmed_window_share_inside_batch": median(
                row["same_instance_batch_overlap_trimmed_ms"] / row["P_admission_window_trimmed_ms"]
                for row in long_waits) if long_waits else None,
            "median_untrimmed_window_share_inside_GET": median(
                row["same_instance_GET_overlap_ms"] / row["P_send_to_admission_estimated_ms"]
                for row in long_waits) if long_waits else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("r20_directory", "r21_directory", "output"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    runs = {}
    for stem, directory, pattern in (("r20", args.r20_directory, r"[ab]-b[12]-cap(?:100|200)"),
                                     ("r21", args.r21_directory, r"[abc]-b3-cap(?:100|200)")):
        paths = [directory / f"{stem}-{role}.jsonl.gz" for role in "PD"]
        runs[stem] = audit(*map(read_bundle, paths), re.compile(pattern))
        runs[stem]["source_sha256"] = {role: sha256(path) for role, path in zip("PD", paths, strict=True)}
    write_json(args.output, {
        "scope": "Outside-engine P HTTP is a difference of durations. Its ingress and reply split uses "
                 "cross-host Unix anchors; batch/GET overlaps are temporal. Trimming 42 ms per edge "
                 "is a sensitivity check. Current traces do not distinguish P API work or IPC "
                 "arrival from the time that EngineCore dequeues a waiting ADD. The previous "
                 "batch's result timestamp marks entry to Scheduler.update_from_output.",
        "runs": runs,
    })


if __name__ == "__main__":
    main()
