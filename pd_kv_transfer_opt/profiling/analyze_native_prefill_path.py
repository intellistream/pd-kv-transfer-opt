"""Attribute native long-session P HTTP and calibrate the observed service stages."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution, interval_union
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import get_batches
from pd_kv_transfer_opt.profiling.analyze_kv_lifetime import change_finish, index_events
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, measure_arms, read_trace, requests_for


GET = "batch_get_into_multi_buffers"
PUT = "batch_put_from_multi_buffers"
ADXL_OPERATIONS = {0: GET, 1: PUT}
REQUEST_STAGES = (
    "proxy_and_return_residual_s", "API_before_ADD_s", "ADD_to_enqueue_s", "IPC_queue_s",
    "dequeue_to_schedule_accept_s", "first_schedule_wait_s", "dispatch_to_P_retain_s",
)


def as_interval(row):
    return row["begin_ns"], row["end_ns"]


def clipped_union(spans, window):
    left, right = window
    return interval_union((max(a, left), min(b, right)) for a, b in spans if a < right and b > left)


def overlap_seconds(spans, window):
    return clipped_union(spans, window) / NS


def caller_timing(events):
    """Match the synchronous Python call to lower-level, possibly parallel ADXL work."""
    store_start = {}
    calls = defaultdict(list)
    transport = defaultdict(list)
    connects = defaultdict(list)
    names = defaultdict(dict)
    active_loads = {}
    for event in sorted(events, key=lambda row: row["monotonic_ns"]):
        kind = event["type"]
        pid, tid = event["pid"], event["thread_id"]
        thread = (pid, tid)
        if kind == "adxl_transfer_sync":
            transport[pid, ADXL_OPERATIONS[event["operation"]]].append(event)
        elif kind == "adxl_connect":
            connects[pid].append(event)
        elif kind == "worker_load_begin":
            active_loads[thread] = (event, [])
        elif kind == "store_begin" and event["operation"] in {GET, PUT}:
            store_start[event["call_id"]] = event
        elif kind == "store_end" and event["operation"] in {GET, PUT}:
            operation = event["operation"]
            begin = store_start.pop(event["call_id"])
            native = dict(event, begin_ns=begin["monotonic_ns"], end_ns=event["monotonic_ns"])
            calls[pid, operation].append(native)
            if operation == GET and thread in active_loads:
                active_loads[thread][1].append(native)
        elif kind == "worker_load_end":
            start, loads = active_loads.pop(thread)
            keys = start["request_ids"]
            if len(keys) == len(loads):
                previous = start["monotonic_ns"]
                for req, result in zip(keys, loads, strict=True):
                    result["request"] = req
                    result["before_Store_in_load_s"] = (result["begin_ns"] - previous) / NS
                    names[req].setdefault(start["instance"], []).append(result)
                    previous = result["end_ns"]
    if store_start or active_loads:
        raise ValueError("Incomplete P Store/worker observations")

    orphan = Counter()
    for (pid, operation), low in transport.items():
        ordered = sorted(calls[pid, operation], key=lambda row: row["begin_ns"])
        starts = [row["begin_ns"] for row in ordered]
        for left, right in zip(ordered, ordered[1:]):
            if left["end_ns"] > right["begin_ns"]:
                raise ValueError(f"Cannot infer ADXL parent of overlapping calls to {operation} in PID {pid}")
        for item in low:
            index = bisect_right(starts, item["begin_ns"]) - 1
            if index >= 0 and item["end_ns"] <= ordered[index]["end_ns"]:
                ordered[index].setdefault("adxl_transfer_sync", []).append(item)
            else:
                orphan[operation] += 1
    all_calls = [row for group in calls.values() for row in group]
    candidate_parents = defaultdict(list)
    for call in all_calls:
        for conn in connects[call["pid"]]:
            if call["begin_ns"] <= conn["begin_ns"] <= conn["end_ns"] <= call["end_ns"]:
                candidate_parents[conn["pid"], conn["begin_ns"]].append((call, conn))
    for options in candidate_parents.values():
        if len(options) == 1:
            options[0][0].setdefault("adxl_connect", []).append(options[0][1])
    for call in all_calls:
        lower = call.get("adxl_transfer_sync", [])
        call["ADXL_payload_bytes"] = sum(row["payload_bytes"] for row in lower)
        call["ADXL_service_s"] = overlap_seconds([as_interval(row) for row in lower], as_interval(call))
        call["Store_residual_s"] = (call["end_ns"] - call["begin_ns"]) / NS - call["ADXL_service_s"]
        if lower:
            first = min(row["begin_ns"] for row in lower)
            last = max(row["end_ns"] for row in lower)
            call["Store_before_ADXL_s"] = (first - call["begin_ns"]) / NS
            call["Store_after_ADXL_s"] = (call["end_ns"] - last) / NS
            call["Store_between_ADXL_s"] = (last - first) / NS - call["ADXL_service_s"]
        call["ADXL_connect_s"] = overlap_seconds(
            [as_interval(row) for row in call.get("adxl_connect", [])], as_interval(call))
    return all_calls, names, orphan


def batch_and_request_index(events):
    arrivals, batches, errors = get_batches(events)
    by_request = defaultdict(list)
    events_by_request = defaultdict(dict)
    result_waits = defaultdict(list)
    for batch in batches:
        for key, offset, tokens in batch["rows"]:
            by_request[key].append((batch, offset, tokens))
    for row in events:
        key = row.get("request_id")
        if key is not None:
            events_by_request[key].setdefault(row["type"], row)
        if row["type"] == "p_admission_result_wait":
            result_waits[row["instance"]].append((row["wait_begin_ns"], row["wait_end_ns"]))
    return arrivals, batches, by_request, events_by_request, result_waits, errors


def request_path(profile, request, arrival, scheduled, events, gets):
    accepted = arrival["arrival_ns"]
    http = events.get("p_admission_http", {}).get("http_enter_ns")
    send = events.get("p_admission_ipc_send", {}).get("send_begin_ns")
    enqueue = events.get("p_admission_enqueue", {}).get("event_ns")
    dequeue = events.get("p_admission_dequeue", {}).get("event_ns")
    steps = [batch for batch, _, _ in scheduled]
    first = steps[0]["dispatch_ns"]
    retained = events["p_handoff_retain"]["monotonic_ns"] if "p_handoff_retain" in events else steps[-1]["ready_ns"]
    shell = None
    if None not in (http, send, enqueue, dequeue):
        clocks = [http, send, enqueue, dequeue, accepted, first, retained]
        if any(a > b for a, b in zip(clocks, clocks[1:])):
            raise ValueError(f"P clocks are out of order for {request}")
        local = [(b - a) / NS for a, b in zip(clocks, clocks[1:])]
        http_s = profile["phases_s"]["P_HTTP"]
        shell = dict(zip(REQUEST_STAGES, [http_s - sum(local), *local], strict=True))
        if shell["proxy_and_return_residual_s"] < -0.002:
            raise ValueError(f"Unclosed HTTP wall-clock interval for {request}")
    start = http if http is not None else accepted
    rank0 = [batch.get("device", {}).get("forward_device_ms") for batch in steps]
    compute = sum(value for value in rank0 if value is not None) / 1000
    physical = list(gets.get(request, {}).get(profile["P"], ()))
    per_rank = defaultdict(lambda: defaultdict(float))
    for call in physical:
        rank = per_rank[call["pid"]]
        rank["Store_s"] += (call["end_ns"] - call["begin_ns"]) / NS
        rank["ADXL_service_s"] += call["ADXL_service_s"]
        for name in ("before_Store_in_load_s", "Store_before_ADXL_s", "Store_after_ADXL_s", "Store_between_ADXL_s"):
            rank[name] += call.get(name, 0)
    slowest = max(per_rank.values(), key=lambda values: values["Store_s"], default={})
    prompt = arrival["prompt_tokens"]
    return {
        "P": profile["P"], "D": profile["D"], "request": request, "prompt_tokens": prompt,
        "route_P_HTTP_s": profile["phases_s"]["P_HTTP"], "output_tokens": profile["output_tokens"],
        "P_HTTP_breakdown": shell, "P_scheduler_queue_s": (first - accepted) / NS,
        "P_elapsed_dispatch_to_retain_s": (retained - first) / NS,
        "P_scheduled_batch_union_s": overlap_seconds(
            [(step["dispatch_ns"], step["ready_ns"]) for step in steps], (first, retained)),
        "P_rank0_forward_s": compute, "P_timed_batches": sum(value is not None for value in rank0),
        "P_batches": len(steps), "P_first_computed_prefix": scheduled[0][1],
        "P_computed_prompt_tokens": sum(max(0, min(n, prompt - offset)) for _, offset, n in scheduled),
        "P_direct_GET_calls": len(physical), "P_direct_GET_bytes": sum(row["payload_bytes"] for row in physical),
        "P_direct_ADXL_bytes": sum(row["ADXL_payload_bytes"] for row in physical),
        "P_direct_slowest_rank_Store_s": slowest.get("Store_s", 0),
        "P_direct_slowest_Store_rank": slowest,
        "P_overlay_http_ns": [start, retained], "P_first_schedule_wait_ns": [accepted, first],
        "P_ipc_wait_ns": [enqueue, dequeue] if None not in (enqueue, dequeue) else None,
    }


def device_work_in_window(batches, window, roots):
    """Count timed rank-0 work of fully contained batches on the same worker stream."""
    start, end = window
    contained = [batch for batch in batches if batch["dispatch_ns"] >= start and batch["ready_ns"] <= end]
    device = [batch.get("device", {}).get("forward_device_ms") for batch in contained]
    return {"batches": len(contained), "timed_batches": sum(value is not None for value in device),
            "full_4096_batches": sum(batch["total"] == 4096 for batch in contained),
            "single_request_batches": sum(len(batch["rows"]) == 1 for batch in contained),
            "scheduled_tokens": sum(batch["total"] for batch in contained),
            "root_scheduled_tokens": sum(tokens for batch in contained for key, _, tokens in batch["rows"]
                                         if key in roots),
            "forward_device_s": sum(value for value in device if value is not None) / 1000,
            "dispatch_to_result_union_s": overlap_seconds(
                [(batch["dispatch_ns"], batch["ready_ns"]) for batch in batches], window)}


def batch_inventory(batches):
    groups = defaultdict(list)
    for row in batches:
        history = max(offset for _, offset, _ in row["rows"])
        h = "0-32K" if history < 32768 else ("32K-96K" if history < 98304 else "96K+")
        size = "4096" if row["total"] == 4096 else "under4096"
        group = f"{size}/{'one' if len(row['rows']) == 1 else 'multi'}/{h}"
        device = row.get("device", {})
        values = {"dispatch_to_result_ms": (row["ready_ns"] - row["dispatch_ns"]) / 1e6}
        if device.get("forward_device_ms") is not None:
            values["rank0_forward_ms"] = device["forward_device_ms"]
        if "worker_enter_ns" in device:
            values["dispatch_to_rank0_worker_ms"] = (device["worker_enter_ns"] - row["dispatch_ns"]) / 1e6
            values["rank0_worker_to_forward_ms"] = (device["forward_enter_ns"] - device["worker_enter_ns"]) / 1e6
        groups[group].append(values)
    return {name: {key: distribution(v[key] for v in timings if key in v)
                   for key in sorted({k for v in timings for k in v})}
            for name, timings in sorted(groups.items())}


def store_inventory(calls, operation):
    calls = [row for row in calls if row["operation"] == operation]
    return {
        "calls": len(calls), "with_ADXL": sum(bool(c.get("adxl_transfer_sync")) for c in calls),
        "ADXL_calls": sum(len(c.get("adxl_transfer_sync", [])) for c in calls),
        "ADXL_errors": sum(row["status"] != 0 for c in calls for row in c.get("adxl_transfer_sync", [])),
        "ADXL_byte_matched": sum(c["ADXL_payload_bytes"] == c["payload_bytes"] for c in calls),
        "negative_returns": sum(c["negative_returns"] for c in calls),
        "Store_payload_bytes": sum(c["payload_bytes"] for c in calls),
        "ADXL_payload_bytes": sum(c["ADXL_payload_bytes"] for c in calls),
        "wall_s": distribution((row["end_ns"] - row["begin_ns"]) / NS for row in calls),
        **{key: distribution(row[key] for row in calls if key in row) for key in
           ("before_Store_in_load_s", "ADXL_service_s", "Store_before_ADXL_s", "Store_after_ADXL_s", "Store_between_ADXL_s",
            "Store_residual_s", "ADXL_connect_s", "payload_bytes")},
    }


def decode_progress(arm, profiles, origin):
    d_events = index_events(arm["d_native"])
    stages = {}
    for key, request in profiles.items():
        events = d_events[key]
        arrival, allocation, receipt = (events[f"d_paged_receive_{name}"] for name in ("arrival", "allocated", "ready"))
        if any(row["instance"] != request["D"] for row in (arrival, allocation, receipt)):
            raise ValueError(f"Decode destination disagrees for {key}")
        entry, reserved, ready = (row["monotonic_ns"] / NS - origin for row in (arrival, allocation, receipt))
        if not request["D_start_s"] <= entry <= reserved <= ready <= request["first_s"]:
            raise ValueError(f"D readiness is outside the client first-token window for {key}")
        stages[key] = {"D_KV_ready_s": ready, "D_proxy_to_ingress_s": entry - request["D_start_s"],
                       "D_ingress_to_alloc_s": reserved - entry,
                       "D_alloc_to_KV_ready_s": ready - reserved,
                       "D_KV_ready_to_first_s": request["first_s"] - ready}
    result = {}
    for role in ("D0", "D1"):
        keys = [key for key, row in profiles.items() if row["D"] == role]
        p_pending = [(profiles[key]["P_start_s"], profiles[key]["P_end_s"]) for key in keys]
        route_ready = [stages[key]["D_KV_ready_s"] for key in keys]
        first_ready = min(route_ready)
        availability = sorted((t, delta) for key in keys
                              for t, delta in ((stages[key]["D_KV_ready_s"], 1), (profiles[key]["end_s"], -1)))
        bounds = [(min(a for a, _ in p_pending), 0), *availability]
        count = 0
        idle_with_P_after_first = 0
        for (left, delta), (right, _) in zip(bounds, bounds[1:]):
            count += delta
            if count == 0 and left >= first_ready:
                idle_with_P_after_first += clipped_union(p_pending, (left, right))
        startup = clipped_union(p_pending, (bounds[0][0], first_ready))
        tail = max(keys, key=lambda key: profiles[key]["end_s"])
        last_end = profiles[tail]["end_s"]
        other_end = max((profiles[key]["end_s"] for key in keys if key != tail), default=0)
        result[role] = {"first_KV_ready_s": first_ready, "last_KV_ready_s": max(route_ready),
                        "P_inflight_before_first_KV_ready_s": startup,
                        "no_KV_ready_unfinished_with_P_inflight_s": startup + idle_with_P_after_first,
                        "no_KV_ready_unfinished_with_P_inflight_after_first_s": idle_with_P_after_first,
                        "last_client_end_s": last_end, "last_request": tail,
                        "tail_alone_after_first_token_s": last_end - max(other_end, profiles[tail]["first_s"])}
    return stages, result


def report(arm, manifest, cap_Gbps=100):
    origin, = (row["monotonic_s"] for row in arm["client"] if row["type"] == "client_origin")
    profiles = requests_for(arm, manifest["workload"]["request_sources"], origin)
    roots = {key for key, row in profiles.items() if row["predecessor"] is None}
    calls, gets, orphan = caller_timing(arm["p_native"])
    arrives, batches, scheduled, p_events, wait_futures, errors = batch_and_request_index(arm["p_native"])
    if errors or len(arrives) != len(profiles) or any("ready_ns" not in batch for batch in batches):
        raise ValueError(f"Invalid P batch/HTTP join: {len(arrives)} of {len(profiles)} arrivals, {errors[:3]}")
    detail = {key: request_path(row, key, arrives[key], scheduled[key], p_events[key], gets)
              for key, row in profiles.items()}
    by_p = defaultdict(list)
    spans_by_p = defaultdict(list)
    for batch in batches:
        by_p[batch["instance"]].append(batch)
    for call in calls:
        spans_by_p[call["instance"], call["operation"]].append(as_interval(call))
    for value in detail.values():
        p = value["P"]
        window = value["P_first_schedule_wait_ns"]
        value["P_prior_compute_during_admitted_wait"] = device_work_in_window(by_p[p], window, roots)
        value["P_any_Store_GET_during_admitted_wait_s"] = overlap_seconds(spans_by_p[p, GET], window)
        value["P_any_Store_GET_during_HTTP_s"] = overlap_seconds(spans_by_p[p, GET], value["P_overlay_http_ns"])
        value["P_any_Store_PUT_during_HTTP_s"] = overlap_seconds(spans_by_p[p, PUT], value["P_overlay_http_ns"])
        if value["P_ipc_wait_ns"] is not None:
            value["P_result_wait_during_IPC_s"] = overlap_seconds(wait_futures[p], value["P_ipc_wait_ns"])
    d_stages, supply = decode_progress(arm, profiles, origin)
    for key, value in detail.items():
        value.update({phase: profiles[key][phase] for phase in ("ready_s", "P_start_s", "P_end_s", "D_start_s", "end_s", "first_s")})
        value.update(d_stages[key])
        value["D_to_first_s"] = profiles[key]["phases_s"]["D_to_first"]
        value["first_to_end_s"] = profiles[key]["phases_s"]["first_to_end"]
    elapsed = measure_arms(arm, profiles, origin, cap_Gbps, {"P": 16, "D": 16})
    waits = {key: value["P_any_Store_GET_during_HTTP_s"] for key, value in detail.items()}
    own = {key: value["P_direct_slowest_rank_Store_s"] for key, value in detail.items()}
    admission = {key: sum(value["P_HTTP_breakdown"][name] for name in REQUEST_STAGES[1:5])
                 for key, value in detail.items() if value["P_HTTP_breakdown"] is not None}
    device = {key: value["P_rank0_forward_s"] + value["P_prior_compute_during_admitted_wait"]["forward_device_s"]
              for key, value in detail.items()}
    graph_modes = Counter(row["runtime_mode"] for row in arm["p_native"] if row["type"] == "p_chunk_graph"
                          and row["aclgraph_replays"] > 0)
    return {
        "baseline": elapsed,
        "method_coverage": {"P_HTTP_joined": sum(row["P_HTTP_breakdown"] is not None for row in detail.values()),
                            "P_requests_with_handoff_retain": sum("p_handoff_retain" in row for row in p_events.values()),
                            "Store_calls_attributed_to_requests": sum("request" in row for row in calls),
                            "unmatched_ADXL_calls": dict(orphan),
                            "rank0_forward_timed": sum("forward_device_ms" in row.get("device", {}) and
                                row["device"]["forward_device_ms"] is not None for row in batches),
                            "rank0_batches_with_graph_replay_by_mode": dict(graph_modes),
                            "P_batches": len(batches)},
        "Store_GET": store_inventory(calls, GET), "Store_PUT": store_inventory(calls, PUT),
        "P_HTTP_stage_cost_s": {
            group: {stage: distribution(value["P_HTTP_breakdown"][stage] for key, value in detail.items()
                                         if (key in roots) == root_turn and value["P_HTTP_breakdown"] is not None)
                    for stage in REQUEST_STAGES}
            for group, root_turn in (("roots", True), ("successors", False))
        },
        "prefill_batch_cost": batch_inventory(batches), "D_progress": supply,
        "D_stage_cost_s": {name: distribution(row[name] for row in d_stages.values())
                           for name in ("D_proxy_to_ingress_s", "D_ingress_to_alloc_s",
                                        "D_alloc_to_KV_ready_s", "D_KV_ready_to_first_s")},
        "conditional_Store_windows": {
            "zero_observed_Store_occupancy_overlapping_each_P_HTTP": change_finish(profiles, waits, "P_HTTP"),
            "zero_observed_direct_per_request_slowest_rank_Store": change_finish(profiles, own, "P_HTTP"),
        },
        "conditional_P_windows": {
            "zero_API_ADD_and_IPC_through_scheduler_arrival": change_finish(profiles, admission, "P_HTTP"),
            "ten_percent_rank0_forward_cost_in_owned_batches": change_finish(
                profiles, {key: value["P_rank0_forward_s"] * 0.1 for key, value in detail.items()}, "P_HTTP"),
            "ten_percent_rank0_forward_cost_in_owned_and_contained_wait_batches":
                change_finish(profiles, {key: value * 0.1 for key, value in device.items()}, "P_HTTP"),
        },
        "conditional_D_windows": {
            "zero_D_alloc_to_KV_ready": change_finish(
                profiles, {key: value["D_alloc_to_KV_ready_s"] for key, value in detail.items()}, "D_to_first"),
            "ten_percent_first_to_end": change_finish(
                profiles, {key: value["first_to_end_s"] * 0.1 for key, value in detail.items()}, "first_to_end"),
        },
        "per_request": detail,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    digest = sha256(args.trace)
    if manifest["files"][args.trace.name]["sha256"] != digest:
        raise ValueError("Profile trace does not match the manifest")
    arm, = read_trace(args.trace).values()
    write_json(args.output, {"trace_sha256": digest, **report(arm, manifest)})


if __name__ == "__main__":
    main()
