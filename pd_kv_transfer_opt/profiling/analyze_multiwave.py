"""Measure P deliveries, D supply waits, and token flow in identical replay windows."""

import argparse
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from statistics import mean, pstdev

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_full_rdma import associate_loads, associate_transfers, find_methods_and_batches
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union, normalized_proxy_id, read_bundle
from pd_kv_transfer_opt.profiling.analyze_natural_restore_ab import SETTINGS, match_targets, route
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import stats
from pd_kv_transfer_opt.profiling.analyze_same_head_ab import cap_readback, port_deltas
from pd_kv_transfer_opt.workloads.pressure_plan import expand


ROUTES = ("D0", "D1")
GET = "batch_get_into_multi_buffers"
NS = 1_000_000_000


def prefill_gets(events):
    started, reads = {}, defaultdict(list)
    for row in sorted(events, key=lambda event: event["monotonic_ns"]):
        if row.get("operation") != GET:
            continue
        key = row["pid"], row["call_id"]
        if row["type"] == "store_begin":
            started[key] = row
        elif row["type"] == "store_end":
            begin = started.pop(key)
            rid = normalized_proxy_id(row.get("request_id") or begin.get("request_id"))
            if rid:
                if begin["payload_bytes"] != row["payload_bytes"] or row["negative_returns"]:
                    raise ValueError("Invalid target GET receipt")
                reads[rid].append((begin["monotonic_ns"], row["monotonic_ns"], row["payload_bytes"]))
    if started:
        raise ValueError("Incomplete P GET capture")
    return reads


def native_prefill_gets(events):
    requests, loads, mismatches = associate_loads(events)
    if mismatches:
        raise ValueError(f"Native synchronous GETs cannot be assigned by load order: {mismatches[:3]}")
    reads = defaultdict(list)
    for identifier, calls in requests.items():
        for call in calls:
            if call["negative_returns"]:
                raise ValueError(f"Failed native GET for {identifier}")
            reads[identifier].append((call["begin_ns"], call["end_ns"], call["payload_bytes"]))
    return reads, loads


def make_request(end, proxy, target, origin, ranks, p_arrival, p_chunks, p_lookup, gets):
    if len(ranks) != 4 or len({row["pid"] for row in ranks}) != 4:
        raise ValueError(f"Incomplete P→D transfer for {end['id']}")
    p_begin, p_end, d_begin, d_first, d_end = (
        (proxy[k]["monotonic_ns"] - origin) / NS
        for k in ("prefill_begin", "prefill_end", "decode_begin", "decode_first_chunk", "decode_end")
    )
    enqueue = (min(row["enqueue_ns"] for row in ranks) - origin) / NS
    kv_ready = (max(row["end_ns"] for row in ranks) - origin) / NS
    if not p_begin <= p_end <= d_begin <= enqueue <= kv_ready <= d_first <= d_end:
        raise ValueError(f"Unexpected transfer event order for {end['id']}")
    first, last = p_chunks[0], p_chunks[-1]
    cache = [row for row in p_lookup if row["monotonic_ns"] <= first["dispatch_ns"]]
    if not cache:
        raise ValueError(f"No target prefix lookup for {end['id']}")
    cache = cache[-1]
    return {
        "wave": target["wave"], "prefill": target["prefill"], "decode": target["decode"],
        "offered_s": end["offered_s"], "client_dispatch_s": end["dispatch_s"],
        "client_first_s": end["first_s"], "client_end_s": end["end_s"],
        "P_begin_s": p_begin, "P_HTTP_return_s": p_end, "D_begin_s": d_begin,
        "D_transfer_enqueued_s": enqueue, "D_KV_ready_s": kv_ready, "D_first_chunk_s": d_first, "D_end_s": d_end,
        "input_tokens": end["prompt_tokens"], "output_tokens": end["completion_tokens"],
        "P_local_prefix_tokens": cache["local_tokens"], "P_external_prefix_tokens": cache["external_tokens"],
        "P_queue_ms": (first["dispatch_ns"] - p_arrival["arrival_ns"]) / 1e6,
        "P_first_dispatch_to_last_result_ms": (last["ready_ns"] - first["dispatch_ns"]) / 1e6,
        "P_GET_calls": len(gets), "P_GET_bytes": sum(read[2] for read in gets),
        "P_GET_max_call_ms": max(((stop - start) / 1e6 for start, stop, _ in gets), default=0),
        "P_to_D_bytes": sum(row["payload_bytes"] for row in ranks),
    }


def supply_time(requests, start, stop, decode_slots=None):
    """Time with P backlog and zero handed-off or KV-ready work, on D's clock."""
    events = defaultdict(lambda: [0, 0, 0])
    for row in requests:
        for slot, enter, leave in ((0, "P_begin_s", "P_HTTP_return_s"),
                                   (1, "P_HTTP_return_s", "D_end_s"),
                                   (2, "D_KV_ready_s", "D_end_s")):
            left, right = max(start, row[enter]), min(stop, row[leave])
            if left < right:
                events[left][slot] += 1
                events[right][slot] -= 1
    p, handed, ready = 0, 0, 0
    previous = start
    seconds = Counter()
    for now, (dp, dh, dr) in sorted(events.items()):
        width = now - previous
        if p:
            seconds["P_pending"] += width
            if not handed:
                seconds["empty_D_while_P_pending"] += width
            if decode_slots is not None and ready < decode_slots:
                seconds["D_below_slots_while_P_pending"] += width
                if handed < decode_slots:
                    seconds["D_handoff_below_slots_while_P_pending"] += width
        if handed and not ready:
            seconds["handed_off_but_no_D_KV_ready"] += width
        if ready:
            seconds["D_KV_ready_present"] += width
        seconds["D_KV_ready_request_seconds"] += width * ready
        p, handed, ready = p + dp, handed + dh, ready + dr
        previous = now
    if (p, handed, ready) != (0, 0, 0):
        raise ValueError("Unbalanced P/D stage lifetimes")
    result = {
        "P_pending_s": seconds["P_pending"],
        "empty_D_while_P_pending_s": seconds["empty_D_while_P_pending"],
        "handed_off_but_no_D_KV_ready_s": seconds["handed_off_but_no_D_KV_ready"],
        "no_D_KV_ready_s": max(0, stop - start - seconds["D_KV_ready_present"]),
        "mean_KV_ready_unfinished_requests": seconds["D_KV_ready_request_seconds"] / (stop - start),
    }
    if decode_slots is not None:
        result["D_below_slots_while_P_pending_s"] = seconds["D_below_slots_while_P_pending"]
        result["D_handoff_below_slots_while_P_pending_s"] = seconds["D_handoff_below_slots_while_P_pending"]
    return result


def sampled_load(samples, start, stop):
    weights = Counter()
    for before, after in zip(samples, samples[1:]):
        elapsed = after["time_s"] - before["time_s"]
        left, right = max(start, before["time_s"]), min(stop, after["time_s"])
        if right <= left or not 0 < elapsed <= 1.5:
            continue
        width = right - left
        weights["covered_s"] += width
        for key in ("running", "waiting", "kv_usage"):
            if key in before:
                weights[key] += width * before[key]
                weights[f"covered_{key}_s"] += width
        if "generation_tokens" in before and "generation_tokens" in after:
            delta = after["generation_tokens"] - before["generation_tokens"]
            if delta < 0:
                raise ValueError("D generation counter reset within an arm")
            weights["generation_tokens"] += width / elapsed * delta
            weights["covered_generation_tokens_s"] += width
    return {"sample_coverage_s": weights["covered_s"],
            "D_generated_tokens_per_s_estimate": weights["generation_tokens"] / weights["covered_generation_tokens_s"]
            if weights["covered_generation_tokens_s"] else None,
            **{f"sampled_mean_{key}": weights[key] / weights[f"covered_{key}_s"]
               for key in ("running", "waiting", "kv_usage") if weights[f"covered_{key}_s"]}}


def window_metrics(requests, bins, samples, start, stop, decode_slots=None):
    result = {}
    length = stop - start
    for name in ROUTES:
        active = [row for row in requests if row["decode"] == name]
        result[name] = {
            "P_HTTP_returns": sum(start <= row["P_HTTP_return_s"] < stop for row in active),
            "D_KV_ready": sum(start <= row["D_KV_ready_s"] < stop for row in active),
            "P_to_D_KV_ready_bytes": sum(row["P_to_D_bytes"] for row in active if start <= row["D_KV_ready_s"] < stop),
            "completed": sum(start <= row["client_end_s"] < stop for row in active),
            "streamed_tokens": sum(row["tokens"] for row in bins if row["route"] == name and start <= row["time_s"] < stop),
            **supply_time(active, start, stop, decode_slots),
            **sampled_load(samples[name], start, stop),
        }
    summed = ("P_HTTP_returns", "D_KV_ready", "P_to_D_KV_ready_bytes", "completed", "streamed_tokens")
    rate = {key + "_per_s": sum(row[key] for row in result.values()) / length for key in summed}
    return {"start_s": start, "end_s": stop, **rate, "D": result}


def aligned_tx_intervals(samples, origin):
    """Align per-port counter deltas on the D host's monotonic clock."""
    if len(samples) < 2:
        return
    start = max(port["monotonic_ns"] for port in samples[0]["ports"])
    stop = min(port["monotonic_ns"] for port in samples[-1]["ports"])
    if start >= stop:
        return

    boundary_rates = []
    for port, points in enumerate(zip(*(sample["ports"] for sample in samples), strict=True)):
        for before, after in pairwise(points):
            left, right = before["monotonic_ns"], after["monotonic_ns"]
            tx_bytes = after["counters"]["mac_tx_total_oct_num"] - before["counters"]["mac_tx_total_oct_num"]
            if left >= right or tx_bytes < 0 or before["port"] != after["port"]:
                raise ValueError("Invalid D port counter sequence")
            if left < stop and right > start:
                boundary_rates.append((max(start, left), port, 8 * tx_bytes / (right - left)))

    rates = [0.0] * len(samples[0]["ports"])
    previous = start
    for time, port, rate in sorted(boundary_rates):
        if previous < time:
            yield (previous - origin) / NS, (time - origin) / NS, sum(rates), tuple(rates)
        previous = time
        rates[port] = rate
    if previous < stop:
        yield (previous - origin) / NS, (stop - origin) / NS, sum(rates), tuple(rates)


def cap_pressure(intervals, stop, cap):
    """Measure observed cap usage from the first offer through the last P reply."""
    seconds = Counter()
    for left, right, rate, _ in intervals:
        begin = max(0, left)
        end = min(stop, right)
        if end > begin:
            seconds["covered"] += end - begin
            if rate >= 0.85 * cap:
                seconds["near_cap"] += end - begin
    return {"D_cap_measurement_window_s": seconds["covered"],
            "D_MAC_TX_at_least_85pct_cap_s": seconds["near_cap"]}


def sustained_windows(paths, bins, intervals, cap, windows, plan):
    """Measure pressure and qualify the prespecified output window."""
    for window in windows.values():
        start, stop = window["start_s"], window["end_s"]
        port_time, time_at_cap, covered, traffic = 0.0, 0.0, 0.0, 0.0
        for left, right, total, ports in intervals:
            width = min(right, stop) - max(left, start)
            if width <= 0:
                continue
            covered += width
            traffic += width * total
            time_at_cap += width * (total >= 0.85 * cap)
            port_time += width * sum(rate >= 0.85 * cap / len(ports) for rate in ports)
        active_P = [(max(row["P_begin_s"], start), min(row["P_HTTP_return_s"], stop)) for row in paths]
        active_P = [(left, right) for left, right in active_P if left < right]
        window.update(planned_offers=sum(start <= row["offered_s"] < stop for row in paths),
                      client_dispatches=sum(start <= row["client_dispatch_s"] < stop for row in paths),
                      D_MAC_TX_coverage_s=covered, D_mean_MAC_TX_Gbps=traffic / covered if covered else None,
                      D_MAC_TX_at_least_85pct_cap_s=time_at_cap, D_ports_at_least_85pct_cap_port_seconds=port_time,
                      P_inflight_present_s=interval_union(active_P),
                      mean_P_inflight_requests=sum(right - left for left, right in active_P) / (stop - start),
                      P_GET_max_call_ms_for_P_replies=stats(row["P_GET_max_call_ms"] for row in paths
                                                           if start <= row["P_HTTP_return_s"] < stop))

    policy = plan["steady_output"]
    primary = windows[policy["window"]]
    start, stop = primary["start_s"], primary["end_s"]
    step = policy["bin_s"]
    if not (0 < step <= stop - start) or (stop - start) % step:
        raise ValueError("The fixed steady window must contain whole output bins.")
    subwindows = []
    for index in range(int((stop - start) / step)):
        left = start + index * step
        count = sum(row["tokens"] for row in bins if left <= row["time_s"] < left + step)
        subwindows.append({"start_s": left, "end_s": left + step, "tokens_per_s": count / step})
    values = [row["tokens_per_s"] for row in subwindows]
    midpoint = (start + stop) / 2
    half = [sum(row["tokens"] for row in bins if left <= row["time_s"] < right)
            for left, right in ((start, midpoint), (midpoint, stop))]
    variations = pstdev(values) / mean(values) if any(values) else None
    drift = abs(half[1] - half[0]) / half[0] if half[0] else None
    under_arrivals = plan["analysis_windows_s"]["arrival"][0] <= start < stop <= plan["analysis_windows_s"]["arrival"][1]
    qualified = (under_arrivals and variations is not None and variations <= policy["max_rate_cv"]
                 and drift is not None and drift <= policy["max_half_rate_change"])
    return {"window": policy["window"], "bins": subwindows, "rate_cv": variations, "half_rate_change": drift,
            "under_live_arrivals": under_arrivals, "qualifies_as_output_plateau": qualified}


def per_case(decode, case, manifest, plan, origin, p_index, d_index):
    target = {row["id"]: {**row, "max_tokens": row["output_tokens"]} for row in plan["target_requests"]}
    mode = plan["cases"][case]["mode"]
    if (manifest["workload_sha256"] != plan["source_workload_sha256"] or manifest["arm_mode"] != mode
            or manifest["background_traffic"] or not manifest["ignore_eos"]):
        raise ValueError(f"Unexpected measurement fixture for {case}")
    requests, proxy = match_targets(decode, manifest, target)
    for name, row in requests.items():
        if row["prompt_tokens"] != target[name]["prompt_tokens"] or row["dispatch_s"] < row["offered_s"]:
            raise ValueError(f"Unexpected input or arrival for {row['id']}")
    cap = plan["cases"][case]["cap"]
    if manifest["aggregate_D_egress_cap_Gbps"] != cap:
        raise ValueError(f"Unexpected bandwidth label for {case}")
    readback = cap_readback(decode, case, cap, range(8))
    if readback != {str(port): cap * 125 for port in range(8)}:
        raise ValueError(f"Missing or uneven shaper readback for {case}")

    captured = {row["name"]: row["data"] for row in decode["cap_control"]}
    if not captured[f"{case}-restore.json"]["restored"]:
        raise ValueError(f"Unrestored shaper settings in {case}")
    starts, batches, lookups, reads, loads = p_index
    paths, p_spans = {}, defaultdict(list)
    for name, end in requests.items():
        rid = proxy[name]["prefill_begin"]["proxy_request_id"]
        paths[name] = make_request(end, proxy[name], target[name], origin, d_index[rid],
                                   starts[rid], batches[rid], lookups[rid], reads[rid])
        p_spans[target[name]["prefill"]].extend((left, right) for left, right, _ in reads[rid])

    bins = [row for row in decode["client_token_bin"] if row["case"] == case]
    for label in ROUTES:
        exact = sum(row["tokens"] for row in bins if row["route"] == label)
        expected = sum(row["output_tokens"] for row in paths.values() if row["decode"] == label)
        if exact != expected or any(row["time_s"] * 2 != int(row["time_s"] * 2) for row in bins):
            raise ValueError(f"Missing exact half-second token bins in {case}/{label}")
    d_samples = defaultdict(list)
    for row in sorted(decode["decode_supply_sample"], key=lambda event: event["monotonic_ns"]):
        if row["case"] == case:
            d_samples[row["route"]].append(dict(row, time_s=(row["monotonic_ns"] - origin) / NS))

    drain = max(row["client_end_s"] for row in paths.values())
    bounds = {key: edges for key, edges in plan["analysis_windows_s"].items() if key != "arrival"}
    if mode == "packed":
        bounds.update({"inject_1": [10.5, 17.5], "inject_2": [17.5, 24.5], "inject_3": [24.5, 35]})
    windows = {key: window_metrics(paths.values(), bins, d_samples, *edges, plan.get("max_num_seqs_per_D"))
               for key, edges in bounds.items()}
    port_samples = sorted((row for row in decode["cap_port_sample"] if row["case"] == case),
                          key=lambda row: row["unix_ns"])
    ports, d_tx_peak = port_deltas(port_samples)
    wave_rows = defaultdict(list)
    for row in paths.values():
        wave_rows[row["wave"]].append(row)
    waves = {wave: {"first_offer_s": min(r["offered_s"] for r in rows),
                    "last_P_HTTP_return_s": max(r["P_HTTP_return_s"] for r in rows),
                    "last_D_KV_ready_s": max(r["D_KV_ready_s"] for r in rows),
                    "max_TTFT_ms": max((r["client_first_s"] - r["offered_s"]) * 1000 for r in rows),
                    "D_to_P_GET_bytes": sum(r["P_GET_bytes"] for r in rows)}
             for wave, rows in sorted(wave_rows.items())}
    wave_sequence = list(waves.values())
    still_publishing_P = sum(earlier["last_P_HTTP_return_s"] >= later["first_offer_s"]
                             for earlier, later in zip(wave_sequence, wave_sequence[1:]))
    still_transferring_KV = sum(earlier["last_D_KV_ready_s"] >= later["first_offer_s"]
                                for earlier, later in zip(wave_sequence, wave_sequence[1:]))
    minimum_D_ready_slack_s = min((later["first_offer_s"] - earlier["last_D_KV_ready_s"]
                                   for earlier, later in zip(wave_sequence, wave_sequence[1:])), default=None)
    output_tokens = sum(row["output_tokens"] for row in paths.values())
    last_P = max(row["P_HTTP_return_s"] for row in paths.values())
    last_D = max(row["D_KV_ready_s"] for row in paths.values())
    pressure = {}
    if mode == "native" or "steady_output" in plan:
        tx_intervals = list(aligned_tx_intervals(port_samples, origin))
    if mode == "native":
        target_ids = {events["prefill_begin"]["proxy_request_id"] for events in proxy.values()}
        target_loads = [load for load in loads if target_ids.intersection(load["proxy_ids"])]
        pressure = {**cap_pressure(tx_intervals, last_P, cap),
                    "P_completed_per_s_until_last_reply": len(paths) / last_P,
                    "D_KV_ready_per_s_until_last_ready": len(paths) / last_D,
                    "P_worker_load_wall_ms": stats(load["worker_ms"] for load in target_loads),
                    "P_worker_native_GET_ms": stats(load["native_GET_sum_ms"] for load in target_loads),
                    "P_worker_native_GET_share_pct": stats(
                        100 * load["native_GET_sum_ms"] / load["worker_ms"] for load in target_loads),
                    "P_inflight_request_seconds": sum(row["P_HTTP_return_s"] - row["P_begin_s"] for row in paths.values()),
                    "P_completed_when_offers_stop": sum(row["P_HTTP_return_s"] <= plan["analysis_windows_s"]["arrival"][1]
                                                        for row in paths.values())}
    if "steady_output" in plan:
        pressure["steady_output"] = sustained_windows(paths.values(), bins, tx_intervals, cap, windows, plan)
        pressure["drain_after_last_offer_s"] = drain - plan["analysis_windows_s"]["arrival"][1]
    return {
        "requests": len(paths), "output_tokens": output_tokens,
        "D_cap_Gbps": cap, "D_port_readback_Mbps": readback, "D_port_counters": ports,
        "D_peak_sampled_MAC_TX_Gbps": d_tx_peak,
        "P_GET_bytes": sum(row["P_GET_bytes"] for row in paths.values()),
        "P_to_D_bytes": sum(row["P_to_D_bytes"] for row in paths.values()),
        "P_GET_activity_s": {side: interval_union(spans) / NS for side, spans in p_spans.items()},
        "P_any_GET_activity_s": interval_union(span for spans in p_spans.values() for span in spans) / NS,
        "P_local_hits": sum(row["P_local_prefix_tokens"] > 0 for row in paths.values()),
        "P_HTTP_ms": stats((r["P_HTTP_return_s"] - r["P_begin_s"]) * 1000 for r in paths.values()),
        "P_offer_to_HTTP_return_ms": stats((r["P_HTTP_return_s"] - r["offered_s"]) * 1000 for r in paths.values()),
        "P_GET_max_call_ms": stats(r["P_GET_max_call_ms"] for r in paths.values()),
        "P_queue_ms": stats(r["P_queue_ms"] for r in paths.values()),
        "P_first_dispatch_to_last_result_ms": stats(r["P_first_dispatch_to_last_result_ms"] for r in paths.values()),
        "D_transfer_wait_ms": stats((r["D_KV_ready_s"] - r["D_begin_s"]) * 1000 for r in paths.values()),
        "D_begin_to_transfer_enqueue_ms": stats(
            (r["D_transfer_enqueued_s"] - r["D_begin_s"]) * 1000 for r in paths.values()),
        "D_transfer_enqueued_to_KV_ready_ms": stats(
            (r["D_KV_ready_s"] - r["D_transfer_enqueued_s"]) * 1000 for r in paths.values()),
        "D_after_KV_ready_to_first_chunk_ms": stats((r["D_first_chunk_s"] - r["D_KV_ready_s"]) * 1000 for r in paths.values()),
        "TTFT_ms": stats((r["client_first_s"] - r["offered_s"]) * 1000 for r in paths.values()),
        "dispatch_jitter_ms": stats((r["client_dispatch_s"] - r["offered_s"]) * 1000 for r in paths.values()),
        "last_P_HTTP_return_s": last_P,
        "last_D_KV_ready_s": last_D,
        "P_pending_waves_at_next_offer": still_publishing_P,
        "D_KV_pending_waves_at_next_offer": still_transferring_KV,
        "minimum_D_KV_ready_to_next_wave_s": minimum_D_ready_slack_s,
        "drain_s": drain, "drain_output_tokens_s": output_tokens / drain,
        "D_first_to_last_P_result": {label: supply_time(
            [r for r in paths.values() if r["decode"] == label], 0, last_P, plan.get("max_num_seqs_per_D"))
            for label in ROUTES},
        "windows": windows, "waves": waves, "per_request": paths, **pressure,
    }


def verify_preparation(decode, case, plan, fixture):
    manifests = {row["case"]: row for row in decode["client_manifest"]}
    before, after = manifests[f"{case}-prep"], manifests[case]
    planned = {row["predecessor"]["id"]: row["predecessor"] for row in plan["target_requests"]}
    ends = {row["id"].split("~", 1)[1]: row for row in decode["client"]
            if row["case"] == f"{case}-prep" and row["type"] == "request_end"}
    if (before["fixed_replay_sha256"] != fixture or after["fixed_replay_sha256"] != fixture
            or before["workload_sha256"] != plan["source_workload_sha256"] or ends.keys() != planned.keys()):
        raise ValueError(f"Unmatched prepared context in {case}")
    ids = {end["id"] for end in ends.values()}
    proxy = defaultdict(dict)
    for row in decode["proxy"]:
        if row["id"] in ids:
            proxy[row["id"]][row["type"]] = row
    for name, end in ends.items():
        expected = planned[name]
        events = proxy[end["id"]]
        if (end["status"] != "ok" or end["prompt_tokens"] != expected["prompt_tokens"]
                or end["completion_tokens"] != expected["output_tokens"]
                or route(events["prefill_begin"]) != expected["prefill"] or route(events["decode_begin"]) != expected["decode"]):
            raise ValueError(f"Unexpected predecessor for {end['id']}")


def compare(low, high):
    low_paths, high_paths = low["per_request"], high["per_request"]
    changed_GET = [name for name, row in low_paths.items() if row["P_GET_bytes"] != high_paths[name]["P_GET_bytes"]]
    equal_GET = [name for name in low_paths if name not in changed_GET]
    result = {
        "fixed_request_GET_payloads": not changed_GET,
        "fixed_request_P_to_D_payloads": all(row["P_to_D_bytes"] == high_paths[name]["P_to_D_bytes"]
                                             for name, row in low_paths.items()),
        "different_GET_payload_requests": changed_GET,
        "P_GET_payload_delta_pct": (high["P_GET_bytes"] / low["P_GET_bytes"] - 1) * 100 if low["P_GET_bytes"] else None,
        "last_P_HTTP_return_delta_s": high["last_P_HTTP_return_s"] - low["last_P_HTTP_return_s"],
        "paired_GET_max_call_delta_equal_payload_ms": stats(
            high_paths[name]["P_GET_max_call_ms"] - low_paths[name]["P_GET_max_call_ms"] for name in equal_GET),
        "paired_P_HTTP_return_delta_ms": stats(
            (high_paths[name]["P_HTTP_return_s"] - row["P_HTTP_return_s"]) * 1000 for name, row in low_paths.items()),
        "paired_D_KV_ready_delta_ms": stats(
            (high_paths[name]["D_KV_ready_s"] - row["D_KV_ready_s"]) * 1000 for name, row in low_paths.items()),
        "P_offer_to_HTTP_p50_delta_ms": high["P_offer_to_HTTP_return_ms"]["p50"] - low["P_offer_to_HTTP_return_ms"]["p50"],
        "TTFT_p50_delta_ms": high["TTFT_ms"]["p50"] - low["TTFT_ms"]["p50"],
        "TTFT_p99_delta_ms": high["TTFT_ms"]["p99"] - low["TTFT_ms"]["p99"],
        "drain_output_tokens_pct_delta": (high["drain_output_tokens_s"] / low["drain_output_tokens_s"] - 1) * 100,
        "window_streamed_tokens_delta_per_s": {
            label: high["windows"][label]["streamed_tokens_per_s"] - first["streamed_tokens_per_s"]
            for label, first in low["windows"].items()
        },
    }
    if "steady_output" in low:
        first, second = low["steady_output"], high["steady_output"]
        if first["window"] != second["window"]:
            raise ValueError("The bandwidth arms use different fixed output windows.")
        label = first["window"]
        baseline = low["windows"][label]["streamed_tokens_per_s"]
        faster = high["windows"][label]["streamed_tokens_per_s"]
        result["fixed_output_window"] = {
            "window": label, "both_output_plateaus": first["qualifies_as_output_plateau"] and second["qualifies_as_output_plateau"],
            "output_tokens_per_s_pct_delta": (faster / baseline - 1) * 100 if baseline else None,
        }
    return result


def analyze(prefill, decode, plan, partial=False):
    p, = prefill["scope"]
    d, = decode["scope"]
    if p["role"] != "P" or d["role"] != "D" or p["run_id"] != d["run_id"]:
        raise ValueError("Expected same-run P and D traces")
    manifests = {row["case"]: row for row in decode["client_manifest"] if row["phase"] == "target"
                 and row["case"] in plan["cases"]}
    if not manifests or (not partial and manifests.keys() != plan["cases"].keys()):
        raise ValueError("Incomplete multiwave case set")
    starts = sorted((row["monotonic_s"], row["case"]) for row in decode["client"]
                    if row["type"] == "client_origin" and row["case"] in manifests)
    if [case for _, case in starts] != [case for case in plan["cases"] if case in manifests]:
        raise ValueError("Unexpected timing or order of target cases")
    p_settings = {k: p["active"][k] for k in SETTINGS}
    if p_settings != {k: d["active"][k] for k in SETTINGS}:
        raise ValueError("P/D replay settings differ")
    for field in ("p_local_prefix_caching", "pool_capacity_GiB"):
        if field in plan and p_settings[field] != plan[field]:
            raise ValueError(f"Runtime {field} differs from the frozen replay plan")
    mode, = {plan["cases"][case]["mode"] for case in manifests}
    p_arrivals, _, p_chunks, p_lookups, _ = find_methods_and_batches(prefill["native"])
    if mode == "native":
        reads, loads = native_prefill_gets(prefill["native"])
    elif mode == "packed":
        reads, loads = prefill_gets(prefill["native"]), []
    else:
        raise ValueError(f"Unknown multiwave read mode: {mode}")
    p_index = p_arrivals, p_chunks, p_lookups, reads, loads
    d_index = associate_transfers(decode["native"])
    fixture, = {manifest["fixed_replay_sha256"] for manifest in manifests.values()}
    if any(scope["active"].get("fixed_replay_sha256", fixture) != fixture for scope in (p, d)):
        raise ValueError("P/D fixture hashes differ")
    cases = {}
    for monotonic_s, case in starts:
        verify_preparation(decode, case, plan, fixture)
        cases[case] = per_case(decode, case, manifests[case], plan, round(monotonic_s * NS), p_index, d_index)
    pairs = {}
    for block in ("b1", "b2"):
        low, high = f"a-{block}-cap100", f"b-{block}-cap200"
        if low in cases and high in cases:
            pairs[block] = compare(cases[low], cases[high])
    return {
        "run_id": p["run_id"], "settings": p_settings, "cases": cases, "pairs_200_minus_100": pairs,
        "window_definition": "All timestamps are from the D host's monotonic clock except single-host P durations. "
                             "Rate windows are fixed arithmetically before replay. Empty D while P pending counts time "
                             "with a P HTTP in flight for that D and no unfinished, handed-off request. "
                             "Slot deficits distinguish handed-off from KV-ready unfinished requests. "
                             "MAC TX rates use the overlap of the per-port counter timelines. "
                             "D engine queue samples are separate from that event-derived interval. "
                             "KV-ready bytes are payloads counted when the last D rank finishes. "
                             "Client output rates use exact streamed token counts, including partially completed requests.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefill", "decode", "plan", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    recipe = read_json(args.plan)
    plan = expand(args.plan) if "source_plan" in recipe else recipe
    report = analyze(read_bundle(args.prefill), read_bundle(args.decode), plan, args.partial)
    report["source_sha256"] = {"P": sha256(args.prefill), "D": sha256(args.decode), "plan": sha256(args.plan)}
    if "source_plan_sha256" in plan:
        report["source_sha256"]["source_plan"] = plan["source_plan_sha256"]
    write_json(args.output, report)


if __name__ == "__main__":
    main()
