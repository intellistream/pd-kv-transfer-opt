"""Screen fixed-byte port allocation, handoff pipelining, and bounded host staging."""

import argparse
from collections import defaultdict
from heapq import heappop, heappush
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_full_rdma import associate_transfers
from pd_kv_transfer_opt.profiling.analyze_multiwave import aligned_tx_intervals
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution, read_bundle
from pd_kv_transfer_opt.profiling.profile_closed_sessions import read_trace


GB = 10**9
GIB = 2**30
NS = 10**9
PORTS = range(8)
D_SLOTS = 16


def proportional_caps(byte_counts, total_mbps):
    """Integer proportional shares for the measured r15 mix."""
    total_bytes = sum(byte_counts)
    shares = [divmod(count * total_mbps, total_bytes) for count in byte_counts]
    rates = [part for part, _ in shares]
    order = sorted(PORTS, key=lambda port: shares[port][1], reverse=True)
    for port in order[:total_mbps - sum(rates)]:
        rates[port] += 1
    if any(not 10_000 <= value <= 200_000 for value in rates):
        raise ValueError("The measured mix has a share outside the shaper range")
    return rates


def port_bounds(byte_counts, total_mbps, trained_caps):
    equal = [total_mbps // 8] * 8

    def last_port(caps):
        return max(count * 8 / (cap * 10**6) for count, cap in zip(byte_counts, caps, strict=True))

    equal_time, weighted_time = last_port(equal), last_port(trained_caps)
    peak = max(PORTS, key=byte_counts.__getitem__)
    return {
        "port_TX_GB": [count / GB for count in byte_counts],
        "max_port": peak, "max_over_mean_bytes": byte_counts[peak] * 8 / sum(byte_counts),
        "equal_port_wire_floor_s": equal_time, "r15_vector_wire_floor_s": weighted_time,
        "aggregate_wire_floor_s": sum(byte_counts) * 8 / (total_mbps * 10**6),
        "r15_vector_floor_change_pct": (weighted_time / equal_time - 1) * 100,
    }


def footprint_by_port(arm, case):
    origin, = (row["monotonic_s"] for row in arm["client"] if row["type"] == "client_origin")
    end = case["total"]["offer_to_last_end_s"]
    per_port = [0.0] * 8
    for left, right, _, rates in aligned_tx_intervals(arm["telemetry"], round(origin * NS)):
        width = max(0, min(end, right) - max(0, left))
        for port, rate in enumerate(rates):
            per_port[port] += width * rate / 8 * GB
    return [round(value) for value in per_port]


def port_screen(results):
    r15 = read_json(results / "2026-11-18-r15-same-head" / "summary.json")
    arms = {name: pair["baseline"] for name, pair in r15["paired_cases"].items()}
    weights = [arms["b1-cap100"]["D_ports"][str(port)]["tx_bytes"] for port in PORTS]
    vectors = {str(cap): proportional_caps(weights, cap * 1000) for cap in (100, 200)}
    output = {"training": "r15/b1-cap100 native", "vectors_Mbps": vectors, "r15": {}}
    for name, arm in arms.items():
        cap = name.split("cap")[1]
        counts = [arm["D_ports"][str(port)]["tx_bytes"] for port in PORTS]
        report = port_bounds(counts, int(cap) * 1000, vectors[cap])
        report["MAC_TX_bytes_above_GET_pct_of_TX"] = (sum(counts) - arm["get_payload_bytes"]) / sum(counts) * 100
        report["observed_drain_s"] = arm["drain_s"]
        output["r15"][name] = report

    for key, folder in (("r26", "2026-11-19-r26-native-pressure"), ("r27c", "2026-11-19-r27c-native-sustained")):
        output[key] = {}
        for name, arm in read_json(results / folder / "summary.json")["cases"].items():
            counts = [arm["D_port_counters"][str(port)]["tx_bytes"] for port in PORTS]
            total = int(arm["D_cap_Gbps"])
            output[key][name] = port_bounds(counts, total * 1000, vectors[str(total)])
    folder = results / "2026-11-19-r28-natural-bandwidth"
    trace_file = folder / "r28-events.jsonl.gz"
    trace_digest = sha256(trace_file)
    if trace_digest != read_json(folder / "manifest.json")["artifacts"][trace_file.name]["sha256"]:
        raise ValueError("The r28 port trace failed archive verification")
    trace = read_trace(trace_file)
    output["r28_port_trace_sha256"] = trace_digest
    output["r28"] = {}
    for name, arm in read_json(folder / "summary.json")["arms"].items():
        total = int(arm["cap_Gbps"])
        counts = footprint_by_port(trace[name], arm)
        output["r28"][name] = {**port_bounds(counts, total * 1000, vectors[str(total)]),
                               "observed_peak_port_1s_Gbps": max(arm["D_peak_one_s_Gbps_by_port"].values())}
    return output


def handoff_timing(requests, calls):
    all_ranks, ranks_critical, available = [], [], []
    for request, transfers in calls.items():
        if len(transfers) != 4:
            raise ValueError(f"Four complete native transfer ranks required for {request}")
        last_end = max(row["end_ns"] for row in transfers)
        last_submit = max(row["submit_ns"] for row in transfers)
        last_call = max(transfers, key=lambda row: row["end_ns"])
        if not last_submit <= last_end:
            raise ValueError(f"Invalid native submit/complete order for {request}")
        metrics = {"handler_queue_ms": last_call["queue_ms"], "host_prepare_ms": last_call["pre_submit_ms"],
                   "native_submit_to_complete_ms": last_call["submit_to_end_ms"],
                   "join_advance_with_zero_native_service_ms": (last_end - last_submit) / 1e6}
        ranks_critical.append(metrics)
        all_ranks.extend(row["submit_to_end_ms"] for row in transfers)
        req = requests[request]
        if req["D_transfer_enqueued_s"] - req["D_begin_s"] <= .25:
            available.append(metrics)
    tail_id = max(requests, key=lambda key: requests[key]["client_end_s"])
    return {"requests": len(requests), "all_rank_submit_to_complete_ms": distribution(all_ranks),
            "last_rank": summarize_rank_paths(ranks_critical),
            "D_enqueue_within_250ms": summarize_rank_paths(available),
            "tail": {"id": tail_id, "last_rank_ms": max(calls[tail_id], key=lambda row: row["end_ns"])["submit_to_end_ms"]}}


def summarize_rank_paths(rows):
    names = ("handler_queue_ms", "host_prepare_ms", "native_submit_to_complete_ms", "join_advance_with_zero_native_service_ms")
    return {"requests": len(rows), **{k: distribution(row[k] for row in rows) for k in names}}


def staging_screen(requests, capacity_gib, copy_gib_s):
    """Event replay at P returns, holding later D admission fixed; counts logical bytes."""
    active = defaultdict(list)
    used = defaultdict(int)
    peak = defaultdict(int)
    copy_until = defaultdict(float)
    staged, ready, wire = 0, 0, 0
    relieved_byte_s = 0.0
    capacity = capacity_gib * GIB
    for row in sorted(requests, key=lambda req: req["P_HTTP_return_s"]):
        now, p, d = row["P_HTTP_return_s"], row["prefill"], row["decode"]
        while active[p] and active[p][0][0] <= now:
            _, size = heappop(active[p])
            used[p] -= size
        occupied = sum(other["decode"] == d and other["D_begin_s"] < now < other["D_end_s"] for other in requests)
        size = row["P_to_D_bytes"]
        if occupied < D_SLOTS or used[p] + size > capacity or copy_until[p] > now:
            continue
        stop_copy = now + size / (copy_gib_s * GIB)
        native_ready = row["D_KV_ready_s"]
        staged += 1
        wire += size
        used[p] += size
        peak[p] = max(peak[p], used[p])
        copy_until[p] = stop_copy
        heappush(active[p], (max(stop_copy, native_ready), size))
        if stop_copy <= row["D_transfer_enqueued_s"]:
            ready += 1
            relieved_byte_s += size * (native_ready - stop_copy)
    original_byte_s = sum(row["P_to_D_bytes"] * (row["D_KV_ready_s"] - row["P_HTTP_return_s"]) for row in requests)
    return {"capacity_GiB_per_P": capacity_gib, "assumed_D2H_GiB_s": copy_gib_s, "D_inflight_trigger": D_SLOTS,
            "copied_requests": staged, "ready_before_native_D_enqueue": ready, "staged_GB": wire / GB,
            "peak_buffer_GiB": {role: value / GIB for role, value in peak.items()},
            "observed_logical_P_holding_GB_s": original_byte_s / GB,
            "modeled_relief_GB_s": relieved_byte_s / GB, "modeled_relief_pct": relieved_byte_s / original_byte_s * 100}


def handoff_screen(summary_path, decode_path):
    summary = read_json(summary_path)
    digest = sha256(decode_path)
    if digest != summary["source_sha256"]["D"]:
        raise ValueError("Native transfer bundle does not match the checked summary")
    decode = read_bundle(decode_path)
    calls = associate_transfers(decode["native"])
    proxy_ids = {row["id"]: row["proxy_request_id"] for row in decode["proxy"] if row["type"] == "decode_begin"}
    arms = {}
    for name, row in summary["cases"].items():
        requests = row["per_request"]
        transfers = {key: calls[proxy_ids[f"{name}~{key}"]] for key in requests}
        arms[name] = handoff_timing(requests, transfers)
        values = list(requests.values())
        arms[name]["bounded_P_host_staging"] = [staging_screen(values, capacity, 8)
                                               for capacity in (4, 8, 16, 32)]
    return {"D_trace_sha256": digest, "arms": arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Root of the checked results archive")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    phases = {}
    for key, name in (("r26", "2026-11-19-r26-native-pressure"), ("r27c", "2026-11-19-r27c-native-sustained")):
        folder = args.results / name
        phases[key] = handoff_screen(folder / "summary.json", folder / f"{key}-D.jsonl.gz")
    write_json(args.output, {"port_wire_service_floors": port_screen(args.results),
                            "native_handoff_and_staging": phases})


if __name__ == "__main__":
    main()
