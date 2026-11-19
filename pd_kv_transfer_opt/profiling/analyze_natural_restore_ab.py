"""Analyze fixed natural continuation bursts after their real predecessors drain."""

import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path
import re
from statistics import mean, median
from urllib.parse import urlsplit

from pd_kv_transfer_opt.common import percentile, read_json, write_json
from pd_kv_transfer_opt.profiling.analyze_full_rdma import find_methods_and_batches
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union, normalized_proxy_id, read_bundle
from pd_kv_transfer_opt.profiling.analyze_packed_readiness import load_spans
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import stats
from pd_kv_transfer_opt.profiling.analyze_rdma_hard_caps import port_rates
from pd_kv_transfer_opt.profiling.analyze_same_head_ab import COPY, cap_readback, copy_metrics, delta_percent, port_deltas


CASE = re.compile(r"([ab])-(b\d+)-cap(100|200)")
ROUTE = {18200: "P0", 18201: "P1", 18300: "D0", 18301: "D1"}
METRICS = ("get_bytes", "D_mac_tx_bytes", "P_mac_rx_bytes", "output_tokens_s", "ttft_p99_s", "drain_s")
SETTINGS = ("p_local_prefix_caching", "d_local_prefix_caching", "p_batched_tokens", "p_graph_capture_sizes",
            "max_model_len", "p_graph_strategy", "d_graph_strategy", "pool_capacity_GiB",
            "experimental_batch_get_concurrency", "rdma_pressure")


def route(event):
    if "route" in event:
        return event["route"]
    return ROUTE[urlsplit(event["url"]).port]


def match_targets(decode, manifest, planned):
    case = manifest["case"]
    sources = {row["id"]: row for row in manifest["request_sources"]}
    ends = {}
    for row in decode["client"]:
        if row.get("case") == case and row["type"] == "request_end":
            prefix, _, stable = row["id"].partition("~")
            if prefix != case or stable not in planned or stable in ends:
                raise ValueError(f"Unexpected target identifier in {case}")
            target = planned[stable]
            if (row["status"] != "ok" or row["completion_tokens"] != target["max_tokens"]
                    or row["prompt_tokens"] != sources[row["id"]]["prompt_tokens"]
                    or abs(row["offered_s"] - target["offered_s"]) > 1e-9
                    or abs(row["ready_s"] - row["offered_s"]) > 1e-9):
                raise ValueError(f"Request output or arrival contract failed: {row['id']}")
            ends[stable] = row
    if ends.keys() != planned.keys():
        raise ValueError(f"Missing client completions in {case}")

    by_id = {row["id"]: name for name, row in ends.items()}
    proxy = defaultdict(dict)
    for row in decode["proxy"]:
        if row["id"] in by_id:
            proxy[by_id[row["id"]]][row["type"]] = row
    for name, target in planned.items():
        if route(proxy[name]["prefill_begin"]) != target["prefill"] or route(proxy[name]["decode_begin"]) != target["decode"]:
            raise ValueError(f"Actual route differs from fixture in {case}: {name}")
    return ends, proxy


def setup_index(prefill):
    events = sorted(prefill["native"], key=lambda row: row["monotonic_ns"])
    arrivals, _, steps, lookups, _ = find_methods_and_batches(events)
    loads_by_request = defaultdict(list)
    loads_by_batch = defaultdict(list)
    for span in load_spans(events):
        loads_by_batch[span.begin.get("batch")].append(span)
        ids = [read.request_id for read in span.reads]
        if any(rid is None for rid in ids):
            ids = span.begin["request_ids"]
        if span.verified or len(ids) != len(span.reads):
            # Track unassignable scopes so a timed request cannot silently lose its bytes.
            for rid in span.begin["request_ids"]:
                loads_by_request[normalized_proxy_id(rid)].append((span, None))
            continue
        for rid, read in zip(ids, span.reads, strict=True):
            loads_by_request[normalized_proxy_id(rid)].append((span, read))
    return {"arrivals": arrivals, "steps": steps, "lookups": lookups, "loads_by_request": loads_by_request,
            "loads_by_batch": loads_by_batch, "events": events}


def request_metrics(end, proxy, index):
    pid = proxy["prefill_begin"]["proxy_request_id"]
    chunks = index["steps"][pid]
    reads = index["loads_by_request"].get(pid, [])
    if any(read is None for _, read in reads):
        raise ValueError(f"Verification or unassignable GET scope inside measured request {end['id']}")
    actual = sum(read.size for _, read in reads)
    # Any omitted attention bytes are the local destination payload. Keep rank-private reads in both terms.
    reconstructed = sum(read.size + (read.copy["payload_bytes"] if read.copy is not None else 0)
                        for _, read in reads)
    if reads and all(read.native_size is not None for _, read in reads):
        equivalent = sum(read.native_size for _, read in reads)
        if reconstructed != equivalent:
            raise ValueError(f"Shadow descriptor total disagrees with GET and copy: {end['id']}")
    else:
        equivalent = reconstructed

    def span(begin, finish):
        return (proxy[finish]["monotonic_ns"] - proxy[begin]["monotonic_ns"]) / 1e6

    begin = end["first_s"]
    tail = end["end_s"] - begin
    # Include other restores in these batches since these also hold the batch future.
    batch_loads = [load for batch in chunks for load in index["loads_by_batch"][batch["batch"]] if load.reads]
    batch_reads = [read for load in batch_loads for read in load.reads]
    first_dispatch = chunks[0]["dispatch_ns"]
    lookup = [row for row in index["lookups"][pid] if row["monotonic_ns"] <= first_dispatch][-1]
    load_window = (max(load.end["monotonic_ns"] for load, _ in reads)
                   - min(load.begin["monotonic_ns"] for load, _ in reads)) / 1e6 if reads else None
    return {"ttft_s": begin - end["ready_s"], "client_dispatch_s": end["dispatch_s"],
            "offered_s": end["offered_s"], "end_s": end["end_s"], "decode_tail_s": tail,
            "TPOT_ms": tail * 1000 / (end["completion_tokens"] - 1) if end["completion_tokens"] > 1 else None,
            "P_http_ms": span("prefill_begin", "prefill_end"),
            "D_first_chunk_ms": span("decode_begin", "decode_first_chunk"),
            "P_end_to_D_first_chunk_ms": span("prefill_end", "decode_first_chunk"),
            "P_queue_ms": (first_dispatch - index["arrivals"][pid]["arrival_ns"]) / 1e6,
            "P_dispatch_to_result_ms": (chunks[-1]["ready_ns"] - first_dispatch) / 1e6,
            "P_local_prefix_tokens": lookup["local_tokens"],
            "P_external_prefix_tokens": lookup["external_tokens"],
            "P_first_precomputed_tokens": next(offset for rid, offset, _ in chunks[0]["rows"]
                                                if normalized_proxy_id(rid) == pid),
            "P_model_scheduled_tokens": sum(tokens for batch in chunks for rid, _, tokens in batch["rows"]
                                            if normalized_proxy_id(rid) == pid),
            "P_load_window_ms": load_window,
            "P_batch_GET_active_union_ms": interval_union((read.begin, read.end) for read in batch_reads) / 1e6,
            "P_batch_load_active_union_ms": interval_union((load.begin["monotonic_ns"], load.end["monotonic_ns"])
                                                            for load in batch_loads) / 1e6,
            "P_batch_copy_active_union_ms": interval_union((read.copy_begin, read.returned) for read in batch_reads
                                                            if read.copy is not None) / 1e6,
            "GET_slowest_worker_ms": max(((read.end - read.begin) / 1e6 for _, read in reads), default=0),
            "get_bytes": actual, "native_equivalent_bytes": equivalent, "get_calls": len(reads)}


def case_metrics(prefill, decode, manifest, planned, index):
    case = manifest["case"]
    limit = manifest["aggregate_D_egress_cap_Gbps"]
    ends, proxy = match_targets(decode, manifest, planned)
    per_request = {name: request_metrics(end, proxy[name], index) for name, end in ends.items()}
    origin, = [row["monotonic_s"] for row in decode["client"] if row["case"] == case and row["type"] == "client_origin"]
    for name, row in per_request.items():
        row["proxy_ingress_delay_ms"] = (proxy[name]["proxy_arrival"]["monotonic_ns"] / 1e9
                                         - origin - ends[name]["offered_s"]) * 1000
    client_summary, = [row for row in decode["client_summary"] if row["case"] == case]
    if (client_summary["counts"] != {"ok": len(planned)} or client_summary["quality"]["errors"]
            or client_summary.get("fixed_output_deviations")):
        raise ValueError(f"Client accounting failed for {case}")
    drain = client_summary["finite_cohort"]["drain_included_s"]
    output = sum(row["completion_tokens"] for row in ends.values())
    gets = sum(row["get_bytes"] for row in per_request.values())
    native = sum(row["native_equivalent_bytes"] for row in per_request.values())
    samples = sorted((row for row in decode["cap_port_sample"] if row["case"] == case), key=lambda row: row["unix_ns"])
    ports, peak = port_deltas(samples)
    d_rates = port_rates(samples)
    p_edges = {row["edge"]: row for row in prefill["p_port_snapshot"] if row["case"] == case}
    p_samples = [p_edges["before"], p_edges["after"]]
    p_ports, _ = port_deltas(p_samples)
    readback = cap_readback(decode, case, limit, ports)
    if readback is None or set(readback.values()) != {limit * 1000 // len(ports)}:
        raise ValueError(f"Missing cap readback for {case}")
    proxy_ids = {events["prefill_begin"]["proxy_request_id"]: name for name, events in proxy.items()}
    copies = copy_metrics(index["events"], proxy_ids, COPY)
    decisions = Counter(row["mode"] for row in index["events"] if row["type"] == "r18_restore_decision"
                        and normalized_proxy_id(row["request_id"]) in proxy_ids)
    batch_keys = {batch["batch"] for pid in proxy_ids for batch in index["steps"][pid]}
    preflight = [row for row in index["events"] if row["type"] == "r18_packed_preflight" and row["batch"] in batch_keys]
    pd_bytes = sum(row["payload_bytes"] for row in decode["native"] if row["type"] == "native_transfer_submit"
                   and normalized_proxy_id(row.get("remote_request_id")) in proxy_ids)
    distributions = ("P_http_ms", "P_queue_ms", "P_dispatch_to_result_ms", "P_load_window_ms",
                     "GET_slowest_worker_ms", "D_first_chunk_ms", "decode_tail_s", "TPOT_ms")
    read_path = [read for pid in proxy_ids for _, read in index["loads_by_request"].get(pid, [])]
    read_submissions = sorted((read.begin, read.size) for read in read_path)
    phases_by_prefill = {}
    for instance in sorted({target["prefill"] for target in planned.values()}):
        names = [name for name, target in planned.items() if target["prefill"] == instance]
        path = [pair for name in names
                for pair in index["loads_by_request"].get(proxy[name]["prefill_begin"]["proxy_request_id"], [])]
        phases_by_prefill[instance] = {
            "requests": len(names), "get_bytes": sum(read.size for _, read in path),
            "get_active_union_s": interval_union((read.begin, read.end) for _, read in path) / 1e9,
            "sync_load_active_union_s": interval_union((span.begin["monotonic_ns"], span.end["monotonic_ns"])
                                                       for span, _ in path) / 1e9,
            "copy_active_union_s": interval_union((read.copy_begin, read.returned) for _, read in path
                                                   if read.copy is not None) / 1e9,
        }
    pressure = {}
    for window_s in (0.5, 1.0, 2.0):
        window = deque()
        current = maximum = 0
        for time_ns, nbytes in read_submissions:
            while window and window[0][0] <= time_ns - round(window_s * 1e9):
                current -= window.popleft()[1]
            window.append((time_ns, nbytes))
            current += nbytes
            maximum = max(maximum, current)
        pressure[str(window_s)] = 8 * maximum / (window_s * 1e9)

    return {"requests": len(ends), "output_tokens": output, "drain_s": drain, "output_tokens_s": output / drain,
            "ttft_p99_s": percentile([row["ttft_s"] for row in per_request.values()], .99),
            "ttft_p50_s": percentile([row["ttft_s"] for row in per_request.values()], .50),
            "dispatch_jitter_ms": stats((row["client_dispatch_s"] - row["offered_s"]) * 1000
                                        for row in per_request.values()),
            "proxy_ingress_delay_ms": stats(row["proxy_ingress_delay_ms"] for row in per_request.values()),
            "get_bytes": gets, "native_equivalent_bytes": native, "copy": copies, "restore_decisions": dict(decisions),
            "preflight": {key: stats(row[f"{key}_ns"] / 1e6 for row in preflight) for key in ("plan", "wait")},
            "requests_with_GET": sum(row["get_calls"] > 0 for row in per_request.values()),
            "P_local_prefix_tokens": sum(row["P_local_prefix_tokens"] for row in per_request.values()),
            "P_external_prefix_tokens": sum(row["P_external_prefix_tokens"] for row in per_request.values()),
            "P_model_scheduled_tokens": sum(row["P_model_scheduled_tokens"] for row in per_request.values()),
            "submitted_GET_peak_Gbps_by_rolling_window_s": pressure,
            "aggregated_payload_floor_s": 8 * gets / (limit * 1e9),
            "GET_active_union_all_workers_s": interval_union((read.begin, read.end) for read in read_path) / 1e9,
            "D_busiest_port_MAC_serialization_s": max(8 * row["tx_bytes"] / (readback[str(port)] * 1e6)
                                                     for port, row in ports.items()),
            "D_mac_tx_bytes": sum(row["tx_bytes"] for row in ports.values()),
            "D_mac_rx_bytes": sum(row["rx_bytes"] for row in ports.values()),
            "P_mac_tx_bytes": sum(row["tx_bytes"] for row in p_ports.values()),
            "P_mac_rx_bytes": sum(row["rx_bytes"] for row in p_ports.values()),
            "P_to_D_descriptor_bytes": pd_bytes, "D_tx_sample_peak_Gbps": peak,
            "D_ports": ports, "P_ports": p_ports, "D_port_readback_Mbps": readback,
            "D_tx_time_windows": d_rates["segments"], "D_counter_delta": d_rates["counters_delta"],
            "P_counter_delta": port_rates(p_samples)["counters_delta"],
            "P_by_instance": phases_by_prefill,
            "phases": {key: stats(row[key] for row in per_request.values() if row[key] is not None)
                       for key in distributions},
            "last_request": max(per_request, key=lambda name: per_request[name]["end_s"]),
            "max_ttft_request": max(per_request, key=lambda name: per_request[name]["ttft_s"]),
            "per_request": per_request}


def compare(prefill, decode, workload, modes=("baseline", "packed"), allow_pair_fixtures=False):
    planned = {row["id"]: row for row in workload["requests"]}
    p_scope, = prefill["scope"]
    d_scope, = decode["scope"]
    if p_scope["role"] != "P" or d_scope["role"] != "D" or p_scope["run_id"] != d_scope["run_id"]:
        raise ValueError("Expected a P and D bundle from one run")
    settings = {key: p_scope["active"][key] for key in SETTINGS}
    if settings != {key: d_scope["active"][key] for key in SETTINGS} or not settings["p_local_prefix_caching"]:
        raise ValueError("Run settings disagree or APC is disabled")
    manifests = {row["case"]: row for row in decode["client_manifest"]}
    summaries = {row["case"]: row for row in decode["client_summary"]}
    controls = {row["name"]: row["data"] for row in decode["cap_control"]}
    predecessor_ids = {name for target in planned.values() for name in target["predecessors"]}
    index = setup_index(prefill)
    arms = defaultdict(dict)
    replay_hashes = defaultdict(set)
    for row in manifests.values():
        match = CASE.fullmatch(row["case"])
        if match is None:
            continue
        arm, block, bandwidth = match.groups()
        case = row["case"]
        mode = modes[0 if arm == "a" else 1]
        if (int(bandwidth) != row["aggregate_D_egress_cap_Gbps"] or len(planned) != row["selected_requests"]
                or row["phase"] != "target" or row["arm_mode"] != mode or not row["ignore_eos"]
                or row["background_traffic"] or row["profiler_enabled"]
                or row["workload_sha256"] != workload["source_workload_sha256"]):
            raise ValueError(f"Manifest protocol differs for {row['case']}")
        prep_manifest = manifests[f"{case}-prep"]
        prep_summary = summaries[f"{case}-prep"]
        prep_sources = {entry["id"] for entry in prep_manifest["request_sources"]}
        if (prep_sources != {f"{case}~{name}" for name in predecessor_ids}
                or prep_summary["counts"] != {"ok": len(predecessor_ids)}
                or prep_summary["fixed_output_deviations"] or prep_summary["quality"]["errors"]
                or controls[f"{case}-prep-progress.json"]["stage"] != "complete"
                or controls[f"{case}-progress.json"]["stage"] != "complete"
                or not controls[f"{case}-restore.json"]["restored"]):
            raise ValueError(f"Cache preparation or shaper restoration incomplete for {case}")
        pair_key = int(bandwidth), block
        replay_hashes[pair_key].update((row["fixed_replay_sha256"], prep_manifest["fixed_replay_sha256"]))
        arms[pair_key][mode] = case_metrics(prefill, decode, row, planned, index)
    if not arms:
        raise ValueError("No measured cases found")
    all_hashes = set().union(*replay_hashes.values())
    if (not all(all_hashes) or any(len(hashes) != 1 for hashes in replay_hashes.values())
            or (not allow_pair_fixtures and len(all_hashes) != 1)):
        raise ValueError("Fixed replay fixtures differ between arms")
    fixtures = ({"fixed_replay_sha256": all_hashes.pop()} if len(all_hashes) == 1 else {
        "fixed_replay_sha256_by_pair": {f"{cap}-{block}": next(iter(hashes))
                                        for (cap, block), hashes in sorted(replay_hashes.items())}})
    paired = []
    for (cap, block), cases in sorted(arms.items()):
        if cases.keys() != set(modes):
            raise ValueError(f"Missing pair at {cap} Gbps, {block}")
        base, packed = (cases[mode] for mode in modes)
        by_request = [(base["per_request"][name], packed["per_request"][name]) for name in planned]
        stages = ("ttft_s", "P_http_ms", "P_queue_ms", "P_dispatch_to_result_ms", "P_batch_GET_active_union_ms",
                  "P_batch_load_active_union_ms", "P_batch_copy_active_union_ms", "P_end_to_D_first_chunk_ms",
                  "decode_tail_s", "TPOT_ms")
        paired.append({"D_egress_cap_Gbps": cap, "block": block, **cases,
                       "delta_percent": {key: delta_percent(base[key], packed[key]) for key in METRICS},
                       "paired_request_stage_delta_packed_minus_baseline": {
                           key: stats(b[key] - a[key] for a, b in by_request if a[key] is not None and b[key] is not None)
                           for key in stages},
                       "requests_with_changed_P_work": [name for name, (a, b) in zip(planned, by_request, strict=True)
                                                        if any(a[key] != b[key] for key in (
                                                            "P_local_prefix_tokens", "P_external_prefix_tokens",
                                                            "P_model_scheduled_tokens", "native_equivalent_bytes"))],
                       "actual_GET_savings_bytes": base["get_bytes"] - packed["get_bytes"],
                       "packed_same_plan_savings_bytes": packed["native_equivalent_bytes"] - packed["get_bytes"],
                       "difference_in_native_demand_bytes": base["native_equivalent_bytes"] - packed["native_equivalent_bytes"]})
    tables = []
    for cap in sorted({row["D_egress_cap_Gbps"] for row in paired}):
        local = [row for row in paired if row["D_egress_cap_Gbps"] == cap]
        totals = {mode: {key: (mean if key == "output_tokens_s" else median)(row[mode][key] for row in local)
                         for key in METRICS} for mode in modes}
        for mode, output in totals.items():
            pooled_ttft = [item["ttft_s"] for row in local for item in row[mode]["per_request"].values()]
            output["pooled_ttft_p99_s"] = percentile(pooled_ttft, .99)
        tables.append({"D_egress_cap_Gbps": cap, "n_pairs": len(local), **totals,
                       "delta_percent": {key: delta_percent(totals[modes[0]][key], totals[modes[1]][key]) for key in METRICS}})
    return {"run_id": p_scope["run_id"], **fixtures, "settings": settings,
            "target_requests_per_arm": len(planned), "predecessor_requests_per_arm": len(predecessor_ids),
            "paired_rounds": paired, "by_condition": tables}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prefill", "decode", "workload", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    write_json(args.output, compare(read_bundle(args.prefill), read_bundle(args.decode), read_json(args.workload)))


if __name__ == "__main__":
    main()
