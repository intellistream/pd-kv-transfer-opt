"""Attribute the r21 matched packed, prepare-only and early-GET trials."""

import argparse
from collections import defaultdict
from pathlib import Path
import re

from pd_kv_transfer_opt.common import read_json, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import (
    interval_union,
    normalized_proxy_id,
    read_bundle,
)
from pd_kv_transfer_opt.profiling.analyze_natural_restore_ab import (
    SETTINGS,
    case_metrics,
    setup_index,
)
from pd_kv_transfer_opt.profiling.analyze_packed_readiness import Read, client_ids, join_forwards, request_names


CASE = re.compile(r"([abc])-(b\d+)-cap(100|200)")
MODES = {"a": "packed", "b": "prepare-only", "c": "early-get"}
WITNESS = "natural-video-processing:28"
MS = 1_000_000
CLOCK_SENSITIVITY_NS = 42 * MS
PROBES = ("forward_started_before_get", "forward_ended_before_get",
          "forward_started_after_get", "forward_ended_after_get")
DEVICE_TIMESTAMP_MARGIN_NS = 100_000


def event_key(row):
    return row["pid"], row["batch"], normalized_proxy_id(row["request_id"])


def index_restores(prefill, timed_requests):
    events = prefill["native"]
    early = {}
    for row in events:
        if row["type"] != "r21_early_get":
            continue
        key = event_key(row)
        if key in early or not row["success"] or not row["get_begin_ns"] < row["get_end_ns"] or row["payload_bytes"] <= 0:
            raise ValueError("Duplicate, failed, or empty timed early GET")
        early[key] = row

    # The ordinary parser joins GET and copy on one worker thread. Keep physical
    # early timestamps and join their later copy explicitly across threads.
    copies = {}
    worker_events = []
    for row in events:
        if row["type"] == "same_head_local_copy" and row.get("request_id") and event_key(row) in early:
            key = event_key(row)
            if key in copies or row["verify"]:
                raise ValueError("Duplicate or verification copy for a timed early GET")
            copies[key] = row
        else:
            worker_events.append(row)
    worker_prefill = {**prefill, "native": worker_events}
    index = setup_index(worker_prefill)
    for (pid, batch, request), row in early.items():
        if request not in timed_requests:
            continue
        candidates = [span for span in index["loads_by_batch"][batch] if span.begin["pid"] == pid and
                      request in {normalized_proxy_id(rid) for rid in span.begin["request_ids"]} and
                      row["get_end_ns"] <= span.end["monotonic_ns"]]
        if len(candidates) != 1:
            raise ValueError("Early GET lacks a unique successor worker load")
        span, = candidates
        if any(old_span.begin["pid"] == pid for old_span, _ in index["loads_by_request"][request]):
            raise ValueError("A rank physically read the same timed request in both paths")
        copy = copies.get((pid, batch, request))
        ordinal = row.get("call_ordinal", copy.get("call_ordinal") if copy else None)
        read = Read(row["get_begin_ns"], row["get_end_ns"], row["payload_bytes"], row["request_id"],
                    span.native_bytes.get(ordinal), copy, ordinal)
        if copy and not read.end <= read.copy_begin <= read.returned <= span.end["monotonic_ns"]:
            raise ValueError("Early GET and successor copy visibility are out of order")
        span.reads.append(read)
        index["loads_by_request"][request].append((span, read))
    index["events"] = events

    batches, nodes, errors, unmatched = join_forwards(worker_prefill)
    if errors or unmatched or len(batches) != len(nodes):
        raise ValueError("Incomplete P scheduler/worker trace")
    return index, nodes, early


def clock_margin_ns(prefill, decode):
    """Clock-fit margin using the two recorded chrony tracking snapshots."""
    seconds = 0
    for bundle in (prefill, decode):
        reports = bundle.get("clock", [])
        if len(reports) != 1 or reports[0].get("returncode") != 0:
            return None
        fields = {label: float(value) for label, value in re.findall(
            r"^(System time|Root delay|Root dispersion)\s*:\s*([+-]?[0-9.]+)", reports[0]["text"], re.M)}
        if len(fields) != 3:
            return None
        seconds += abs(fields["System time"]) + fields["Root delay"] / 2 + fields["Root dispersion"]
    return max(CLOCK_SENSITIVITY_NS, round(seconds * 1e9))


def port_window(samples, start, end, readback, margin):
    """Use only adjacent samples whose entire query frames lie inside the window."""
    by_port = defaultdict(list)
    if margin is not None:
        for sample in samples:
            if "monotonic_ns" not in sample:
                continue
            offset = sample["unix_ns"] - sample["monotonic_ns"]
            for row in sample["ports"]:
                if "start_monotonic_ns" not in row or "end_monotonic_ns" not in row:
                    continue
                before, after = row["start_monotonic_ns"] + offset, row["end_monotonic_ns"] + offset
                if start + margin <= before <= after <= end - margin:
                    by_port[str(row["port"])].append((row["monotonic_ns"], before, after, row["counters"]))

    result = {}
    for port, limit in sorted(readback.items()):
        windows = []
        for a, b in zip(by_port[port], by_port[port][1:]):
            elapsed = b[0] - a[0]
            transmitted = b[3]["mac_tx_total_oct_num"] - a[3]["mac_tx_total_oct_num"]
            if elapsed <= 0 or transmitted < 0:
                raise ValueError("D port clock or counter moved backward")
            fastest_ns, slowest_ns = b[1] - a[2], b[2] - a[1]
            windows.append({
                "start_since_window_ms": ((a[1] + a[2]) / 2 - start) / MS,
                "end_since_window_ms": ((b[1] + b[2]) / 2 - start) / MS,
                "coverage_ms": elapsed / MS, "tx_bytes": transmitted,
                "tx_average_Gbps": 8 * transmitted / elapsed,
                "tx_average_Gbps_bounds": [8 * transmitted / slowest_ns,
                                           8 * transmitted / fastest_ns if fastest_ns > 0 else None],
                "mac_rx_bytes": b[3]["mac_rx_total_oct_num"] - a[3]["mac_rx_total_oct_num"],
            })
            if "mac_rx_pfc_pkt_num" in a[3] and "mac_rx_pfc_pkt_num" in b[3]:
                windows[-1]["pfc_rx_frames"] = b[3]["mac_rx_pfc_pkt_num"] - a[3]["mac_rx_pfc_pkt_num"]
        result[port] = {"configured_Gbps": limit / 1000, "interior_samples": len(by_port[port]), "bins": windows}
        if windows:
            ns = by_port[port][-1][0] - by_port[port][0][0]
            bytes_tx = sum(item["tx_bytes"] for item in windows)
            result[port].update(
                covered_average_tx_Gbps=8 * bytes_tx / ns,
                covered_peak_tx_average_Gbps=max(item["tx_average_Gbps"] for item in windows),
                nominal_unused_capacity_bytes=max(0, limit * ns / 8000 - bytes_tx),
            )
    return {"window_ms": (end - start) / MS, "clock_margin_ms": margin / MS if margin is not None else None,
            "ports_with_interval": sum(bool(row["bins"]) for row in result.values()), "ports": result}


def batch_metrics(node, names, origin):
    batch = node["batch"]
    forwards = list(node["forwards"].values())
    first_worker = min(row["worker_enter_ns"] for row in forwards)
    last_forward = max(row["forward_enter_ns"] for row in forwards)
    return {
        "batch": batch["batch"], "instance": batch["instance"],
        "shape": sorted([names[normalized_proxy_id(rid)].split("~", 1)[1], offset, tokens]
                        for rid, offset, tokens in batch["rows"]),
        "dispatch_since_P_arrival_ms": (batch["dispatch_ns"] - origin) / MS,
        "dispatch_to_first_worker_ms": (first_worker - batch["dispatch_ns"]) / MS,
        "first_worker_to_last_host_forward_ms": (last_forward - first_worker) / MS,
        "last_host_forward_to_result_ms": (batch["ready_ns"] - last_forward) / MS,
        "dispatch_to_scheduler_result_ms": (batch["ready_ns"] - batch["dispatch_ns"]) / MS,
        "scheduler_result_since_P_arrival_ms": (batch["ready_ns"] - origin) / MS,
        "rank0_device_forward_ms": batch.get("device", {}).get("forward_device_ms"),
    }


def full_port_deltas(controls, case):
    before = controls.get(f"{case}-ports-full-before.json")
    after = controls.get(f"{case}-ports-full-after.json")
    if before is None or after is None:
        return None
    totals = {}
    for left, right in zip(before["ports"], after["ports"], strict=True):
        if left["port"] != right["port"]:
            raise ValueError("D full port snapshot order changed")
        counters_a, counters_b = left["counters"], right["counters"]
        changes = {key: counters_b[key] - counters_a[key] for key in sorted(counters_a.keys() & counters_b.keys())}
        if any(value < 0 for value in changes.values()):
            raise ValueError("D full port counter moved backward")
        totals[left["port"]] = changes
    return totals


def device_intersection(event, consume, earliest_marker_ns):
    """Conservative intersection shared by all forward start times in the bracket."""
    begin, end = event["get_begin_ns"], event["get_end_ns"]
    if event.get("forward_started_before_get") is True and event.get("forward_ended_after_get") is False:
        return begin, end
    # A marker recorded by _model_forward cannot have elapsed before that
    # worker's preceding synchronous load finished (or the worker entered).
    earliest = max(earliest_marker_ns, event.get("forward_start_earliest_ns") or 0)
    latest = event.get("forward_start_latest_ns")
    if latest is None and event.get("forward_started_after_get") is True:
        latest = event["monotonic_ns"]
    duration = consume.get("old_forward_device_ms")
    if latest is None or duration is None or latest < earliest:
        return None
    lo = max(begin, latest)
    hi = min(end, earliest + round(duration * MS) - DEVICE_TIMESTAMP_MARGIN_NS)
    return (lo, hi) if hi > lo else None


def attribute_witness(prefill, decode, case, per_case, index, nodes, early, margin):
    names = client_ids(decode)
    request, = [pid for pid, name in names.items() if name == f"{case}~{WITNESS}"]
    successor = min((node for node in nodes if any(normalized_proxy_id(rid) == request
                    for rid, _, _ in node["batch"]["rows"])), key=lambda node: node["batch"]["dispatch_ns"])
    batch = successor["batch"]
    first_worker = min(row["worker_enter_ns"] for row in successor["forwards"].values())
    old_nodes = [node for node in nodes if node["batch"]["instance"] == batch["instance"] and
                 node["batch"]["dispatch_ns"] < batch["dispatch_ns"] and
                 all(names.get(normalized_proxy_id(rid), "").startswith(f"{case}~") for rid, _, _ in node["batch"]["rows"])]
    plans = [row for row in prefill["native"] if row["type"] == "r21_plan" and
             normalized_proxy_id(row["request_id"]) == request]
    consumes = [row for row in prefill["native"] if row["type"] == "r21_consume" and
                normalized_proxy_id(row["request_id"]) == request]
    by_rank_consume = {row["tp_rank"]: row for row in consumes}
    releases = [row for row in prefill["native"] if row["type"] == "r21_release" and
                normalized_proxy_id(row["request_id"]) == request and row["batch"] == batch["batch"]]
    named = {row["predecessor_batch"] for row in plans + releases if row.get("predecessor_batch")}
    matching = [node for node in old_nodes if node["batch"]["batch"] in named] if named else old_nodes
    predecessor = max(matching, key=lambda node: node["batch"]["dispatch_ns"], default=None)
    origin = index["arrivals"][request]["arrival_ns"]
    ranks = {row["pid"]: rank for rank, row in successor["forwards"].items()}
    restore_jobs = defaultdict(list)
    for span in index["loads_by_batch"][batch["batch"]]:
        rank = ranks[span.begin["pid"]]
        restore_jobs[rank].extend(
            {"request": name.split("~", 1)[1], "bytes": read.size}
            for name, read in zip(request_names(span, names), span.reads, strict=True))
    result = {
        "early_ranks": sorted({row["tp_rank"] for row in early.values()
                               if normalized_proxy_id(row["request_id"]) == request}),
        "prepared_plan_reused_ranks": sorted(by_rank_consume),
        "successor": batch_metrics(successor, names, origin),
        "successor_restore_jobs_by_rank": dict(sorted(restore_jobs.items())),
        "scheduler_watch_arrivals_since_P_arrival_ms": [(row["monotonic_ns"] - origin) / MS for row in prefill["native"]
            if row["type"] == "r21_witness_arrival" and normalized_proxy_id(row["request_id"]) == request],
        "scheduler_release_unavailable": [{key: row[key] for key in ("mode", "metadata_name", "candidates") if key in row}
            for row in prefill["native"] if row["type"] == "r21_release_unavailable" and row["batch"] == batch["batch"]],
        "scheduler_releases": [{key: row[key] for key in ("only_restore", "mode", "zero_block_count", "prior_used_block_overlap")
                               if key in row} | {"since_P_arrival_ms": (row["dispatch_ns"] - origin) / MS}
                               for row in releases],
        "plans": [{key: row[key] for key in ("tp_rank", "mode", "zero_overlap_bytes", "staging_bytes",
                   "zeroer_active", "zero_block_count", "total_bytes", "prior_used_block_overlap", "reason") if key in row} | {
                   "prepare_since_P_arrival_ms": (row["prepare_begin_ns"] - origin) / MS,
                   "prepare_ms": (row["prepare_end_ns"] - row["prepare_begin_ns"]) / MS,
                   "safe_live_write_since_P_arrival_ms": (row["safe_live_write_ns"] - origin) / MS
                   if "safe_live_write_ns" in row else None}
                  for row in plans],
        "consumes": [{key: row[key] for key in ("tp_rank", "mode", "wait_ns", "stage_copy_ns", "stage_copy_device_ms",
                      "preparation_saved_ns", "old_forward_device_ms", "used_early_get") if key in row} | {
                      ("stage_scatter_begin_since_P_arrival_ms" if row.get("used_early_get")
                       else "live_page_GET_return_marker_since_P_arrival_ms"): (row["safe_live_write_ns"] - origin) / MS
                      if "safe_live_write_ns" in row else None} for row in consumes],
    }
    if predecessor is None:
        return result
    if len(named) > 1:
        raise ValueError("The witness ranks named different predecessor batches")
    prior_batch = predecessor["batch"]
    result["predecessor"] = batch_metrics(predecessor, names, origin)
    result["rank_paths"] = []
    actual_gets, host_intersections, device_intersections = [], [], []
    for span, read in index["loads_by_request"][request]:
        rank = ranks[span.begin["pid"]]
        early_event = early.get((span.begin["pid"], batch["batch"], request), {})
        if early_event and early_event["tp_rank"] != rank:
            raise ValueError("Early GET rank does not match its successor worker")
        prior_forward = predecessor["forwards"][rank]
        host_begin = max(read.begin, prior_forward["forward_enter_ns"])
        host_end = min(read.end, prior_forward["forward_cpu_return_ns"])
        if host_end > host_begin:
            host_intersections.append((host_begin, host_end))
        probes = {key: early_event[key] for key in PROBES if key in early_event}
        prior_load = predecessor["loads"].get(rank)
        earliest_marker = prior_load.end["monotonic_ns"] if prior_load else prior_forward["worker_enter_ns"]
        lower = device_intersection(early_event, by_rank_consume.get(rank, {}), earliest_marker) if early_event else None
        if lower:
            device_intersections.append(lower)
        actual_gets.append((read.begin, read.end))
        result["rank_paths"].append({
            "rank": rank, "early": bool(early_event), "bytes": read.size,
            "get_start_since_P_arrival_ms": (read.begin - origin) / MS, "get_ms": (read.end - read.begin) / MS,
            "get_return_since_P_arrival_ms": (read.end - origin) / MS,
            "local_copy_return_since_P_arrival_ms": (read.returned - origin) / MS,
            "host_observed_overlap_ms": max(0, host_end - host_begin) / MS, "device_event_probes": probes,
            "device_forward_overlap_lower_ms": (lower[1] - lower[0]) / MS if lower else None,
        })
    prior_reads = [read for span in index["loads_by_batch"][prior_batch["batch"]] for read in span.reads]
    if not actual_gets:
        raise ValueError("The matched witness no longer has its remote read")
    result["rank_paths"].sort(key=lambda row: row["rank"])
    first_get, last_get = min(begin for begin, _ in actual_gets), max(end for _, end in actual_gets)
    result["GET_span_first_submit_to_last_completion_ms"] = (last_get - first_get) / MS
    result["last_GET_completion_minus_first_successor_worker_ms"] = (last_get - first_worker) / MS
    result["get_active_union_ms"] = interval_union(actual_gets) / MS
    result["host_forward_overlap_union_ms"] = interval_union(host_intersections) / MS
    result["device_forward_overlap_lower_union_ms"] = interval_union(device_intersections) / MS
    result["device_forward_overlap_probe_rank_count"] = sum(bool(row["device_event_probes"]) for row in result["rank_paths"])
    offset = batch["unix_ns"] - batch["monotonic_ns"]
    samples = sorted((row for row in decode["cap_port_sample"] if row["case"] == case), key=lambda row: row["unix_ns"])
    result["D_port_counters_during_target_GET"] = port_window(
        samples, first_get + offset, last_get + offset, per_case["D_port_readback_Mbps"], margin)
    other_reads = [(read.begin, read.end)
                   for rid, spans in index["loads_by_request"].items() if rid != request and
                   names.get(rid, "").startswith(f"{case}~") for _, read in spans if read is not None]
    result["other_P_get_overlap_during_target_GET_ms"] = interval_union(
        (max(first_get, begin), min(last_get, end)) for begin, end in other_reads
        if begin < last_get and end > first_get) / MS
    if not prior_reads:
        return result
    old_done = max(read.end for read in prior_reads)
    # The no-demand interval ends at either the next worker or the actual early
    # GET. Keep the previous compute window separate from this NIC gap.
    gap_end = min(first_worker, first_get)
    result["prior_last_get_to_successor_first_worker_ms"] = (first_worker - old_done) / MS
    if gap_end > old_done:
        result["other_P_get_overlap_union_in_gap_ms"] = interval_union(
            (max(old_done, begin), min(gap_end, end)) for begin, end in other_reads
            if begin < gap_end and end > old_done) / MS
        result["D_port_counters_in_pre_GET_gap"] = port_window(
            samples, old_done + offset, gap_end + offset, per_case["D_port_readback_Mbps"], margin)
        nominal = port_window(samples, old_done + offset, gap_end + offset, per_case["D_port_readback_Mbps"], 0)
        result["D_port_counters_in_pre_GET_gap_no_clock_trim"] = {
            key: {name: value for name, value in row.items() if name != "bins"} for key, row in nominal["ports"].items()}
    return result


def analyze(prefill, decode, workload, allow_partial=False, block=None):
    p_scope, = prefill["scope"]
    d_scope, = decode["scope"]
    if p_scope["role"] != "P" or d_scope["role"] != "D" or p_scope["run_id"] != d_scope["run_id"]:
        raise ValueError("Expected the paired P and D run")
    settings = {key: p_scope["active"][key] for key in SETTINGS}
    if settings != {key: d_scope["active"][key] for key in SETTINGS} or not settings["p_local_prefix_caching"]:
        raise ValueError("Run settings differ or local APC was disabled")

    planned = {row["id"]: row for row in workload["requests"]}
    predecessors = {name for target in planned.values() for name in target["predecessors"]}
    manifests = {row["case"]: row for row in decode["client_manifest"]}
    summaries = {row["case"]: row for row in decode["client_summary"]}
    controls = {row["name"]: row["data"] for row in decode["cap_control"]}
    names = client_ids(decode)
    target_rows = {key: row for key, row in manifests.items() if (match := CASE.fullmatch(key))
                   and key in summaries and (block is None or match[2] == block)}
    if not target_rows:
        raise ValueError("No completed timed R21 arms")
    timed_requests = {pid for pid, name in names.items() if name.split("~", 1)[0] in target_rows}
    index, nodes, early = index_restores(prefill, timed_requests)
    margin = clock_margin_ns(prefill, decode)
    runs = defaultdict(dict)
    fixture_by_block = defaultdict(set)
    for case, manifest in sorted(target_rows.items()):
        arm, block, cap = CASE.fullmatch(case).groups()
        if (manifest["phase"], manifest["arm_mode"], manifest["aggregate_D_egress_cap_Gbps"]) != (
                "target", MODES[arm], int(cap)) or not manifest["ignore_eos"] or manifest["background_traffic"] or manifest["profiler_enabled"]:
            raise ValueError(f"Unexpected protocol for {case}")
        prep = manifests[f"{case}-prep"]
        if (manifest["workload_sha256"] != workload["source_workload_sha256"] or
                prep["workload_sha256"] != manifest["workload_sha256"] or
                {row["id"] for row in prep["request_sources"]} != {f"{case}~{name}" for name in predecessors} or
                summaries[f"{case}-prep"]["counts"] != {"ok": len(predecessors)} or
                summaries[f"{case}-prep"]["quality"]["errors"] or
                summaries[f"{case}-prep"]["fixed_output_deviations"] or
                controls[f"{case}-prep-progress.json"]["stage"] != "complete" or
                controls[f"{case}-progress.json"]["stage"] != "complete" or not controls[f"{case}-restore.json"]["restored"]):
            raise ValueError(f"Preparation, manifest, or D shaper restoration invalid for {case}")
        fixture_by_block[cap, block].update((manifest["fixed_replay_sha256"], prep["fixed_replay_sha256"]))
        metrics = case_metrics(prefill, decode, manifest, planned, index)
        metrics["witness"] = attribute_witness(prefill, decode, case, metrics, index, nodes, early, margin)
        metrics["D_full_counter_delta_by_port"] = full_port_deltas(controls, case)
        if arm != "c" and metrics["witness"]["early_ranks"]:
            raise ValueError(f"An early GET ran in control arm {case}")
        runs[cap, block][MODES[arm]] = metrics

    if any(len(hashes) != 1 or not all(hashes) for hashes in fixture_by_block.values()):
        raise ValueError("Fixed fixtures differ within a comparison block")
    comparisons = []
    for (cap, block), arms in sorted(runs.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        if set(arms) != set(MODES.values()) and not allow_partial:
            raise ValueError(f"Incomplete A/B/C block at {cap} Gbps, {block}")
        diffs = {}
        for left, right in (("packed", "prepare-only"), ("prepare-only", "early-get"), ("packed", "early-get")):
            if left not in arms or right not in arms:
                continue
            a, b = arms[left], arms[right]
            physical_fields = ("get_bytes", "get_calls", "native_equivalent_bytes")
            work_changes = [rid for rid in planned if any(
                a["per_request"][rid][field] != b["per_request"][rid][field] for field in
                ("P_local_prefix_tokens", "P_external_prefix_tokens", "P_model_scheduled_tokens"))]
            size_changes = [rid for rid in planned if any(
                a["per_request"][rid][field] != b["per_request"][rid][field] for field in physical_fields)]
            if size_changes or a["P_to_D_descriptor_bytes"] != b["P_to_D_descriptor_bytes"]:
                raise ValueError(f"Physical bytes or call counts changed between {left} and {right}: {size_changes}")
            delta = {field: b[field] - a[field] for field in ("ttft_p99_s", "drain_s", "output_tokens_s")}
            for which in ("predecessor", "successor"):
                stage_a, stage_b = (item["witness"].get(which) for item in (a, b))
                if stage_a and stage_b:
                    delta[which] = {
                        "matched_shape": stage_a["shape"] == stage_b["shape"],
                        **{field: stage_b[field] - stage_a[field] for field in
                           ("dispatch_to_scheduler_result_ms", "scheduler_result_since_P_arrival_ms",
                            "rank0_device_forward_ms") if stage_a[field] is not None and stage_b[field] is not None},
                    }
            delta["P_work_changed_requests"] = work_changes
            delta["critical_requests"] = {}
            for rid in sorted({WITNESS, a["max_ttft_request"], b["max_ttft_request"],
                               a["last_request"], b["last_request"]}):
                per_a, per_b = (item["per_request"][rid] for item in (a, b))
                delta["critical_requests"][rid] = {
                    field: per_b[field] - per_a[field] for field in
                    ("ttft_s", "P_http_ms", "P_queue_ms", "P_dispatch_to_result_ms",
                     "D_first_chunk_ms", "decode_tail_s", "TPOT_ms")
                    if per_a[field] is not None and per_b[field] is not None}
            diffs[f"{right}_minus_{left}"] = delta
        comparisons.append({"D_egress_cap_Gbps": int(cap), "block": block, "arms": arms, "deltas": diffs})
    if not allow_partial and {item["D_egress_cap_Gbps"] for item in comparisons} != {100, 200}:
        raise ValueError("100 and 200 Gbps comparisons are both required")
    return {"run_id": p_scope["run_id"], "settings": settings, "target_requests_per_arm": len(planned),
            "witness": WITNESS, "comparisons": comparisons,
            "interpretation": "Port bins use entirely interior query frames; the clock sensitivity trims at least 42 ms per edge. "
                              "Chrony snapshots are a sensitivity input, not a bound on cross-host sync error. "
                              "Short GET intervals may leave sparse or no port bins and MAC counters include any other port traffic. "
                              "Absent bins do not imply idle. "
                              "Device overlap is a lower bound for ranks with start/end probes. "
                              "Matched time deltas are exploratory and include natural changes in batch shape."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefill", "decode", "workload", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Summarize completed diagnostic arms only.")
    parser.add_argument("--block", help="Select one repeat block, such as b2, when other D traces lack matching P data.")
    args = parser.parse_args()
    if args.block is not None and not re.fullmatch(r"b\d+", args.block):
        parser.error("--block must use the form b2")
    write_json(args.output, analyze(read_bundle(args.prefill), read_bundle(args.decode), read_json(args.workload),
                                    args.allow_partial, args.block))


if __name__ == "__main__":
    main()
