"""Measure the exact P-side ADD queue boundary in fixed-cohort captures."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import normalized_proxy_id, read_bundle
from pd_kv_transfer_opt.profiling.analyze_natural_restore_ab import SETTINGS, match_targets, route
from pd_kv_transfer_opt.profiling.analyze_packed_readiness import join_forwards
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import stats
from pd_kv_transfer_opt.profiling.analyze_same_head_ab import cap_readback, port_deltas
from pd_kv_transfer_opt.profiling.audit_prefill_admission import MS, clip_ms, observed_gets


CASE = re.compile(r"([ab])-(b\d+)-cap(100|200)")
COMPLETION_ID = re.compile(r"^(?:cmpl|chatcmpl)-([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})(?:-|$)")
REQUEST_EVENTS = (
    "p_admission_http", "p_admission_api", "p_admission_frontend_add", "p_admission_ipc_send",
    "p_admission_ipc_recv", "p_admission_enqueue", "p_admission_dequeue", "p_admission_early_add",
)
PATH_TIMES = {
    "HTTP_body_ms": ("http_enter", "http_body_end"),
    "HTTP_body_end_to_API_begin_ms": ("http_body_end", "api_begin"),
    "P_HTTP_enter_to_ADD_enqueue_ms": ("http_enter", "enqueue"),
    "API_processing_ms": ("api_begin", "api_end"),
    "API_before_ADD_submit_ms": ("api_begin", "send_begin"),
    "frontend_add_ms": ("frontend_begin", "frontend_end"),
    "ADD_send_call_ms": ("send_begin", "send_return"),
    "ADD_submit_to_decoded_ms": ("send_begin", "decode_end"),
    "ADD_receiver_preprocess_ms": ("decode_end", "preprocess_end"),
    "ADD_decode_to_enqueue_ms": ("decode_end", "enqueue"),
    "ADD_preprocess_to_enqueue_ms": ("preprocess_end", "enqueue"),
    "ADD_enqueue_to_dequeue_ms": ("enqueue", "dequeue"),
    "ADD_dequeue_to_scheduler_ms": ("dequeue", "admit"),
    "scheduler_to_first_dispatch_ms": ("admit", "first_dispatch"),
    "ADD_enqueue_to_first_dispatch_ms": ("enqueue", "first_dispatch"),
    "ADD_enqueue_to_rank0_worker_ms": ("enqueue", "first_rank0_worker"),
    "ADD_enqueue_to_first_worker_ms": ("enqueue", "first_worker"),
    "ADD_enqueue_to_last_host_forward_ms": ("enqueue", "last_host_forward"),
    "API_begin_to_first_dispatch_ms": ("api_begin", "first_dispatch"),
    "P_HTTP_enter_to_first_dispatch_ms": ("http_enter", "first_dispatch"),
    "P_HTTP_enter_to_first_worker_ms": ("http_enter", "first_worker"),
    "P_HTTP_enter_to_last_host_forward_ms": ("http_enter", "last_host_forward"),
    "P_first_dispatch_to_rank0_worker_ms": ("first_dispatch", "first_rank0_worker"),
    "P_first_dispatch_to_first_worker_ms": ("first_dispatch", "first_worker"),
    "P_first_worker_to_last_host_forward_ms": ("first_worker", "last_host_forward"),
    "P_last_host_forward_to_first_result_ms": ("last_host_forward", "first_result"),
    "P_first_dispatch_to_last_result_ms": ("first_dispatch", "last_result"),
    "P_first_batch_GET_span_ms": ("first_batch_GET_begin", "first_batch_GET_end"),
}


def request_id(value):
    match = COMPLETION_ID.match(value or "")
    return normalized_proxy_id(value) or (match[1] if match else value)


def latency(origin, end):
    return (end - origin) / MS if origin is not None and end is not None else None


def batch_identity(batch):
    return batch["instance"], batch["pid"], batch["batch"], batch["dispatch_ns"]


def analyze_events(prefill):
    events = prefill["native"]
    staged = {(row["pid"], row["batch"], request_id(row["request_id"])) for row in events
              if row["type"] == "r21_early_get"}
    stock_events = prefill if not staged else {**prefill, "native": [
        row for row in events if row["type"] != "same_head_local_copy"
        or (row["pid"], row["batch"], request_id(row.get("request_id"))) not in staged
    ]}
    batches, joined, errors, unmatched_loads = join_forwards(stock_events)
    if any(kind == "model_output_without_dispatch" for kind, _ in errors):
        raise ValueError("Incomplete P dispatch/result join")
    if unmatched_loads and prefill["scope"][0]["run_id"].endswith("-r24-prefill-admission"):
        raise ValueError(f"R24 has {len(unmatched_loads)} worker loads without a unique P batch/rank")
    chunks, instance_batches, waits, updates, requests = (defaultdict(list) for _ in range(5))
    arrivals = {request_id(row["request_id"]): row for row in events if row["type"] == "p_chunk_arrival"}
    staged_batches = {batch for _, batch, _ in staged}
    worker_paths = {batch_identity(node["batch"]): first_worker_path(node, node["batch"]["batch"] not in staged_batches)
                    for node in joined}
    for batch in sorted(batches, key=lambda row: row["dispatch_ns"]):
        instance_batches[batch["instance"], batch["pid"]].append(batch)
        for rid in {request_id(rid) for rid, _, _ in batch["rows"]}:
            chunks[rid].append(batch)
    for row in sorted(events, key=lambda item: item["monotonic_ns"]):
        kind = row["type"]
        if kind in REQUEST_EVENTS:
            requests[request_id(row["request_id"]), kind].append(row)
        elif kind == "p_admission_result_wait":
            if row["wait_end_ns"] < row["wait_begin_ns"]:
                raise ValueError("P result wait moved backward")
            waits[row["instance"], row["pid"]].append(row)
        elif kind == "p_admission_result_update":
            if row["update_end_ns"] < row["update_begin_ns"]:
                raise ValueError("P result update moved backward")
            updates[row["instance"], row["pid"], row["batch"]].append(row)
    return arrivals, chunks, instance_batches, waits, updates, requests, worker_paths


def first_worker_path(node, synchronous):
    ranks = node["forwards"]
    if any(row["worker_enter_ns"] > row["forward_enter_ns"] for row in ranks.values()):
        raise ValueError("P rank worker entry followed host forward")
    loads = node["loads"]
    reads = [read for load in loads.values() for read in load.reads]
    if any(not ranks[rank]["worker_enter_ns"] <= read.begin <= read.end <= ranks[rank]["forward_enter_ns"]
           for rank, load in loads.items() for read in load.reads):
        raise ValueError("First-batch GET is outside the joined rank's pre-forward interval")
    stamps = {
        "first_worker": min(row["worker_enter_ns"] for row in ranks.values()),
        "last_host_forward": max(row["forward_enter_ns"] for row in ranks.values()),
    }
    if reads:
        stamps.update(first_batch_GET_begin=min(read.begin for read in reads),
                      first_batch_GET_end=max(read.end for read in reads))
    if synchronous:
        return stamps, {"rank_forward_count": len(ranks), "GET_calls": len(reads),
                        "GET_payload_bytes": sum(read.size for read in reads)}
    return {key: stamps[key] for key in ("first_worker", "last_host_forward")}, {"rank_forward_count": len(ranks)}


def choose_request_events(events, rid, instance):
    result = {}
    for kind in REQUEST_EVENTS:
        options = [row for row in events.get((rid, kind), []) if row["instance"] == instance]
        if len(options) == 1:
            result[kind] = options[0]
        elif options:
            raise ValueError(f"Ambiguous {kind} for {rid}")
    return result


def previous_result(intervals, start, stop):
    if start is None or stop is None:
        return None
    matches = [row for row in intervals if row.get("future_done_at_entry") is False
               and start < row["wait_end_ns"] and row["wait_begin_ns"] < stop]
    return max(matches, key=lambda row: min(stop, row["wait_end_ns"]) - max(start, row["wait_begin_ns"]),
               default=None)


BATCH_PHASES = ("dispatch_to_worker_ms", "worker_to_last_host_forward_ms",
                "last_host_forward_to_result_update_ms", "GET_span_ms")


def batch_snapshot(batch, names, enqueue, worker_paths):
    if batch is None:
        return None
    snapshot = {
        "batch": batch["batch"],
        "instance": batch["instance"],
        "shape": sorted((names.get(request_id(rid), request_id(rid)).split("~", 1)[-1], offset, tokens)
                        for rid, offset, tokens in batch["rows"]),
        "scheduled_tokens": batch["total"],
        "dispatch_since_ADD_enqueue_ms": latency(enqueue, batch["dispatch_ns"]),
        "result_update_since_ADD_enqueue_ms": latency(enqueue, batch.get("ready_ns")),
        "dispatch_to_result_update_ms": latency(batch["dispatch_ns"], batch.get("ready_ns")),
        "rank0_device_forward_ms": batch.get("device", {}).get("forward_device_ms"),
    }
    if worker_path := worker_paths.get(batch_identity(batch)):
        stamps, metadata = worker_path
        snapshot.update(metadata)
        snapshot.update(dict(zip(BATCH_PHASES, (
            latency(batch["dispatch_ns"], stamps["first_worker"]),
            latency(stamps["first_worker"], stamps["last_host_forward"]),
            latency(stamps["last_host_forward"], batch.get("ready_ns")),
            latency(stamps.get("first_batch_GET_begin"), stamps.get("first_batch_GET_end")),
        ), strict=True)))
    return snapshot


def request_metrics(rid, end, proxy, names, index, gets, read_bytes):
    arrivals, chunks, all_batches, waits, updates, requests, worker_paths = index
    arrival = arrivals[rid]
    instance = arrival["instance"]
    first, last = chunks[rid][0], chunks[rid][-1]
    key = instance, arrival["pid"]
    if any((row["instance"], row["pid"]) != key for row in chunks[rid]) or "ready_ns" not in last:
        raise ValueError(f"Incomplete or migrating P chunks for {end['id']}")
    rows = choose_request_events(requests, rid, instance)
    stamps = {out: row[source] for row in rows.values() for source, out in (
        ("http_enter_ns", "http_enter"), ("http_body_end_ns", "http_body_end"),
        ("api_begin_ns", "api_begin"), ("api_end_ns", "api_end"),
        ("frontend_begin_ns", "frontend_begin"), ("frontend_end_ns", "frontend_end"),
        ("send_begin_ns", "send_begin"), ("send_return_ns", "send_return"),
        ("decode_end_ns", "decode_end"), ("preprocess_end_ns", "preprocess_end"),
    ) if source in row}
    for label in ("enqueue", "dequeue"):
        if (row := rows.get(f"p_admission_{label}")) is not None:
            if row["pid"] != arrival["pid"]:
                raise ValueError("The ADD queue and scheduler have different owners")
            stamps[label] = row["event_ns"]
    stamps.update(admit=arrival["arrival_ns"], first_dispatch=first["dispatch_ns"], last_result=last["ready_ns"])
    if device := first.get("device"):
        stamps["first_rank0_worker"] = device["worker_enter_ns"]
    first_chunk = batch_snapshot(first, names, stamps.get("enqueue"), worker_paths)
    if worker_path := worker_paths.get(batch_identity(first)):
        worker_stamps, _ = worker_path
        stamps.update(worker_stamps, first_result=first["ready_ns"])
    spans = {label: latency(stamps.get(start), stamps.get(stop)) for label, (start, stop) in PATH_TIMES.items()}
    inversions = [label for label, duration in spans.items() if duration is not None and duration < 0]
    for label in inversions:
        spans[label] = None

    enqueue, dequeue = stamps.get("enqueue"), stamps.get("dequeue")
    result_wait = previous_result(waits.get(key, ()), enqueue, dequeue)
    older = [row for row in all_batches[key] if row["dispatch_ns"] < first["dispatch_ns"]]
    prior = max((row for row in older if enqueue is None or row["dispatch_ns"] <= enqueue),
                key=lambda row: row["dispatch_ns"], default=None)
    if result_wait is not None and result_wait.get("batch"):
        matching = [row for row in older if row["batch"] == result_wait["batch"]
                    and row["dispatch_ns"] <= result_wait["wait_begin_ns"]]
        prior = matching[-1] if matching else prior
    prior_result = prior.get("ready_ns") if prior else None
    overlap = 0.0
    result_update = None
    if result_wait is not None and dequeue >= enqueue:
        overlap = clip_ms(((row["wait_begin_ns"], row["wait_end_ns"]) for row in waits[key]
                           if row.get("future_done_at_entry") is False), enqueue, dequeue)
        if prior is not None and prior["batch"] == result_wait.get("batch"):
            result_update = next((row for row in updates[(*key, prior["batch"])]
                                  if result_wait["wait_end_ns"] <= row["update_begin_ns"] <= first["dispatch_ns"]),
                                 None)
    verified = bool(result_update and enqueue < result_update["update_begin_ns"]
                    <= result_update["update_end_ns"] <= dequeue)
    if result_wait is not None:
        stamps.update(prior_wait_begin=result_wait["wait_begin_ns"], prior_wait_end=result_wait["wait_end_ns"])
    if result_update is not None:
        stamps.update(prior_update_begin=result_update["update_begin_ns"], prior_update_end=result_update["update_end_ns"])
    early = rows.get("p_admission_early_add")
    if early is not None:
        if (early["pid"] != arrival["pid"] or dequeue is None or not
                dequeue <= early["begin_ns"] <= arrival["arrival_ns"] <= early["end_ns"]):
            raise ValueError("Early ADD did not use the original P Scheduler.add_request path")
        stamps.update(early_add_begin=early["begin_ns"], early_add_end=early["end_ns"])
    start = stamps.get("http_enter", stamps.get("api_begin", stamps.get("send_begin", stamps["admit"])))
    out = end["completion_tokens"]
    prefill_begin, prefill_end = proxy["prefill_begin"], proxy["prefill_end"]
    decode_begin, decode_first = proxy["decode_begin"], proxy["decode_first_chunk"]
    return {
        "instance": instance,
        "P_timeline_ms_from_first_API_marker": {label: latency(start, stamp) for label, stamp in stamps.items()},
        "P_spans_ms": spans, "P_invalid_timing_spans": inversions,
        "P_prior_result_wait_overlap_with_ADD_queue_ms": overlap,
        "P_prior_result_wait": {field: result_wait[field] for field in (
            "batch", "batch_queue_size", "batch_queue_pending", "future_done_at_entry"
        )} if result_wait is not None else None,
        "P_prior_result_update_to_ADD_dequeue_ms": latency(prior_result, dequeue),
        "P_prior_result_update_end_to_ADD_dequeue_ms": latency(
            result_update["update_end_ns"] if result_update else None, dequeue),
        "P_prior_result_update_ms": latency(result_update["update_begin_ns"], result_update["update_end_ns"])
        if result_update else None,
        "P_ADD_queued_during_wait_and_dequeued_after_update": verified,
        "P_early_ADD": {"batch": early["batch"],
                        "handler_ms": latency(early["begin_ns"], early["end_ns"]),
                        "result_wait_pending_at_scheduler_admission": bool(
                            result_wait and early["batch"] == result_wait["batch"]
                            and stamps["admit"] < result_wait["wait_end_ns"])}
        if early else None,
        "P_GET_overlap_with_ADD_queue_ms": clip_ms(gets[instance], enqueue, dequeue)
        if enqueue is not None and dequeue is not None else None,
        "P_predecessor_batch": batch_snapshot(prior, names, enqueue, worker_paths),
        "P_first_chunk_batch": first_chunk,
        "TTFT_ms": (end["first_s"] - end["ready_s"]) * 1000,
        "P_HTTP_on_D_clock_ms": latency(prefill_begin["monotonic_ns"], prefill_end["monotonic_ns"]),
        "D_to_first_chunk_ms": latency(decode_begin["monotonic_ns"], decode_first["monotonic_ns"]),
        "D_TPOT_ms": (end["end_s"] - end["first_s"]) * 1000 / (out - 1) if out > 1 else None,
        "D_output_tail_ms": (end["end_s"] - end["first_s"]) * 1000,
        "P_GET_payload_bytes": read_bytes[rid],
        "output_tokens": out,
    }


def cohort_plan(case, manifest, decode):
    routes = defaultdict(dict)
    for row in decode["proxy"]:
        if row["id"].startswith(case + "~") and row["type"] in ("prefill_begin", "decode_begin"):
            routes[row["id"]][row["type"]] = route(row)
    return {row["id"].split("~", 1)[1]: {
        "offered_s": row["offered_s"], "max_tokens": row["max_tokens"],
        "prefill": routes[row["id"]]["prefill_begin"], "decode": routes[row["id"]]["decode_begin"],
    } for row in manifest["request_sources"]}


def collect_case(decode, case, manifest, planned, summaries, names, index, gets, reads):
    ends, proxy = match_targets(decode, manifest, planned)
    paths = {name: request_metrics(proxy[name]["prefill_begin"]["proxy_request_id"], end, proxy[name],
                                  names, index, gets, reads) for name, end in ends.items()}
    summary = summaries[case]
    if (not manifest["ignore_eos"] or manifest["background_traffic"] or manifest["profiler_enabled"]
            or summary["counts"] != {"ok": len(planned)} or summary["quality"]["errors"]
            or summary.get("fixed_output_deviations")):
        raise ValueError(f"Unexpected workload settings or incomplete client accounting in {case}")
    cap = manifest["aggregate_D_egress_cap_Gbps"]
    readback = cap_readback(decode, case, cap, range(8))
    samples = sorted((row for row in decode["cap_port_sample"] if row["case"] == case), key=lambda row: row["unix_ns"])
    by_port = port_deltas(samples)[0] if len(samples) >= 2 else {}
    footprint = {name: (planned[name]["offered_s"], planned[name]["max_tokens"], planned[name]["prefill"],
                        planned[name]["decode"], end["prompt_tokens"]) for name, end in ends.items()}
    duration = summary["finite_cohort"]["drain_included_s"]
    output = sum(path["output_tokens"] for path in paths.values())
    max_ttft = max(paths, key=lambda name: paths[name]["TTFT_ms"])
    drain_tail = max(ends, key=lambda name: ends[name]["end_s"])
    return {
        "scope": {k: manifest.get(k) for k in ("arm_mode", "workload_sha256", "fixed_replay_sha256",
                                             "ignore_eos", "background_traffic", "profiler_enabled",
                                             "admission_mode")},
        "offered_fixture": footprint, "requests": len(paths), "output_tokens": output,
        "drain_s": duration, "output_tokens_s": output / duration,
        "TTFT_ms": stats(path["TTFT_ms"] for path in paths.values()),
        "D_TPOT_ms": stats(path["D_TPOT_ms"] for path in paths.values() if path["D_TPOT_ms"] is not None),
        "D_port_cap_readback_Mbps": readback, "D_port_counters": by_port,
        "P_GET_payload_bytes": sum(path["P_GET_payload_bytes"] for path in paths.values()),
        "P_max_TTFT_request": max_ttft, "D_drain_tail_request": drain_tail,
        "P_ADD_queue_complete_count": sum(
            path["P_spans_ms"]["ADD_enqueue_to_dequeue_ms"] is not None for path in paths.values()),
        "P_all_rank_first_chunk_count": sum(path["P_first_chunk_batch"].get("rank_forward_count") == 4
                                           for path in paths.values()),
        "P_early_ADD_count": sum(path["P_early_ADD"] is not None for path in paths.values()),
        "P_ADD_queued_during_wait_and_dequeued_after_update_count": sum(
            path["P_ADD_queued_during_wait_and_dequeued_after_update"] for path in paths.values()),
        "P_path_stats_ms": {label: stats(path["P_spans_ms"][label] for path in paths.values()
                                        if path["P_spans_ms"][label] is not None) for label in PATH_TIMES},
        "per_request": paths,
    }


def pair_results(native, modified):
    identical = native["offered_fixture"] == modified["offered_fixture"]
    mode_fields = {"arm_mode", "admission_mode"}
    scope = {k: val for k, val in native["scope"].items() if k not in mode_fields}
    matched = (identical and scope == {k: val for k, val in modified["scope"].items() if k not in mode_fields}
               and native["D_port_cap_readback_Mbps"] is not None
               and native["D_port_cap_readback_Mbps"] == modified["D_port_cap_readback_Mbps"])
    comparison = {"fixed_fixture_and_cap_verified": matched, "predecessor_batches": [], "per_request": {}}
    if not matched:
        return comparison
    matched_predecessors = set()
    for name, base in native["per_request"].items():
        trial = modified["per_request"][name]
        base_chunk, trial_chunk = base["P_first_chunk_batch"], trial["P_first_chunk_batch"]
        base_batch_bytes, trial_batch_bytes = (chunk.get("GET_payload_bytes") for chunk in (base_chunk, trial_chunk))
        comparison["per_request"][name] = {
            "baseline_proven_P_ADD_result_wait": base["P_ADD_queued_during_wait_and_dequeued_after_update"],
            "intervention_P_early_ADD": trial["P_early_ADD"] is not None,
            "TTFT_delta_ms": trial["TTFT_ms"] - base["TTFT_ms"],
            "D_TPOT_delta_ms": trial["D_TPOT_ms"] - base["D_TPOT_ms"]
            if base["D_TPOT_ms"] is not None and trial["D_TPOT_ms"] is not None else None,
            "P_physical_GET_delta_bytes": trial["P_GET_payload_bytes"] - base["P_GET_payload_bytes"],
            "P_first_chunk_shape_matches": (base_chunk["instance"], base_chunk["shape"])
            == (trial_chunk["instance"], trial_chunk["shape"]),
            "P_first_chunk_GET_payload_delta_bytes": trial_batch_bytes - base_batch_bytes
            if base_batch_bytes is not None and trial_batch_bytes is not None else None,
            "P_path_delta_ms": {label: trial["P_spans_ms"][label] - val
                                for label, val in base["P_spans_ms"].items()
                                if val is not None and trial["P_spans_ms"][label] is not None},
        }
        a, b = base["P_predecessor_batch"], trial["P_predecessor_batch"]
        if (a is None or b is None or (a["instance"], a["shape"]) != (b["instance"], b["shape"])
                or a["dispatch_to_result_update_ms"] is None or b["dispatch_to_result_update_ms"] is None):
            continue
        key = a["instance"], a["batch"], b["batch"]
        if key in matched_predecessors:
            continue
        matched_predecessors.add(key)
        comparison["predecessor_batches"].append({
            "instance": a["instance"], "shape": a["shape"], "native_batch": a["batch"], "modified_batch": b["batch"],
            "dispatch_to_result_update_delta_ms": b["dispatch_to_result_update_ms"] - a["dispatch_to_result_update_ms"],
            "rank0_device_forward_delta_ms": b["rank0_device_forward_ms"] - a["rank0_device_forward_ms"]
            if a["rank0_device_forward_ms"] is not None and b["rank0_device_forward_ms"] is not None else None,
            "execution_phases_delta_ms": {label: b[label] - a[label] for label in BATCH_PHASES
                                          if a.get(label) is not None and b.get(label) is not None},
            "GET_payload_delta_bytes": b["GET_payload_bytes"] - a["GET_payload_bytes"]
            if a.get("GET_payload_bytes") is not None and b.get("GET_payload_bytes") is not None else None,
        })
    for label in ("output_tokens_s", "output_tokens", "drain_s", "P_GET_payload_bytes"):
        comparison[label + "_delta"] = modified[label] - native[label]
    comparison["TTFT_p99_delta_ms"] = modified["TTFT_ms"]["p99"] - native["TTFT_ms"]["p99"]
    comparison["physical_GET_payloads_fixed"] = all(
        not row["P_physical_GET_delta_bytes"] for row in comparison["per_request"].values())
    comparison["matched_first_chunk_shape_count"] = sum(
        row["P_first_chunk_shape_matches"] for row in comparison["per_request"].values())
    comparison["P_path_delta_stats_ms"] = {
        label: stats(row["P_path_delta_ms"][label] for row in comparison["per_request"].values()
                     if label in row["P_path_delta_ms"])
        for label in PATH_TIMES
    }
    exercised = [row for row in comparison["per_request"].values()
                 if row["baseline_proven_P_ADD_result_wait"] and row["intervention_P_early_ADD"]]
    comparison["paired_requests_with_baseline_wait_and_intervention"] = len(exercised)
    comparison["P_path_delta_for_exercised_requests_ms"] = {
        label: stats(row["P_path_delta_ms"][label] for row in exercised if label in row["P_path_delta_ms"])
        for label in PATH_TIMES
    }
    return comparison


def analyze(prefill, decode, pattern=CASE, workload=None):
    p_scope, = prefill["scope"]
    d_scope, = decode["scope"]
    if (p_scope["role"], d_scope["role"]) != ("P", "D") or p_scope["run_id"] != d_scope["run_id"]:
        raise ValueError("Expected paired P and D capture roles from the same transport epoch")
    settings = {key: p_scope.get("active", {}).get(key) for key in SETTINGS}
    if settings != {key: d_scope.get("active", {}).get(key) for key in SETTINGS}:
        raise ValueError("P and D captures disagree on deployment settings")
    summaries = {row["case"]: row for row in decode["client_summary"]}
    manifests = {row["case"]: row for row in decode["client_manifest"]
                 if row.get("phase") == "target" and pattern.fullmatch(row["case"]) and row["case"] in summaries}
    if not manifests:
        raise ValueError("No completed matching timed cases")
    names = {row["proxy_request_id"]: row["id"] for row in decode["proxy"] if row["type"] == "prefill_begin"}
    index = analyze_events(prefill)
    gets = observed_gets(prefill["native"])
    read_bytes = Counter()
    for row in prefill["native"]:
        if row["type"] == "store_end" and row.get("operation") == "batch_get_into_multi_buffers":
            read_bytes[request_id(row.get("request_id"))] += row["payload_bytes"]
    plan = {row["id"]: row for row in workload["requests"]} if workload else None
    if workload:
        predecessors = {name for target in plan.values() for name in target["predecessors"]}
        preparation = {row["case"]: row for row in decode["client_manifest"] if row["case"].endswith("-prep")}
        for case, timed in manifests.items():
            prep, summary = preparation[f"{case}-prep"], summaries[f"{case}-prep"]
            if (timed["workload_sha256"] != workload["source_workload_sha256"]
                    or prep["workload_sha256"] != timed["workload_sha256"]
                    or prep["fixed_replay_sha256"] != timed["fixed_replay_sha256"]
                    or {row["id"] for row in prep["request_sources"]} != {f"{case}~{name}" for name in predecessors}
                    or summary["counts"] != {"ok": len(predecessors)} or summary["quality"]["errors"]
                    or summary.get("fixed_output_deviations")):
                raise ValueError(f"Prepared session history differs from the supplied plan in {case}")
    cases = {case: collect_case(decode, case, manifest, plan or cohort_plan(case, manifest, decode),
                               summaries, names, index, gets, read_bytes) for case, manifest in sorted(manifests.items())}
    comparisons = {case[2:]: pair_results(base, cases[f"b{case[1:]}"])
                   for case, base in cases.items() if case.startswith("a-") and f"b{case[1:]}" in cases}
    return {
        "run_id": p_scope["run_id"],
        "settings": settings,
        "P_admission_event_counts": dict(Counter(row["type"] for row in prefill["native"]
                                                if row["type"].startswith("p_admission_"))),
        "measurement": "P phase spans use one host's monotonic clock. D/client timings use their local clock. "
                       "Paired deltas are B minus A. Predecessor deltas require equal stable batch rows. "
                       "First worker is the earliest of the four ranks, last host forward is the latest. "
                       "First-batch GET spans include gaps between recorded native calls. "
                       "The pending batch count excludes the future just popped for the result wait. "
                       "ADD queue attribution requires overlap with the matched prior worker wait and dequeue "
                       "after the scheduler finishes its result update.",
        "cases": cases,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("prefill", "decode", "output"):
        parser.add_argument(label, type=Path)
    parser.add_argument("--cases", type=re.compile, default=CASE)
    parser.add_argument("--workload", type=Path)
    args = parser.parse_args()
    workload = read_json(args.workload) if args.workload else None
    report = analyze(read_bundle(args.prefill), read_bundle(args.decode), args.cases, workload)
    report["source_sha256"] = {"P": sha256(args.prefill), "D": sha256(args.decode)}
    write_json(args.output, report)


if __name__ == "__main__":
    main()
