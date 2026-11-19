"""Audit how r20 same-batch overlap propagates to matched client requests."""

import argparse
from collections import defaultdict
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, write_json
from pd_kv_transfer_opt.profiling.analyze_full_rdma import associate_transfers
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import normalized_proxy_id, read_bundle
from pd_kv_transfer_opt.profiling.analyze_natural_restore_ab import CASE
from pd_kv_transfer_opt.profiling.analyze_packed_overlap import MODES
from pd_kv_transfer_opt.profiling.analyze_packed_readiness import join_forwards, replay, request_names
from pd_kv_transfer_opt.profiling.analyze_r19_copy_opportunities import pipelined_forward_ns


MS = 1_000_000


def client_path(end, origin, proxy, ranks, d_lookup, measured, p_arrival, p_batches):
    """All outer boundaries are recorded on D's monotonic clock."""
    if len(ranks) != 4:
        raise ValueError(f"Expected four completed D transfer handlers for {end['id']}")
    timestamps = [round((origin + end["ready_s"]) * 1e9), round((origin + end["dispatch_s"]) * 1e9),
                  *(proxy[kind]["monotonic_ns"] for kind in
                    ("proxy_arrival", "prefill_begin", "prefill_end", "decode_begin")),
                  min(row["enqueue_ns"] for row in ranks), max(row["end_ns"] for row in ranks),
                  proxy["decode_first_chunk"]["monotonic_ns"], round((origin + end["first_s"]) * 1e9),
                  round((origin + end["end_s"]) * 1e9)]
    names = ("client_ready_to_dispatch", "client_to_proxy_ingress", "proxy_ingress_to_P_start", "P_HTTP",
             "P_end_to_D_start", "D_start_to_transfer_enqueue", "D_transfer_earliest_enqueue_to_last_end",
             "D_last_transfer_end_to_first_chunk", "proxy_first_chunk_to_client_content", "D_output_tail")
    stages = {name: (end_ns - begin_ns) / MS for name, begin_ns, end_ns in zip(names, timestamps, timestamps[1:])}
    # The proxy's first-chunk marker can follow the first client callback by a small amount.
    # Keep signed timestamp differences so the accounting stays exact.
    if any(delta < 0 for name, delta in stages.items() if name != "proxy_first_chunk_to_client_content"):
        raise ValueError(f"Unexpected D clock or stage order for {end['id']}")
    p_queue = measured["P_queue_ms"]
    p_execution = measured["P_dispatch_to_result_ms"]
    p_residual = stages["P_HTTP"] - p_queue - p_execution
    last_batch = p_batches[-1]["batch"]
    arrival_unix = p_arrival["unix_ns"] + p_arrival["arrival_ns"] - p_arrival["monotonic_ns"]
    result_unix = last_batch["unix_ns"] + last_batch["ready_ns"] - last_batch["monotonic_ns"]
    ingress_ms = (arrival_unix - proxy["prefill_begin"]["unix_ns"]) / MS
    response_ms = (proxy["prefill_end"]["unix_ns"] - result_unix) / MS
    d_start = proxy["decode_begin"]["monotonic_ns"]
    d_enqueue = min(row["enqueue_ns"] for row in ranks)
    d_match = next(row for row in d_lookup if d_start <= row["monotonic_ns"] <= d_enqueue)
    return {"TTFT_ms": sum(stages[name] for name in names[:-1]), "outer_ms": stages,
            "P_HTTP_breakdown_ms": {"P_queue": p_queue, "first_dispatch_to_last_ready": p_execution,
                                    "P_HTTP_unattributed_edges": p_residual},
            "P_HTTP_edges_estimated_with_unix_anchors_ms": {"P_send_to_engine_admission": ingress_ms,
                                                           "last_scheduler_result_to_P_HTTP": response_ms},
            "P_batch_GET_union_ms": measured["P_batch_GET_active_union_ms"],
            "P_batch_load_union_ms": measured["P_batch_load_active_union_ms"],
            "D_start_to_transfer_enqueue_breakdown_ms": {
                "D_start_to_pool_match_begin": (d_match["monotonic_ns"] - d_start - d_match["elapsed_ns"]) / MS,
                "D_pool_match_method": d_match["elapsed_ns"] / MS,
                "D_pool_match_return_to_enqueue": (d_enqueue - d_match["monotonic_ns"]) / MS},
            "D_transfer_slowest_submit_to_end_ms": max(row["submit_to_end_ms"] for row in ranks)}


def relevant_batch(node, origin, names):
    batch = node["batch"]
    loads = node["loads"]
    forwards = node["forwards"].values()
    order = request_names(loads[0], names) if loads else ()
    if loads and (len(loads) != 4 or not order or any(request_names(load, names) != order for load in loads.values())):
        raise ValueError("Unmatched measured rank load order")
    result = {"instance": batch["instance"], "batch": batch["batch"], "scheduled_tokens": batch["total"],
              "rows": sorted([names[normalized_proxy_id(rid)], offset, tokens] for rid, offset, tokens in batch["rows"]),
              "restore_order": list(order), "dispatch_ms": (batch["dispatch_ns"] - origin) / MS,
              "latest_worker_entry_ms": (max(row["worker_enter_ns"] for row in forwards) - origin) / MS,
              "latest_forward_ms": (max(row["forward_enter_ns"] for row in forwards) - origin) / MS,
              "scheduler_ready_ms": (batch["ready_ns"] - origin) / MS,
              "latest_rank_worker_to_forward_ms": (max(row["forward_enter_ns"] for row in forwards)
                                                  - max(row["worker_enter_ns"] for row in forwards)) / MS,
              "rank0_device_forward_ms": batch.get("device", {}).get("forward_device_ms"),
              "worker_ranks": {rank: {
                  "GET_sum_ms": sum(read.end - read.begin for read in load.reads) / MS,
                  "between_GET_sum_ms": sum(right.begin - left.end
                                            for left, right in zip(load.reads, load.reads[1:])) / MS,
              } for rank, load in sorted(loads.items())}}
    return result


def analyze(prefill, decode, summary):
    scopes = [data["scope"][0]["run_id"] for data in (prefill, decode)]
    if set(scopes) != {summary["run_id"]}:
        raise ValueError("Profiler and matched summary run IDs differ")
    clients = defaultdict(dict)
    origins = {}
    proxies = defaultdict(dict)
    names = {}
    request_cases = {}
    for row in decode["client"]:
        if not CASE.fullmatch(row["case"]):
            continue
        if row["type"] == "client_origin":
            origins[row["case"]] = row["monotonic_s"]
        elif row["type"] == "request_end":
            clients[row["case"]][row["id"].split("~", 1)[1]] = row
    for row in decode["proxy"]:
        case, _, name = row["id"].partition("~")
        if CASE.fullmatch(case):
            proxies[row["id"]][row["type"]] = row
            if row["type"] == "prefill_begin":
                names[row["proxy_request_id"]] = name
                request_cases[row["proxy_request_id"]] = case
    transfers = associate_transfers(decode["native"])
    d_lookups = defaultdict(list)
    for row in sorted(decode["native"], key=lambda row: row["monotonic_ns"]):
        if row["type"] == "pool_match_tokens":
            d_lookups[normalized_proxy_id(row["request_id"])].append(row)
    batches, joined, errors, unmatched = join_forwards(prefill)
    if errors or unmatched or len(joined) != len(batches):
        raise ValueError("Incomplete paired profiler coverage")
    by_case = defaultdict(list)
    by_request = defaultdict(list)
    arrivals = {normalized_proxy_id(row["request_id"]): row for row in prefill["native"]
                if row["type"] == "p_chunk_arrival"}
    for node in joined:
        ids = [normalized_proxy_id(rid) for rid, _, _ in node["batch"]["rows"]]
        cases = {request_cases[rid] for rid in ids if rid in request_cases}
        if cases:
            case, = cases
            if any(request_cases.get(rid) != case for rid in ids):
                raise ValueError("Prefill batch mixes timed replay cases or contains an untracked request")
            by_case[case].append(node)
            for rid in ids:
                by_request[rid].append(node)

    output = []
    for pair in summary["paired_rounds"]:
        cap, block = pair["D_egress_cap_Gbps"], pair["block"]
        record = {"D_egress_cap_Gbps": cap, "block": block, "arms": {}}
        for letter, mode in zip("ab", MODES, strict=True):
            case = f"{letter}-{block}-cap{cap}"
            values = pair[mode]
            nodes = by_case[case]
            dispatch_origins = {instance: min(node["batch"]["dispatch_ns"] for node in nodes
                                              if node["batch"]["instance"] == instance)
                                for instance in {node["batch"]["instance"] for node in nodes}}
            requests = {}
            for name, end in clients[case].items():
                proxy = proxies[end["id"]]
                pid = proxy["prefill_begin"]["proxy_request_id"]
                req = client_path(end, origins[case], proxy, transfers[pid], d_lookups[pid], values["per_request"][name],
                                  arrivals[pid], by_request[pid])
                if abs(req["TTFT_ms"] - values["per_request"][name]["ttft_s"] * 1000) > 0.01:
                    raise ValueError(f"TTFT stage identity failed: {end['id']}")
                requests[name] = req
            arm_batches = []
            for node in nodes:
                entry = relevant_batch(node, dispatch_origins[node["batch"]["instance"]], names)
                if letter == "a" and len(entry["restore_order"]) > 1:
                    entry["fixed_batch_replay_forward_advance_at_10ms_enqueue_ms"] = (
                        replay(node)[1] - pipelined_forward_ns(node, 10 * MS)) / MS
                arm_batches.append(entry)
            first_pending_by_instance = {}
            if letter == "b":
                for instance, dispatch_ns in sorted(dispatch_origins.items()):
                    enqueues = [row["enqueue_end_ns"] for row in pair["measured_overlap"]["copies"]
                                if row["instance"] == instance and row["next_request"] is not None]
                    if enqueues:
                        first_pending_by_instance[instance] = (min(enqueues) - dispatch_ns) / MS
            record["arms"][mode] = {"TTFT_tail_request": values["max_ttft_request"],
                                   "drain_tail_request": values["last_request"], "requests": requests,
                                   "batches": arm_batches,
                                   "first_pending_copy_submission_ms_from_instance_first_dispatch": first_pending_by_instance}
        output.append(record)
    return {"run_id": summary["run_id"], "matched_P_batches_in_source": len(joined), "paired_rounds": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefill", "decode", "summary", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(read_bundle(args.prefill), read_bundle(args.decode), read_json(args.summary)))


if __name__ == "__main__":
    main()
