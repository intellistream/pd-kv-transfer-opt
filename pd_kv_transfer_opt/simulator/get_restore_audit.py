"""Screen all r69 restore batches and replay same-batch shared Attention reads."""

from __future__ import annotations

import argparse
from collections import defaultdict
from gzip import open as gzip_open
from itertools import pairwise
import json
from pathlib import Path
from statistics import median_low

from pd_kv_transfer_opt.common import read_jsonl, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import ALIGN_STATE_OBJECT_BYTES, ATTENTION_OBJECT_BYTES
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import interval_union
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS
from pd_kv_transfer_opt.simulator.freeinference_ab import calibration
from pd_kv_transfer_opt.simulator.get_criticality import (
    METHODS, against_baseline, clip_seconds, load_method_evidence, p_reply_advances, scenario,
)
from pd_kv_transfer_opt.simulator.get_overlap_screen import batches, free_spans, port_evidence
from pd_kv_transfer_opt.simulator.rdma_replay import drained_measurement, load_native_archive
from pd_kv_transfer_opt.simulator.terminal_pressure import planned_recovery_start


GET = "batch_get_into_multi_buffers"
P_ROLES = ("P0", "P1")
CLOCK_GUARD_NS = 42_000_000


def identity(row):
    return row["instance"], row["pid"], row["monotonic_ns"], row["type"]


def read_raw_p(path):
    with gzip_open(path, "rt", encoding="utf-8") as stream:
        rows = {}
        for row in map(json.loads, stream):
            if row["type"] in {"p_chunk_dispatch", "p_chunk_device", "store_begin", "store_end"}:
                key = identity(row)
                if key in rows:
                    raise ValueError("Non-unique raw P event")
                rows[key] = row
    devices = {(row["instance"], row["batch"]): row for row in rows.values()
               if row["type"] == "p_chunk_device"}
    api_starts = {(row["instance"], row["pid"], row["call_id"]): row for row in rows.values()
                  if row["type"] == "store_begin" and row["operation"] == GET}
    return rows, devices, api_starts


def checked_raw_fields(p_events, raw):
    rows, devices, api_starts = raw
    forward, calls = {}, {}
    for event in p_events:
        if event["type"] == "p_chunk_dispatch":
            row = rows[identity(event)]
            if (row["dispatch_ns"], row["total"]) != (event["dispatch_ns"], event["total"]):
                raise ValueError("Raw and exported scheduler events differ")
            forward[event["instance"], event["batch"]] = devices.get((row["instance"], row["batch"]))
        elif event["type"] == "store_end" and event["operation"] == GET:
            row = rows[identity(event)]
            start = api_starts[row["instance"], row["pid"], row["call_id"]]
            if any(row[key] != event[key] for key in ("elapsed_ns", "payload_bytes", "negative_returns")) or (
                    start["payload_bytes"], start["keys"]) != (row["payload_bytes"], row["keys"]):
                raise ValueError("Raw and exported Store calls differ")
            calls[row["instance"], row["pid"], row["monotonic_ns"]] = row["keys"]
    return forward, calls


def source_prefixes(path, workload):
    digest = sha256(path)
    if digest != workload["source_workload_sha256"]:
        raise ValueError("Terminal-Bench source ledger did not match r69")
    families, tips = defaultdict(list), {}
    for row in read_jsonl(path):
        families[row["task"]].append(row)
    for family, rows in families.items():
        rows.sort(key=lambda r: r["step_id"])
        tips.update({row["id"]: (family, index) for index, row in enumerate(rows)})
    for prompt in workload["request_sources"]:
        family, index = tips[prompt["id"]]
        row = families[family][index]
        if (row["prompt_tokens"], row["token_ids_uint32le_sha256"]) != (prompt["prompt_tokens"], prompt["prompt_sha256"]):
            raise ValueError("Source prefix is for a different token sequence")
    return digest, families, tips


def shared_prefix_blocks(a, b, families, tips):
    family, i = tips[a]
    other_family, j = tips[b]
    if family != other_family:
        return 0
    left, right = sorted((i, j))
    common = min(current["source_prompt_lcp_to_previous"] if current["predecessor"] == previous["id"] else 0
                 for previous, current in pairwise(families[family][left:right + 1]))
    return common // ATTENTION_BLOCK_TOKENS


def shareable_blocks(key, earlier, observed, families, tips):
    begin, end = observed[key]
    reused = set()
    for donor in earlier:
        left, right = observed[donor]
        common = shared_prefix_blocks(key, donor, families, tips)
        reused.update(range(max(begin, left), min(end, right, common)))
    return len(reused)


def checked_read_plan(chunk, ranks, lookup, orders, calls, families, tips):
    role = chunk["id"].split("/")[0]
    ids = orders[role, ranks[0]["worker"], ranks[0]["start"]]
    if any(orders[role, r["worker"], r["start"]] != ids or len(r["reads"]) != len(ids) for r in ranks):
        raise ValueError("One synchronous GET per request must appear in the same order on all four ranks")
    observed = {}
    for key in ids:
        local_tokens, new_tokens = lookup[key]
        if local_tokens % ATTENTION_BLOCK_TOKENS or new_tokens % ATTENTION_BLOCK_TOKENS or new_tokens <= 0:
            raise ValueError("Unexpected r69 prefix block alignment")
        observed[key] = (local_tokens // ATTENTION_BLOCK_TOKENS, (local_tokens + new_tokens) // ATTENTION_BLOCK_TOKENS)
    plan = []
    for index, key in enumerate(ids):
        begin, end = observed[key]
        reused = shareable_blocks(key, ids[:index], observed, families, tips)
        matched_blocks = end - begin
        payload = matched_blocks * ATTENTION_OBJECT_BYTES + 3 * ALIGN_STATE_OBJECT_BYTES
        for rank in ranks:
            _, stop, actual_payload = rank["reads"][index]
            if actual_payload != payload or calls[role, rank["worker"], stop] != matched_blocks + 3:
                raise ValueError(f"Read shape did not match the lookup and ordinal in {chunk['id']}")
        plan.append({"request": key, "attention_span_in_blocks": [begin, end],
                     "native_bytes_all_ranks": payload * len(ranks),
                     "same_rank_shareable_attention_bytes": reused * ATTENTION_OBJECT_BYTES * len(ranks)})
    return plan


def window_evidence(chunk, previous, compute, samples, busy, p_to_d_ns, origin_ns, cap, ranks, plan):
    start = min(a for r in ranks for a, _, _ in r["reads"])
    stop = max(b for r in ranks for _, b, _ in r["reads"])

    def relative(t):
        return (t + p_to_d_ns - origin_ns) / NS

    begin = max(chunk["dispatch"], previous["start"])
    end = min(start, previous["ready"])
    gaps = list(free_spans(begin, end, busy)) if begin < end else []
    left, right = max(gaps, key=lambda t: t[1] - t[0], default=(begin, begin))
    width = right - left
    nominal = cap / 8 * width
    ports = port_evidence(samples, (left, right), p_to_d_ns, cap)
    guarded = (port_evidence(samples, (left + CLOCK_GUARD_NS, right - CLOCK_GUARD_NS), p_to_d_ns, cap)
               if width > 2 * CLOCK_GUARD_NS else [])
    forward = None
    if compute:
        host_start, host_stop = compute["forward_enter_ns"], compute["forward_cpu_return_ns"]
        forward = {"host_forward_window_s": [relative(host_start), relative(host_stop)],
                   "device_forward_duration_ms": compute["forward_device_ms"],
                   "candidate_overlap_with_host_forward_s": (
                       sum(max(0, min(b, host_stop) - max(a, host_start)) for a, b in gaps) / NS)}
    return {
        "original_GET_envelope_s": [relative(start), relative(stop)],
        "longest_cross_batch_gap_s": width / NS,
        "gap_s_since_first_offer_approx": [relative(left), relative(right)] if width else None,
        "all_cross_batch_GET_free_spans_s_approx": [[relative(a), relative(b)] for a, b in gaps],
        "nominal_empty_link_budget_bytes": nominal,
        "calls_that_fit_nominally": [row["request"] for row in plan if row["native_bytes_all_ranks"] <= nominal],
        "raw_port_polls": ports, "residual_by_port_bytes_with_clock_guard": guarded,
        "previous_rank0_forward": forward,
    }


def exposed_worker_cuts(chunk, ranks, plan, local_copy_GBps):
    last = max(r["end"] for r in ranks)
    copy_bytes = sum(row["same_rank_shareable_attention_bytes"] for row in plan) / len(ranks)
    copy_ns = copy_bytes / local_copy_GBps
    without_setup, duplicate_read = [], []
    for rank in ranks:
        frontier = max(chunk["start"], rank["start"])
        exposed = [max(0, b - max(a, frontier)) for a, b, _ in rank["reads"]]
        setup = max(0, rank["end"] - frontier) - sum(exposed)
        if setup < 0:
            raise ValueError("Rank calls overlap inside the synchronous worker")
        delta = sum(t * request["same_rank_shareable_attention_bytes"] / request["native_bytes_all_ranks"]
                    for t, request in zip(exposed, plan, strict=True))
        without_setup.append(rank["end"] - setup)
        duplicate_read.append(rank["end"] - delta)
    removed = last - max(duplicate_read)
    return {"no_worker_time_outside_GET": (last - max(without_setup)) / NS,
            "share_prefix_no_copy": removed / NS,
            "share_prefix_with_copy": (removed - copy_ns) / NS,
            "local_copy_break_even_GBps_per_rank": copy_bytes / removed if removed > 0 else None}


def analyze_arm(arm, events, payload_bytes, raw, families, tips, fits, cap, local_copy_GBps):
    streams, requests, origin = arm
    p_events, chunks, loads, joined = batches(events, requests, payload_bytes)
    forward, calls = checked_raw_fields(p_events, raw)
    chunk_map = {c["id"]: c for sequence in chunks.values() for c in sequence}
    orders = {(r["instance"], r["pid"], r["monotonic_ns"]): r["request_ids"]
              for r in p_events if r["type"] == "worker_load_begin"}
    hits = defaultdict(list)
    for row in sorted(p_events, key=lambda r: r["monotonic_ns"]):
        if row["type"] == "pool_match_tokens":
            hits[row["instance"], row["request_id"]].append(row)
    clock, = (r["anchor"] for r in streams["client"] if r["type"] == "client_origin")
    p_to_d = median_low(r["unix_ns"] - r["monotonic_ns"] for r in p_events) - clock["unix_ns"] + clock["monotonic_ns"]
    origin_ns = round(origin * NS)
    busy = {p: sorted((a, b) for (role, _), ranks in loads.items() if role == p
                      for rank in ranks for a, b, _ in rank["reads"]) for p in P_ROLES}
    all_busy = sorted(busy["P0"] + busy["P1"])
    since = planned_recovery_start(requests)
    baseline = scenario(requests, {}, fits, since)
    measured = drained_measurement(requests)
    if (not measured["all_natural_EOS"]
            or abs(baseline["fixed_observed_downstream"]["duration_s"] - measured["duration_s"]) > 1e-6):
        raise ValueError("Zero-change replay did not reproduce the natural-EOS baseline")

    def replay(cuts, hold_dispatch):
        replies = p_reply_advances(chunks, cuts, propagate=True, hold_dispatch=hold_dispatch)
        return against_baseline(scenario(requests, replies, fits, since), baseline)

    cases = {}
    for role, sequence in chunks.items():
        other_p = "P1" if role == "P0" else "P0"
        for previous, chunk in pairwise(sequence):
            key = chunk["id"]
            if key not in joined:
                continue
            group = joined[key]
            ranks = loads[role, tuple(group["request_ids_in_GET"])]
            confirmed = []
            for candidate in group["scheduler_matches"]:
                trace = forward[role, chunk_map[candidate]["batch"]]
                if trace:
                    worker, = [r for r in ranks if r["worker"] == trace["pid"]]
                    if trace["worker_enter_ns"] <= worker["start"] < worker["end"] <= trace["forward_enter_ns"]:
                        confirmed.append(candidate)
            if confirmed != [key] and (confirmed or not group["unique_dispatch_join"]):
                raise ValueError("Raw rank-0 data did not resolve the GET join")
            lookup = {}
            for request in group["request_ids_in_GET"]:
                eligible = [r for r in hits[role, request] if r["monotonic_ns"] <= chunk["dispatch"]]
                row = eligible[-1]
                lookup[request] = row["local_tokens"], row["external_tokens"]
            plan = checked_read_plan(chunk, ranks, lookup, orders, calls, families, tips)
            window = window_evidence(chunk, previous, forward[role, previous["batch"]], streams["telemetry"], all_busy,
                                     p_to_d, origin_ns, cap, ranks, plan)
            first = min(a for r in ranks for a, _, _ in r["reads"])
            last = max(b for r in ranks for _, b, _ in r["reads"])
            concurrent = interval_union((max(a, first), min(b, last)) for a, b in busy[other_p] if a < last and b > first)
            slowest = max(ranks, key=lambda r: r["end"])
            api_ns = sum(b - a for a, b, _ in slowest["reads"])
            cases[key] = {
                "P": role, "joined_by_rank0": confirmed == [key], "previous_chunk": previous["id"],
                "payload_bytes": sum(row["native_bytes_all_ranks"] for row in plan),
                "read_plan": plan, "same_rank_shareable_attention_bytes": sum(
                    row["same_rank_shareable_attention_bytes"] for row in plan),
                "GET_envelope_overlap_with_other_P_GET_s": concurrent / NS,
                "GET_envelope_s": (last - first) / NS,
                "last_rank_GET_s": api_ns / NS,
                "last_rank_worker_load_outside_GET_s": (slowest["end"] - slowest["start"] - api_ns) / NS,
                "charged_cuts_s": exposed_worker_cuts(chunk, ranks, plan, local_copy_GBps),
                "window": window, "remove_GET_at_fixed_dispatch": replay({key: clip_seconds(group, 1)}, True),
                "remove_GET_following_worker_progress": replay({key: clip_seconds(group, 1)}, False),
            }
    if set(cases) != set(joined) or sum(c["payload_bytes"] for c in cases.values()) != payload_bytes:
        raise ValueError("Lost a restore batch or descriptor bytes")
    changes = {}
    for mode in ("no_worker_time_outside_GET", "share_prefix_no_copy", "share_prefix_with_copy"):
        cuts = {k: row["charged_cuts_s"][mode] for k, row in cases.items() if row["charged_cuts_s"][mode]}
        changes[mode] = {"cuts_s": cuts, "fixed_dispatch": replay(cuts, True),
                         "follow_worker_progress": replay(cuts, False)}
    duplicate_bytes = sum(c["same_rank_shareable_attention_bytes"] for c in cases.values())
    return {
        "measured": measured, "baseline_replay": baseline,
        "P_GET_descriptor_bytes": payload_bytes, "batches": cases, "service_sensitivities": changes,
        "token_confirmed_overlap_bytes_in_same_rank_attention_spans": duplicate_bytes,
        "descriptor_bytes_if_all_candidate_keys_are_shared": payload_bytes - duplicate_bytes,
        "five_latest_original_completions": sorted(
            ({"request": k, "P": r["P"], "D": r["D"], "end_s": r["end_s"]} for k, r in requests.items()),
            key=lambda r: -r["end_s"])[:5],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--raw-p", type=Path, required=True)
    parser.add_argument("--terminal-ledger", type=Path, required=True)
    parser.add_argument("--local-copy-GBps", type=float, default=10.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.local_copy_GBps < float("inf"):
        parser.error("Use a finite positive per-rank copy rate")
    directory = args.results / METHODS
    manifest, request_digest, arms = load_native_archive(directory)
    if manifest["settings"]["P_pool_GiB"] != 0:
        raise ValueError("Expected candidate reads to target D Store segments")
    method_events, counts, method_digest = load_method_evidence(directory, manifest)
    ledger_digest, families, tips = source_prefixes(args.terminal_ledger, manifest["workload"])
    raw = read_raw_p(args.raw_p)
    fits = {n: meta["decode_fit"] for n, (_, meta) in calibration(args.results).items()}
    result = {case: analyze_arm(arm, method_events[case], counts[case]["P_GET_descriptor_bytes"], raw, families, tips,
                                fits, int(case.rsplit("-cap", 1)[1]), args.local_copy_GBps)
              for case, arm in arms.items()}
    write_json(args.output, {
        "archive": METHODS, "request_trace_sha256": request_digest, "method_trace_sha256": method_digest,
        "raw_P_trace_sha256": sha256(args.raw_p), "terminal_ledger_sha256": ledger_digest,
        "alternative_local_copy_GBps_per_rank": args.local_copy_GBps,
        "port_overlap_scope": "Both P instances' recorded GET intervals are excluded. The amount of instantaneous "
                              "per-port headroom, actual flow-to-port mapping, immutable source keys and destination "
                              "page fences are not observed. Device duration is available on rank 0 with host "
                              "submission times, without a device time origin.",
        "shared_prefix_scope": "Intersect measured missing Attention spans using a conservative token-LCP bound. "
                               "The read shape is checked per call, including all three private state objects per rank. "
                               "Store key identities and destination page layout are unobserved. This does not merge "
                               "ranks or alter batch membership. Copy cost is an illustrative per-rank byte cost; service "
                               "saved is assumed linear in the removed share of each API and does not reschedule the network.",
        "cases": result,
    })


if __name__ == "__main__":
    main()
