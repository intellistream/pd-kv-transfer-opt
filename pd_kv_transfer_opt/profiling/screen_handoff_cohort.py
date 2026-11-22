"""Build a Terminal-Bench cohort from witnessed P-local, D-miss handoffs."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from gzip import open as gzip_open
from heapq import heappop, heappush
from itertools import count, pairwise
import json
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_multiwave import aligned_tx_intervals
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, read_trace, requests_for
from pd_kv_transfer_opt.profiling.screen_bidirectional_workload import D_RATE, P_RATE, peak_demand


COUNTS = (1, 5, 10, 30)


def ancestors(specs, key):
    while (key := specs[key]["predecessor"]) is not None:
        yield key


def profile(arm, summary, specs, ledgers, rank_map):
    origin, = [row["monotonic_s"] for row in arm["client"] if row["type"] == "client_origin"]
    profiles = requests_for(arm, list(specs.values()), origin)
    p_lookup, d_alloc = {}, {}
    submits = defaultdict(dict)
    native_ends = defaultdict(dict)
    write_bytes = Counter()
    p_before = sorted((x for x in arm["p_native"] if x["type"] == "pool_match_tokens"),
                      key=lambda x: x["monotonic_ns"])
    for row in p_before:
        p_lookup.setdefault(row["request_id"], row)
    for row in arm["p_native"]:
        if row["type"] == "adxl_transfer_sync" and row["operation"] == 1 and row["status"] == 0:
            write_bytes[row["instance"]] += row["payload_bytes"]
    for row in arm["d_native"]:
        kind = row["type"]
        if kind == "d_paged_receive_allocated":
            key = row["request_id"]
            if key in d_alloc:
                raise ValueError(f"Multiple D allocations for {key}")
            d_alloc[key] = row
        elif kind in ("native_transfer_submit", "native_handle_end"):
            key = row["remote_request_id"]
            rank = rank_map[row["instance"]][str(row["pid"])]
            target = submits if kind == "native_transfer_submit" else native_ends
            if rank in target[key] or row["instance"] != profiles[key]["D"]:
                raise ValueError(f"Unexpected native D rank or route for {key}")
            target[key][rank] = row
    if not profiles.keys() == p_lookup.keys() == d_alloc.keys() == submits.keys() == native_ends.keys():
        raise ValueError("Method observations did not cover all requests")
    result = {}
    for key, client in profiles.items():
        p, d, detail = p_lookup[key], d_alloc[key], summary["per_request"][key]
        if (p["instance"], d["instance"]) != (client["P"], client["D"]):
            raise ValueError(f"Unexpected allocation route for {key}")
        rank_submits, rank_ends = submits[key], native_ends[key]
        if rank_submits.keys() != rank_ends.keys() or rank_submits.keys() != set(range(4)):
            raise ValueError(f"Incomplete four-rank handoff for {key}")
        moves = []
        for rank, event in rank_submits.items():
            released = event["monotonic_ns"] / NS - origin
            handled = rank_ends[rank]["monotonic_ns"] / NS - origin
            # The pinned equal-TP connector selects the same rank on the named source.
            source_rail = 4 * int(client["P"][-1]) + rank
            if not client["D_start_s"] <= released <= handled <= detail["D_KV_ready_s"]:
                raise ValueError(f"Invalid handoff chronology for {key}")
            moves.append((released, source_rail, event["payload_bytes"]))
        local = p["local_tokens"]
        if ledgers[key]["length"] != detail["prompt_tokens"]:
            raise ValueError(f"Wrong token ledger for {key}")
        needed_blocks = (local + 1023) // 1024
        donor = None
        if needed_blocks and needed_blocks <= len(ledgers[key]["blocks"]):
            prefix = ledgers[key]["blocks"][:needed_blocks]
            donor = next((prev for prev in ancestors(specs, key)
                          if profiles[prev]["P"] == client["P"] and ledgers[prev]["blocks"][:needed_blocks] == prefix), None)
        result[key] = {**client, "P_local": local, "P_first_computed_prefix": detail["P_first_computed_prefix"],
                       "D_local": d["local_tokens"], "D_external": d["external_tokens"],
                       "P_computed": detail["P_computed_prompt_tokens"], "P_GET_bytes": detail["P_direct_GET_bytes"],
                       "input": detail["prompt_tokens"], "donor": donor, "moves": moves,
                       "handoff_bytes": sum(move[2] for move in moves),
                       "D_alloc_s": d["monotonic_ns"] / NS - origin,
                       "D_KV_ready_s": detail["D_KV_ready_s"],
                       "last_handle_s": max(x["monotonic_ns"] / NS - origin for x in rank_ends.values())}
    handoff = sum(row["handoff_bytes"] for row in result.values())
    if handoff != summary["physical_paths"]["P_to_D_Mooncake_descriptor_bytes"]:
        raise ValueError("Handoff bytes differ from the independent trace receipt")
    per_compute = {p: write_bytes[p] / sum(row["P_computed"] for row in result.values() if row["P"] == p)
                   for p in ("P0", "P1")}
    return result, origin, per_compute


def is_witness(row):
    return (row["P_first_computed_prefix"] >= row["P_local"] >= 32 * 1024
            and row["P_computed"] <= 0.3 * row["input"] and row["donor"] is not None
            and row["P_GET_bytes"] == 0 and row["D_external"] >= 8192 and row["handoff_bytes"] >= 10**9)


def offer_peaks(offers):
    offers = sorted(offers)
    return {"aggregate": {str(w): peak_demand(offers, w) for w in COUNTS},
            "each_logical_source_rail": [{str(w): peak_demand([x for x in offers if x[1] == p], w) for w in COUNTS}
                                         for p in range(8)]}


def receiver_pressure(telemetry, origin, stop, counter="mac_rx_total_oct_num"):
    # Reuse the irregular-sample alignment. Measured RX also includes Store traffic and wire overhead.
    samples = [{"ports": [{**entry, "counters": {"mac_tx_total_oct_num": entry["counters"][counter]}}
                          for entry in sample["ports"]]} for sample in telemetry]
    intervals = [(max(0, a), min(stop, b), rates) for a, b, _, rates in aligned_tx_intervals(samples, round(origin * NS))
                 if max(0, a) < min(stop, b)]
    timestamps = [row[0] for row in intervals] + [intervals[-1][1]]
    peaks = []
    for port in range(8):
        rates = [row[2][port] for row in intervals]
        data = [0.0]
        for (a, b), rate in zip(pairwise(timestamps), rates, strict=True):
            data.append(data[-1] + (b - a) * rate)

        def integral(t):
            point = min(len(rates) - 1, bisect_right(timestamps, t) - 1)
            return data[point] + (t - timestamps[point]) * rates[point]

        def rolling_max(seconds):
            begin, end = timestamps[0], timestamps[-1] - seconds
            candidates = {begin, end, *(x for x in timestamps if begin < x < end),
                          *(x - seconds for x in timestamps if begin < x - seconds < end)}
            return max((integral(x + seconds) - integral(x)) / seconds for x in candidates)

        peaks.append({str(width): rolling_max(width) for width in COUNTS})
    return peaks


def overlap(spans, other):
    return interval_union((max(a, c), min(b, d)) for a, b in spans for c, d in other if max(a, c) < min(b, d))


def decode_gaps(times, actual, witnesses, wire):
    result = {}
    for role in ("D0", "D1"):
        keys = [k for k in times if actual[k]["D"] == role]
        if not keys:
            continue
        available = sorted((t, delta) for k in keys for t, delta in ((times[k]["ready"], 1), (times[k]["end"], -1)))
        pending = [(times[k]["alloc"], times[k]["ready"]) for k in keys if k in witnesses]
        busy = 0
        no_work = []
        for (left, delta), (right, _) in pairwise(available):
            busy += delta
            if busy == 0 and left < right:
                no_work.append((left, right))
        candidate_gaps = [(max(a, c), min(b, d)) for a, b in no_work for c, d in pending if max(a, c) < min(b, d)]
        result[role] = {"no_KV_ready_request_with_witness_pending_s": interval_union(candidate_gaps),
                        "also_overlaps_any_modeled_P_rail_busy_s": overlap(candidate_gaps, wire)}
    return result


def simulate(actual, selected, witnesses, offered, background_per_compute, cap, include_background):
    serial = count()
    queue = []
    send_until = [0.0] * 8
    service = [[] for _ in send_until]
    entries, times, submits = {}, {}, []
    child = {row["predecessor"]: row["id"] for row in selected if row["predecessor"] is not None}

    def insert(at, event, data):
        heappush(queue, (at, next(serial), event, data))

    def release(at, key):
        row = actual[key]
        entries[key] = [at, 4, 0.0]
        for when, rail, size in row["moves"]:
            insert(at + when - row["ready_s"], "send", (key, rail, size))
        if include_background:
            bkg_bytes = round(row["P_computed"] * background_per_compute[row["P"]] / 16)
            for f in (.25, .50, .75, 1.0):
                ready = row["P_start_s"] + (row["P_end_s"] - row["P_start_s"]) * f
                for rank in range(4):
                    port = 4 * int(row["P"][-1]) + rank
                    insert(at + ready - row["ready_s"], "send", (None, port, bkg_bytes))

    for spec in selected:
        if spec["predecessor"] is None:
            release(offered[spec["id"]], spec["id"])
    while queue:
        at, _, _, (key, rail, payload) = heappop(queue)
        if payload:
            start = max(at, send_until[rail])
            finish = start + 64 * payload / (cap * NS)
            service[rail].append((start, finish))
            send_until[rail] = finish
            submits.append((at, rail, payload, key or ""))
        else:
            finish = at
        if key is None:
            continue
        item = entries[key]
        item[1] -= 1
        item[2] = max(item[2], finish)
        if item[1]:
            continue
        row = actual[key]
        shifted = item[0] - row["ready_s"]
        delay = max(0.0, item[2] - (row["last_handle_s"] + shifted))
        times[key] = {"alloc": row["D_alloc_s"] + shifted, "ready": row["D_KV_ready_s"] + shifted + delay,
                      "end": row["end_s"] + shifted + delay, "added_s": delay}
        if key in child:
            nxt = child[key]
            release(max(offered[nxt], times[key]["end"] + actual[nxt]["tool_gap_s"]), nxt)
    if len(times) != len(selected):
        raise ValueError("Incomplete replay of the candidate dependency chains")
    tail = max(times, key=lambda key: times[key]["end"])
    end = times[tail]["end"]
    output = sum(actual[k]["output_tokens"] for k in times)
    busy_rails = [entry for rail in service for entry in rail]
    return {"duration_s": end - min(offered.values()), "output_tokens": output,
            "output_tokens_s": output / (end - min(offered.values())), "tail": tail,
            "released_successors_through_30s": sum(entry[0] < 30 and actual[key]["predecessor"] is not None
                                                    for key, entry in entries.items()),
            "D_KV_ready_through_30s": sum(row["ready"] < 30 for row in times.values()),
            "last_KV_ready_s": max(row["ready"] for row in times.values()),
            "D_supply": decode_gaps(times, actual, witnesses, busy_rails),
            "max_added_handoff_s": max(row["added_s"] for row in times.values()),
            "offered_peaks": offer_peaks(submits)}


def analyze_case(arm, actual, origin, bkg, selected, offered, anchors):
    keys = {row["id"] for row in selected}
    witnesses = {key for key in anchors if is_witness(actual[key])}
    def flows(ids):
        return [(at, port, size, key) for key in ids for at, port, size in actual[key]["moves"]]

    moves = flows(keys)
    observed_writes = flows(witnesses)
    volumes = [sum(size for _, rail, size, _ in moves if rail == p) for p in range(8)]
    witness_volumes = [sum(size for _, rail, size, _ in observed_writes if rail == p) for p in range(8)]
    compute = sum(actual[key]["P_computed"] for key in keys)
    output = sum(actual[key]["output_tokens"] for key in keys)
    baseline_duration = max(row["end_s"] for row in actual.values())
    historical_times = {key: {"alloc": row["D_alloc_s"], "ready": row["D_KV_ready_s"], "end": row["end_s"]}
                        for key, row in actual.items()}
    return {"observed_full_trace_output_tokens_s": sum(row["output_tokens"] for row in actual.values()) / baseline_duration,
            "witnesses": len(witnesses), "witness_bytes": sum(witness_volumes),
            "selected_overall_handoff_bytes": sum(volumes), "selected_P_computed": compute, "selected_output_tokens": output,
            "selected_upper_handoff_sustained_Gbps": sum(volumes) * 8 / NS / max(compute / P_RATE, output / D_RATE),
            "witness_hottest_rail_serialization_s_at_12_5": max(witness_volumes) * 8 / (12.5 * NS),
            "selected_hottest_rail_serialization_s_at_12_5": max(volumes) * 8 / (12.5 * NS),
            "witness_bytes_by_logical_P_rail": witness_volumes, "selected_bytes_by_logical_P_rail": volumes,
            "historical_witness_peaks": offer_peaks(observed_writes),
            "historical_selection_peaks": offer_peaks(moves),
            "historical_D_MAC_RX_peak_Gbps_by_port": receiver_pressure(arm["telemetry"], origin, baseline_duration),
            "historical_D_supply": decode_gaps(historical_times, actual, witnesses, []),
            "witness_receipts": {k: {field: actual[k][field] for field in (
                "P", "D", "P_local", "P_first_computed_prefix", "P_computed", "D_local", "D_external", "D_KV_ready_s", "handoff_bytes",
                "P_GET_bytes", "donor", "output_tokens", "moves")} for k in sorted(witnesses)},
            "replays": {label: {str(cap): simulate(actual, selected, witnesses, offered, bkg, cap, with_background)
                                for cap in (100, 200)} for label, with_background in (("handoff", False), ("plus_estimated_P_PUT", True))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method_archive", type=Path)
    parser.add_argument("prefix_ledger", type=Path)
    parser.add_argument("rank_map", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.method_archive / "manifest.json")
    trace_file = args.method_archive / "r43-method-events.jsonl.gz"
    if sha256(trace_file) != manifest["trace_sha256"]:
        raise ValueError("Trace fingerprint changed")
    data = read_trace(trace_file)
    method = read_json(args.method_archive / "summary.json")["cases"]
    root = manifest["workload"]
    specs = {row["id"]: row for row in root["request_sources"]}
    with gzip_open(args.prefix_ledger, "rt", encoding="utf-8") as stream:
        ledgers = json.load(stream)
    ranks = read_json(args.rank_map)
    profiles = {case: profile(arm, method[case], specs, ledgers, ranks) for case, arm in data.items()}
    seen = Counter(key for actual, _, _ in profiles.values() for key in specs if is_witness(actual[key]))
    keys = seen.keys() | {parent for key in seen for parent in ancestors(specs, key)}
    selected = [row for key, row in specs.items() if key in keys]
    root_offer = {row["session_id"]: order / 8 for order, row in enumerate(sorted(
        (s for s in selected if s["predecessor"] is None), key=lambda item: item["offered_s"]))}
    offered = {row["id"]: root_offer[row["session_id"]] for row in selected}
    candidate = {key: root[key] for key in ("source_repo", "source_revision", "source_workload_sha256", "tokenizer_fingerprint",
                                            "ignore_eos", "temperature", "tool_gap_policy", "max_tokens_policy", "route_policy")}
    candidate.update(parent_r42_sha256=root["workload_sha256"], requests=len(selected), sessions=len(root_offer),
                     last_session_start_s=max(root_offer.values()), pressure_window_s=[30, 90],
                     arrival_policy="8 roots/s in original relative r42 order, plus original predecessor dependencies",
                     request_sources=[{**row, "original_offered_s": row["offered_s"], "offered_s": offered[row["id"]]}
                                      for row in selected])
    results = {case: analyze_case(data[case], rows, origin, bkg, selected, offered, seen)
               for case, (rows, origin, bkg) in profiles.items()}
    write_json(args.output / "workload.json", candidate)
    write_json(args.output / "summary.json", {
        "source_trace_sha256": manifest["trace_sha256"], "prefix_ledger_sha256": sha256(args.prefix_ledger),
        "rank_map_sha256": sha256(args.rank_map), "candidate_plan_sha256": sha256(args.output / "workload.json"),
        "target_witnessed_count": len(seen), "target_witnessed_in_both": [key for key, n in seen.items() if n == len(profiles)],
        "from_r28_targets": sum(key.startswith("natural-") for key in seen),
        "requests": len(selected), "sessions": len(root_offer), "native_profile_arms": results,
        "scope": "Observed cache and rank facts from r43. Per-rank source assumes one rail per equal-TP sender. "
                 "Physical historic counters are D RX. Replay freezes observed stage costs and routes, propagates causal releases "
                 "and exposes additional FIFO overrun on first token. Background sensitivity distributes each request's estimated "
                 "P STORE PUT bytes in four equal steps through its observed P stage."})


if __name__ == "__main__":
    main()
