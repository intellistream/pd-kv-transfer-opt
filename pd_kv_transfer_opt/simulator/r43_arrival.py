"""Compare r43's root arrivals with conditional Poisson arrivals on CPU."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
from math import isclose
from pathlib import Path
from random import Random
from statistics import linear_regression, mean, median

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import get_batches
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, read_trace, requests_for
from pd_kv_transfer_opt.simulator.agent_replay import GB, PAGE, Replay, Service
from pd_kv_transfer_opt.simulator.decode_supply import decode_samples, fit_decode, native_arms
from pd_kv_transfer_opt.simulator.freeinference_ab import report
from pd_kv_transfer_opt.simulator.get_criticality import clip_seconds, join_loads, prefill_chain, worker_loads
from pd_kv_transfer_opt.simulator.rdma_replay import egress_epochs
from pd_kv_transfer_opt.workloads.freeinference import Session, Turn


ARCHIVE = "2026-11-20-r43-native-decode-apc"
METHODS = "2026-11-20-r43-native-decode-apc-methods"
PREFIX = "2026-11-21-terminal-handoff-cohort/prefix-ledger.json.gz"
WINDOW = 92.5
SEEDS = (43, 143, 243)
RATES_PER_MINUTE = (4, 5, 6, 7, 8, 10, 12, 16, 18, 20, 24)
CAPS = (100, 200)


def load_inputs(results):
    archive = results / ARCHIVE
    manifest = read_json(archive / "manifest.json")
    settings, workload = manifest["settings"], manifest["workload"]
    engines = settings["engines"]
    if (settings["D_pool_GiB"] != 128 or settings["D_egress_ports"] != 8
            or not engines["D"]["recompute_scheduler_enable"] or workload["ignore_eos"]
            or workload["route_policy"] != "upstream native without route override"
            or not all(e["enable_prefix_caching"] and e["max_num_batched_tokens"] == 4096
                       and e["max_num_seqs"] == 16 for e in engines.values())):
        raise ValueError("Wrong r43 deployment or workload")
    digest, arms = native_arms(archive)
    for case, (raw, _, _) in arms.items():
        cap = int(case.rsplit("cap", 1)[1])
        shaping, = raw["shaping.json"]
        init, = raw["cache-initialization.json"]
        if (shaping["aggregate_Gbps"] != cap or set(shaping["readback"]) != set(map(str, range(8)))
                or any(row["configured_Mbps"] != cap * 125 or row["current_Mbps"] != cap * 125
                       for row in shaping["readback"].values())
                or init["after"]["master_key_count"] != 0 or any(row.get("fixed_routes") for row in raw["proxy"])):
            raise ValueError(f"Invalid original cap, route or starting cache in {case}")
    profile_dir = results / METHODS
    profile_manifest = read_json(profile_dir / "manifest.json")
    profile_path = profile_dir / "r43-method-events.jsonl.gz"
    summary_path = profile_dir / "summary.json"
    hashes = {path.name: sha256(path) for path in (profile_path, summary_path)}
    if (profile_manifest["workload"]["request_sources"] != workload["request_sources"]
            or any(profile_manifest["files"][name]["sha256"] != value for name, value in hashes.items())):
        raise ValueError("Method trace does not have the original workload or artifact digest")
    with gzip.open(results / PREFIX, "rt", encoding="utf-8") as stream:
        prefix = json.load(stream)
    if set(prefix) != set(arms["b1-cap200"][1]):
        raise ValueError("Prefix ledger is not the r43 request set")
    return manifest, arms, prefix, read_trace(profile_path), read_json(summary_path)["cases"], {
        "r43_workload_sha256": workload["workload_sha256"], "formal_trace_sha256": digest,
        "prefix_ledger_sha256": sha256(results / PREFIX), **hashes,
    }


def reference_eos(arms, spec, block):
    high = f"b{block}-cap200"
    order = (high, f"b{3 - block}-cap200", f"b{block}-cap100", f"b{3 - block}-cap100")
    outcomes = {}
    for key, row in spec.items():
        for case in order:
            observed = arms[case][1][key]
            if observed["finish_reason"] == "stop" and 0 < observed["output_tokens"] < row["max_tokens"]:
                outcomes[key] = {"tokens": observed["output_tokens"], "source": case, "finish": "stop"}
                break
        else:
            raise ValueError(f"No archived natural EOS below the r43 budget for {key}")
    return outcomes


def materialize(spec, prefix, outcomes):
    groups, page_ids = defaultdict(list), {}
    for row in spec.values():
        groups[row["session_id"]].append(row)
    sessions, offers, ids = [], {}, {}
    for name, rows in groups.items():
        turns, previous_pages, previous_id = [], (), None
        start = rows[0]["offered_s"]
        for step, row in enumerate(rows):
            key = row["id"]
            record, outcome = prefix[key], outcomes[key]
            if (row["predecessor"] != previous_id or record["predecessor"] != previous_id
                    or record["session"] != name or row["offered_s"] != start
                    or record["length"] != row["prompt_tokens"]
                    or len(record["blocks"]) != record["length"] // PAGE
                    or not 0 < outcome["tokens"] <= row["max_tokens"]):
                raise ValueError(f"Changed r43 input, predecessor, arrival floor or budget for {key}")
            pages = tuple(page_ids.setdefault(h, len(page_ids) + 1) for h in record["blocks"])
            shared = next((i for i, (a, b) in enumerate(zip(previous_pages, pages)) if a != b),
                          min(len(previous_pages), len(pages)))
            gap = round(1000 * row["tool_gap_s"])
            if gap != 1000 * row["tool_gap_s"]:
                raise ValueError(f"Tool interval loses resolution for {key}")
            turns.append(Turn(gap, record["length"], outcome["tokens"], pages, shared, outcome["finish"]))
            ids[f"{name}:{step}"] = key
            previous_pages, previous_id = pages, key
        sessions.append(Session(name, round(1000 * start), 0, 0, tuple(turns)))
        offers[name] = start
    if (len(sessions) != 48 or len(ids) != 254 or min(offers.values()) != 0 or max(offers.values()) != WINDOW
            or len(set(offers.values())) != 48):
        raise ValueError("Unexpected r43 start envelope")
    return sessions, offers, ids


def conditional_arrivals(original, seed):
    rng = Random(seed)
    names = sorted(original, key=original.get)
    times = [0., *sorted(rng.uniform(0, WINDOW) for _ in names[2:]), WINDOW]
    return dict(zip(names, times, strict=True))


def exponential_arrivals(original, seed, sessions_per_minute):
    rng = Random(seed)
    now, offers = 0., {}
    for name in sorted(original, key=original.get):
        offers[name] = now
        now += rng.expovariate(sessions_per_minute / 60)
    return offers


def fit_prefill(raw, observed):
    rows = observed["per_request"]
    events = raw["p_native"]
    _, batches, errors = get_batches(events)
    if errors:
        raise ValueError("Incomplete original prefill profiling")
    chain = prefill_chain(events, rows)
    by_chunk = join_loads(chain, worker_loads(events), observed["Store_GET"]["Store_payload_bytes"])
    excluded = {(role, item["dispatch"]) for role, chunks in chain.items() for item in chunks
                if item["id"] in by_chunk}
    counts, timings, previous_ready = [], [], defaultdict(int)
    for batch in sorted(batches, key=lambda b: b["ready_ns"]):
        role = batch["instance"]
        elapsed = (batch["ready_ns"] - max(batch["begin_ns"], previous_ready[role])) / NS
        previous_ready[role] = batch["ready_ns"]
        if (role, batch["dispatch_ns"]) not in excluded and elapsed > 0:
            counts.append(batch["total"])
            timings.append(elapsed)
    slope, overhead = linear_regression(counts, timings)
    if slope <= 0 or overhead < 0:
        raise ValueError("r43 needs a positive prefill affine fit")
    large = [r["payload_GB"] / clip_seconds(r, 1.) for r in by_chunk.values() if r["payload_GB"] >= .5]
    before = observed["Store_GET"]["Store_before_ADXL_s"]["p50"]
    p_put = observed["Store_PUT"]["Store_payload_bytes"]
    d_put = sum(row["payload_bytes"] for row in raw["d_native"]
                if row["type"] == "store_end" and row["operation"] == "batch_put_from_multi_buffers")
    return {"non_GET_batches": len(timings), "chunk_overhead_s": overhead, "tokens_per_s": 1 / slope,
            "median_abs_residual_s": median(abs(t - (overhead + slope * n))
                                            for t, n in zip(timings, counts, strict=True)),
            "large_GET_batches": len(large), "large_GET_effective_GB_s_p50": median(large),
            "GET_setup_s": before, "P_Store_PUT_payload_GB": p_put / GB, "D_Store_PUT_payload_GB": d_put / GB,
            "Store_resident_scale_proxy": 1 + d_put / p_put}


def fit_service(arms, profile, method, block, prefill):
    case = f"b{block}-cap200"
    _, rows, _ = arms[case]
    decode = fit_decode(list(decode_samples(*arms[case])))
    ports = [sum((b - a) * rates[i] / 8 for a, b, rates in egress_epochs(arms[case])) for i in range(8)]
    shares = tuple(value / sum(ports) for value in ports)
    # The fixed rail shares approximate aggregate imbalance. Positive exposed GET intervals
    # from both caps give a single efficiency to use for all schedules and caps in a block.
    efficiency = mean(prefill[cap]["large_GET_effective_GB_s_p50"] / (cap / (64 * max(shares))) for cap in CAPS)
    if not 0 < efficiency <= 1:
        raise ValueError("GET fit exceeds physical capacity")
    p_fit = prefill[CAPS[block - 1]]
    # Compact JSON request body size per token measured on this deployment in prior screens.
    bytes_per_token = 4.12420475216543
    ingress = median(row["P_start_s"] - row["ready_s"] for row in rows.values())
    obs = method["per_request"]
    pull_bytes = Counter()
    for row in profile["d_native"]:
        if row["type"] == "native_transfer_submit":
            pull_bytes[row["remote_request_id"]] += row["payload_bytes"]
    if pull_bytes.keys() != obs.keys():
        raise ValueError("P to D descriptor ledger is incomplete")
    pull_cost, pull_base = linear_regression([pull_bytes[key] / GB for key in obs],
                                            [row["D_alloc_to_KV_ready_s"] for row in obs.values()])
    if pull_cost <= 0 or pull_base < 0:
        raise ValueError("P to D Pull service fit is invalid")
    d_ingress = median(row["D_proxy_to_ingress_s"] for row in obs.values())
    d_start = max(0., median(row["D_KV_ready_to_first_s"] for row in obs.values())
                  - decode["first_sequence_period_s"])
    service = Service(shares, decode["first_sequence_period_s"], decode["additional_sequence_period_s"],
                      p_fit["tokens_per_s"], ingress, 0., bytes_per_token, 4,
                      get_setup_s=p_fit["GET_setup_s"], first_admission_s=0., rail_efficiency=efficiency,
                      warm_tokens_s=p_fit["tokens_per_s"], prefill_chunk_overhead_s=p_fit["chunk_overhead_s"],
                      decode_ingress_s=d_ingress, pull_setup_s=pull_base, pull_GB_s=1 / pull_cost, decode_start_s=d_start,
                      release_suffix_first=True, store_high_watermark=.95, store_eviction_ratio=.05,
                      store_resident_scale=p_fit["Store_resident_scale_proxy"], publish_decode_kv=True)
    return service, {"decode_case": case, "decode_fit": decode, "ports_GB": ports, "prefill_fit": p_fit,
                     "D_pull_profile_requests": len(pull_bytes), "parameters": asdict(service)}


def flow_window(run, offers, begin, end):
    """Integrate resource time and completion progress over the same wall-clock window."""
    if not begin < end <= run.now:
        raise ValueError("Invalid pressure measurement window")
    span = end - begin
    jobs = list(run.requests.values())
    first_D = max(min(job.row["D_first_s"] for job in jobs if job.row["D"] == role) for role in range(2))
    last_P = max(job.row["P_reply_s"] for job in jobs)
    P_busy, D_consume, D_active = [0., 0.], 0., 0.
    idle_with_P, idle_with_GET = 0., 0.
    for a, b, widths, pending, stages, gets, d_backlog in run.supply_epochs:
        seconds = min(b, end) - max(a, begin)
        if seconds <= 0:
            continue
        for role, (width, stage) in enumerate(zip(widths, stages, strict=True)):
            P_busy[role] += seconds * (stage != "idle")
            D_active += seconds * width
            if width:
                step = run.service.decode_first_s + run.service.decode_add_s * (width - 1)
                D_consume += seconds * width / step
        eligible = min(b, end, last_P) - max(a, begin, first_D)
        if eligible > 0 and any(pending):
            empty_instances = sum(not w and not q for w, q in zip(widths, d_backlog, strict=True))
            idle_with_P += eligible * empty_instances
            if gets:
                idle_with_GET += eligible * empty_instances
    qe = sum(max(0., min(j.row["P_dispatch_s"], end) - max(j.row["P_admit_s"], begin)) for j in jobs)
    session_finish = {j.session.name: j.row["EOS_s"] for j in jobs if j.step + 1 == len(j.session.turns)}
    flight = sum(max(0., min(stop, end) - max(offers[name], begin)) for name, stop in session_finish.items())
    wire_GB = sum(max(0., min(b, end) - max(a, begin)) * sum(rates) / 8
                  for a, b, rates in run.tx_epochs)
    get_s = sum(max(0., min(b, end) - max(a, begin)) for a, b, rates in run.tx_epochs if any(rates))
    return {
        "begin_s": begin, "end_s": end,
        "P_mean_waiting": qe / span, "P_busy_fraction_by_instance": [s / span for s in P_busy],
        "active_sessions_mean": flight / span,
        "session_starts": sum(begin < at <= end for at in offers.values()),
        "session_finishes": sum(begin < stop <= end for stop in session_finish.values()),
        "P_replies": sum(begin <= j.row["P_reply_s"] < end for j in jobs),
        "D_KV_ready": sum(begin <= j.row["D_KV_ready_s"] < end for j in jobs),
        "D_mean_active_sequences": D_active / span, "D_generation_tokens_s": D_consume / span,
        "D_empty_with_P_pending_instance_s": idle_with_P, "D_empty_with_P_pending_during_GET_instance_s": idle_with_GET,
        "D_to_P_GET_GB": wire_GB, "D_to_P_GET_mean_Gbps": wire_GB * 8 / span,
        "any_GET_wire_busy_s": get_s,
    }


def metrics(run, offers, *, detailed=False):
    result = report(run, detailed=detailed)
    jobs = list(run.requests.values())
    result["P_queue"] = {}
    for name, subset in (("all", jobs), ("roots", [j for j in jobs if j.step == 0]),
                         ("successors", [j for j in jobs if j.step > 0])):
        waits = [j.row["P_dispatch_s"] - j.row["P_admit_s"] for j in subset]
        result["P_queue"][name] = {"p50_s": median(waits), "p95_s": percentile(waits, .95), "sum_request_s": sum(waits)}
    result["D_to_Store_PUT_payload_GB"] = run.d_put_GB
    result["final_root_s"] = max(offers.values())
    result["last_P_reply_s"] = max(j.row["P_reply_s"] for j in jobs)
    result["drain_after_last_root_s"] = run.now - result["final_root_s"]
    result["during_root_window"] = flow_window(run, offers, 0., result["final_root_s"])
    result["all_time"] = flow_window(run, offers, 0., run.now)
    result["P_to_D_Pull_GB"] = result["cache"]["P_to_D_bytes"] / GB
    result["routes"] = {role: dict(Counter(j.row[role] for j in jobs)) for role in ("P", "D")}
    result["P_HBM_hit_tokens"] = result["cache"]["P_local_pages"] * PAGE
    result["D_HBM_hit_tokens"] = result["cache"]["D_local_pages"] * PAGE
    return result


def emit_requests(stream, run, ids, label, cap, block):
    for key, job in run.requests.items():
        record = {"plan": label, "cap_Gbps": cap, "block": block, "id": ids[key], "session": job.session.name,
                  "step": job.step + 1, "input_tokens": job.turn.prompt, "output_tokens": job.turn.output,
                  "finish_reason": job.turn.finish, "tool_gap_ms": job.turn.tool_gap_ms, **job.row}
        print(json.dumps(record, separators=(",", ":")), file=stream)


def run_curves(sessions, original, service, block):
    runs = []
    for rate in RATES_PER_MINUTE:
        for seed in SEEDS:
            offers = exponential_arrivals(original, seed, rate)
            times = list(offers.values())
            first, mid, last = times[11], times[29], times[-1]
            for cap in CAPS:
                run = Replay(sessions, offers, service, cap).run()
                a = flow_window(run, offers, first, mid)
                b = flow_window(run, offers, mid, last)
                runs.append({"rate_sessions_per_min": rate, "seed": seed, "cap_Gbps": cap, "block": block,
                             "output_tokens": sum(s.output for s in sessions), "finish_s": run.now,
                             "throughput_tokens_s": sum(s.output for s in sessions) / run.now,
                             "drain_s": run.now - last,
                             "P_queue_p95_s": percentile([j.row["P_dispatch_s"] - j.row["P_admit_s"]
                                                          for j in run.requests.values()], .95),
                             "D_to_P_GET_GB": sum(j.row["GET_bytes"] for j in run.requests.values()) / GB,
                             "P_to_D_Pull_GB": sum(j.row["P_to_D_bytes"] for j in run.requests.values()) / GB,
                             "early": a, "late": b, "P_queue_drift": b["P_mean_waiting"] - a["P_mean_waiting"],
                             "active_session_drift": b["active_sessions_mean"] - a["active_sessions_mean"]})
    return runs


def calibration_check(spec, prefix, service, cap, actual, profile=None):
    outputs = {key: {"tokens": row["output_tokens"], "finish": row["finish_reason"]} for key, row in actual.items()}
    session, offers, ids = materialize(spec, prefix, outputs)
    run = Replay(session, offers, service, cap).run()
    finish = max(row["end_s"] for row in actual.values())
    last_p = max(row["P_end_s"] for row in actual.values())
    output = sum(o["tokens"] for o in outputs.values())
    result = {"archived": {"finish_s": finish, "last_P_reply_s": last_p, "output_tokens": output,
                           "throughput_tokens_s": output / finish, "finish_reasons": dict(Counter(o["finish"] for o in outputs.values()))},
              "cpu": metrics(run, offers), "finish_error_pct": 100 * (run.now / finish - 1),
              "equal_original_routes_of_254": {role: sum(actual[ids[key]][role] == f"{role}{j.row[role]}"
                                                         for key, j in run.requests.items()) for role in ("P", "D")}}
    if profile:
        rows, paths = profile["per_request"], profile["physical_paths"]
        result["archived"].update(
            P_queue_p95_s=percentile([r["P_scheduler_queue_s"] for r in rows.values()], .95),
            P_computed_tokens=sum(r["P_computed_prompt_tokens"] for r in rows.values()),
            D_to_P_GET_GB=paths["D_to_P_Store_GET_descriptor_bytes"] / GB,
            P_to_D_Pull_GB=paths["P_to_D_Mooncake_descriptor_bytes"] / GB,
            D_HBM_hit_tokens=paths["D_observed_local_prefix_tokens"],
        )
    return result


def execute(results, dest):
    manifest, native, prefix, raw, method, digests = load_inputs(results)
    workload = manifest["workload"]
    spec = {r["id"]: r for r in workload["request_sources"]}
    prefill = {cap: fit_prefill(raw[f"b1-cap{cap}"], method[f"b1-cap{cap}"]) for cap in CAPS}
    result = {"sources": {**digests, "source_repo": workload["source_repo"],
                          "source_revision": workload["source_revision"]},
              "controls": {"sessions": 48, "requests": 254, "root_window_s": WINDOW, "seeds": SEEDS,
                           "tool_gap_s_in_manifest": sorted({r["tool_gap_s"] for r in spec.values()}),
                           "settings": manifest["settings"], "EOS": "frozen archived natural stop per block"},
              "calibration": {}, "reference_validation": {}, "results": {}, "continuous_rate_scan": []}
    plan_book = {"root_plans_s": {}, "EOS_by_block": {}}
    dest.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest / "requests.jsonl.gz", "wt", encoding="utf-8") as stream:
        for block in (1, 2):
            service, fit = fit_service(native, raw["b1-cap200"], method["b1-cap200"], block, prefill)
            result["calibration"][block] = fit
            outputs = reference_eos(native, spec, block)
            plan_book["EOS_by_block"][block] = outputs
            sessions, original, ids = materialize(spec, prefix, outputs)
            plans = {"original": original, **{f"conditional-{seed}": conditional_arrivals(original, seed) for seed in SEEDS}}
            plan_book["root_plans_s"] = plans
            expected_tokens = sum(s.output for s in sessions)
            block_result, validation = {}, {}
            result["results"][block] = block_result
            result["reference_validation"][block] = validation
            order = CAPS if block == 1 else CAPS[::-1]
            for label, offers in plans.items():
                block_result[label], routes = {}, {}
                for cap in order:
                    run = Replay(sessions, offers, service, cap).run()
                    details = metrics(run, offers, detailed=True)
                    if details["output_tokens"] != expected_tokens or run.requests.keys() != ids.keys():
                        raise RuntimeError("Simulation changed the common request or EOS ledger")
                    routes[cap] = {key: (j.row["P"], j.row["D"]) for key, j in run.requests.items()}
                    block_result[label][cap] = details
                    emit_requests(stream, run, ids, label, cap, block)
                low, high = (block_result[label][cap] for cap in CAPS)
                block_result[label]["comparison"] = {
                    "200_vs_100_throughput_pct": 100 * (high["throughput_tokens_s"] / low["throughput_tokens_s"] - 1),
                    "same_p_d_routes_across_caps": [sum(routes[100][k][i] == routes[200][k][i] for k in ids) for i in range(2)],
                }
            for cap in order:
                tag = f"b{block}-cap{cap}"
                validation[tag] = calibration_check(spec, prefix, service, cap, native[tag][1])
                if block == 1:
                    origin, = (q["monotonic_s"] for q in raw[tag]["client"] if q["type"] == "client_origin")
                    actual = requests_for(raw[tag], workload["request_sources"], origin)
                    validation[f"profile-{tag}"] = calibration_check(spec, prefix, service, cap, actual, method[tag])
            print(f"r43 block {block}: fixed-window plans and archive checks complete", flush=True)
            result["continuous_rate_scan"].extend(run_curves(sessions, original, service, block))
            print(f"r43 block {block}: independent-start rate scan complete", flush=True)
    write_json(dest / "arrivals.json", plan_book)
    result["artifacts"] = {name: sha256(dest / name) for name in ("requests.jsonl.gz", "arrivals.json")}
    write_json(dest / "summary.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.results, args.output)


if __name__ == "__main__":
    main()
