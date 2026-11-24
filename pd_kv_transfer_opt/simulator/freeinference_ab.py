"""Profile whole FreeInference sessions in paired native CPU bandwidth replays."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import gzip
import json
from math import isclose
from pathlib import Path

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.simulator.agent_replay import GB, PAGE, Replay, Service
from pd_kv_transfer_opt.simulator.decode_supply import decode_samples, fit_decode, native_arms
from pd_kv_transfer_opt.simulator.terminal_pressure import peak_window
from pd_kv_transfer_opt.workloads.freeinference import load


ARCHIVE_1 = "2026-11-22-r73-terminal-source-restore-expanded"
ARCHIVE_2 = "2026-11-22-r75-terminal-source-restore-pilot"
R78 = "2026-11-22-r78-terminal-source-restore-stable-tail"
SOURCE_GATE = "2026-11-22-cpu-agent-trace-gap"
PUBLISHED_BASELINE = "5d591893ca8c7adc03c797068b44ad43dad27398"
ORDER = {"forward": (100, 200), "reverse": (200, 100)}


def calibration(results):
    workload = read_json(results / R78 / "manifest.json")["workload"]
    input_fit = workload["arrival_order_screen"]
    ingress = input_fit["admission_fit"]
    source = workload["request_sources"]
    bytes_per_token = percentile([input_fit["wire_body_bytes"][r["id"]] / r["prompt_tokens"] for r in source], .5)
    archived = read_json(results / "2026-11-22-cpu-terminal-supply" / "summary.json")
    models = {}
    for order, ref, path, width in zip(ORDER, archived["reference_states"], (ARCHIVE_1, ARCHIVE_2), (4, 3), strict=True):
        digest, arms = native_arms(results / path)
        fit = fit_decode([row for arm in arms.values() for row in decode_samples(*arm)])
        service = Service(tuple(ref["shares"]), fit["first_sequence_period_s"], fit["additional_sequence_period_s"],
                          ref["cold_tokens_s_per_P"], ingress["offset_s"], ingress["s_per_byte"],
                          bytes_per_token, width)
        models[order] = (service, {"decode_archive": path, "decode_trace_sha256": digest, "decode_fit": fit,
                                   "prefill_and_rail_archive": ref["archive"], "prefill_and_rail_case": ref["case"],
                                   "prefill_and_rail_trace_sha256": ref["trace_sha256"],
                                   "ingress_source": R78})
    return models


def stage_profile(replay):
    durations = Counter(dict.fromkeys(("D_mean_active_area", "both_D_full_s", "any_D_running_s",
                                       "no_P_work_or_D_active_or_waiting_s", "D_egress_wire_busy_s",
                                       "D_active_seq_s_while_GET", "both_D_full_while_GET_s",
                                       "both_D_idle_with_GET_and_no_D_queue_s",
                                       "one_or_more_D_idle_with_GET_and_no_local_D_queue_s"), 0.0))
    longest = consecutive = 0.0
    p_compute = [0.0, 0.0]
    p_get = [0.0, 0.0]
    for start, end, widths, pending, stages, gets, d_backlog in replay.supply_epochs:
        dt = end - start
        for role, stage in enumerate(stages):
            if stage == "compute":
                p_compute[role] += dt
            if stage == "get" or stage == "get_setup":
                p_get[role] += dt
        width = sum(widths)
        durations["D_mean_active_area"] += dt * width
        if width == replay.service.slots * 2:
            durations["both_D_full_s"] += dt
        if width:
            durations["any_D_running_s"] += dt
        if not width and not sum(pending) and not sum(d_backlog):
            durations["no_P_work_or_D_active_or_waiting_s"] += dt
        if gets:
            durations["D_egress_wire_busy_s"] += dt
            durations["D_active_seq_s_while_GET"] += dt * width
            if width == replay.service.slots * 2:
                durations["both_D_full_while_GET_s"] += dt
            if not width and not sum(d_backlog):
                durations["both_D_idle_with_GET_and_no_D_queue_s"] += dt
            if any(not w and not q for w, q in zip(widths, d_backlog, strict=True)):
                durations["one_or_more_D_idle_with_GET_and_no_local_D_queue_s"] += dt
            consecutive += dt
            longest = max(longest, consecutive)
        else:
            consecutive = 0.0
    return {**dict(durations), "mean_D_sequences_while_GET": (
        durations["D_active_seq_s_while_GET"] / durations["D_egress_wire_busy_s"]
        if durations["D_egress_wire_busy_s"] else None),
        "longest_continuous_D_egress_wire_s": longest,
        "P_compute_wall_s_by_instance": p_compute, "P_synchronous_GET_wall_s_by_instance": p_get,
        "D_sequence_slots_peak": [max(episode[2][role] for episode in replay.supply_epochs) for role in range(2)],
        "D_busy_fraction_of_peak_sequence_slots": durations["D_mean_active_area"] / replay.now /
        (replay.service.slots * 2)}


def supply_pair(low, high, *, start_s=0.):
    begin = max(min((job.row["D_first_s"] for job in replay.requests.values() if job.row["D"] == role),
                    default=replay.now)
                for replay in (low, high) for role in range(2))
    begin = max(begin, start_s)
    end = min(max(job.row["P_reply_s"] for job in replay.requests.values()) for replay in (low, high))
    comparison = Counter()
    i = j = 0
    continuous = longest = 0.0
    while i < len(low.supply_epochs) and j < len(high.supply_epochs):
        a, b = low.supply_epochs[i], high.supply_epochs[j]
        duration = min(a[1], b[1], end) - max(a[0], b[0], begin)
        if duration > 0:
            if a[5]:
                low_active, high_active = sum(a[2]), sum(b[2])
                comparison["100_wire_busy_s"] += duration
                comparison["100_active_seq_s"] += duration * low_active
                comparison["200_active_seq_s"] += duration * high_active
                if low_active < high_active and any(a[3]) and not all(a[6]):
                    comparison["100_fewer_D_active_with_P_pending_s"] += duration
                    continuous += duration
                    longest = max(longest, continuous)
                else:
                    continuous = 0.0
            else:
                continuous = 0.0
        i += a[1] <= b[1]
        j += b[1] <= a[1]
    active = comparison["100_wire_busy_s"]
    return {"common_begin_s": begin, "common_end_s": end, **dict(comparison),
            "longest_100_fewer_D_active_with_P_pending_s": longest,
            "mean_active_D_during_100_GET": [comparison[k] / active for k in ("100_active_seq_s", "200_active_seq_s")]
            if active else None}


def report(replay, *, detailed=True):
    rows = list(replay.requests.values())
    total = Counter()
    for job in rows:
        r = job.row
        total.update({k: r[k] for k in ("P_local_pages", "P_external_pages", "P_lookup_pages", "P_computed_tokens",
                                        "P_adjacent_local_pages", "P_adjacent_store_pages", "P_adjacent_eviction_pages",
                                        "P_store_capacity_lost_pages", "GET_bytes", "GET_states",
                                        "D_local_pages", "P_to_D_bytes")})
        total["store_hit_requests"] += r["GET_bytes"] > 0
        total["adjacent_pages"] += job.turn.adjacent_pages
    output = sum(job.turn.output for job in rows)
    volume = total["GET_bytes"] / GB
    per_port = [sum((b - a) * rates[i] / 8 for a, b, rates in replay.tx_epochs)
                for i in range(8)]
    if any(not isclose(gb, volume * share, rel_tol=1e-8, abs_tol=1e-6)
           for gb, share in zip(per_port, replay.service.shares, strict=True)):
        raise RuntimeError("GET payload and per-port wire service disagree")
    cache = {
        **dict(total), "P_missed_lookup_pages": total["P_lookup_pages"] - total["P_local_pages"],
        "P_novel_lookup_pages": total["P_lookup_pages"] - total["P_local_pages"] - total["P_external_pages"]
        - total["P_store_capacity_lost_pages"],
        "adjacent_not_yet_seen_or_not_compatible_pages": total["adjacent_pages"]
        - sum(total[k] for k in ("P_adjacent_local_pages", "P_adjacent_store_pages", "P_adjacent_eviction_pages")),
        "Store_peak_GB": replay.store.peak / GB, "Store_evicted_objects": replay.store.evicted,
        "Store_evicted_GB": replay.store.eviction_bytes / GB, "Store_failed_puts": replay.store.failed,
        "Store_committed_GB": replay.store.written / GB, "P_to_D_Store_put_GB": replay.put_GB,
        "P_HBM_evicted_pages": [replica.cache.evictions for replica in replay.p],
        "D_HBM_evicted_pages": [replica.cache.evictions for replica in replay.d]}
    result = {"requests": len(rows), "output_tokens": output, "finish_s": replay.now,
              "throughput_tokens_s": output / replay.now, "GET_GB": volume, "GET_MB_per_output": volume * 1000 / output,
              "store_model": {"layout_id": "historical-d-only",
                              "capacity_bytes_by_role": {"P": 0, "D": replay.service.store_bytes},
                              "GET_path": "D_to_P", "two_sided_layout_calibrated": False},
              "D_to_P_GB_by_port": per_port, "cache": cache, "stage": stage_profile(replay)}
    if not detailed:
        return result
    tail = max(rows, key=lambda job: job.row["EOS_s"])
    chain = [replay.requests[f"{tail.session.name}:{step}"] for step in range(len(tail.session.turns))]
    stage = Counter()
    for job in chain:
        r = job.row
        wire_s = r["GET_done_s"] - r["GET_submit_s"] if r["GET_bytes"] else 0.0
        stage.update(tool_gap_s=job.turn.tool_gap_ms / 1000, P_ingress_s=r["P_admit_s"] - job.offered,
                     P_queue_s=r["P_dispatch_s"] - r["P_admit_s"], P_own_GET_s=wire_s,
                     P_compute_s=r["P_compute_s"], P_other_or_cohort_s=(r["P_reply_s"] - r["P_dispatch_s"]
                        - wire_s - r["P_compute_s"]), D_alloc_from_P_reply_s=r["D_alloc_s"] - r["P_reply_s"],
                     P_to_D_pull_s=r["D_KV_ready_s"] - r["D_alloc_s"], D_generation_s=r["EOS_s"] - r["D_KV_ready_s"])
    stage["first_turn_plan_s"] = chain[0].offered
    stage["planned_release_wait_s"] = sum(
        max(0, right.offered - left.row["EOS_s"] - right.turn.tool_gap_ms / 1000)
        for left, right in zip(chain, chain[1:]))
    if not isclose(sum(stage.values()), replay.now, abs_tol=1e-6):
        raise RuntimeError("The last session chain does not sum to the drained duration")
    last_P = max(job.row["P_reply_s"] for job in rows)
    result.update(
        peak_D_to_P_aggregate_Gbps={str(window): peak_window(replay.tx_epochs, window) for window in (1, 10, 30, 120)},
        peak_1s_D_to_P_port_Gbps=[peak_window([(a, b, (rates[i],)) for a, b, rates in replay.tx_epochs], 1)
                                 for i in range(8)],
        D_to_P_hot_port_floor_Gbps=.85 * replay.cap / 8,
        own_GET_wait_p50_p95_s=[percentile(w, q) for q in (.5, .95)] if
            (w := [job.row["GET_done_s"] - job.row["GET_submit_s"] for job in rows if job.row["GET_bytes"]]) else [],
        tail_session={"session": tail.session.name, "turns": len(chain), "phases_s": dict(stage)},
        last_P_reply_s=last_P, drain_after_last_P_reply_s=replay.now - last_P)
    return result


def trace(dest, workload, order, cap, replay):
    if dest is None:
        return
    for key, job in replay.requests.items():
        row = {"workload": workload, "order": order, "cap": cap, "key": key,
               "prompt": job.turn.prompt, "output": job.turn.output, "tool_gap_ms": job.turn.tool_gap_ms,
               "finish": job.turn.finish, "adjacent_pages": job.turn.adjacent_pages, **job.row}
        print(json.dumps(row, separators=(",", ":")), file=dest)


def paired(sessions, offers, service, order, *, workload="", dest=None, detailed=True, collect=False):
    runs, arms = {}, {}
    for cap in ORDER[order]:
        model = Replay(sessions, offers, service, cap).run()
        runs[cap], arms[cap] = model, report(model, detailed=detailed)
        trace(dest, workload, order, cap, model)
    low, high = runs[100], runs[200]
    if arms[100]["output_tokens"] != arms[200]["output_tokens"] or low.requests.keys() != high.requests.keys():
        raise RuntimeError("A bandwidth change altered the requests or natural output work")
    summary = {"arms": arms, "100_loss_relative_to_200": 1 - high.now / low.now,
               "200_gain_relative_to_100": low.now / high.now - 1,
               "same_P_routes": sum(row.row["P"] == high.requests[key].row["P"] for key, row in low.requests.items()),
               "same_D_routes": sum(row.row["D"] == high.requests[key].row["D"] for key, row in low.requests.items()),
               "D_supply_pair": supply_pair(low, high)}
    if not collect:
        return summary
    # Keep the pilot counts needed to select a new cohort using cache outcomes.
    stats = {}
    for cap, run in runs.items():
        grouped = {}
        for job in run.requests.values():
            data = grouped.setdefault(job.session.name, Counter())
            data.update({name: job.row[name] for name in ("GET_bytes", "P_store_capacity_lost_pages",
                                                          "P_local_pages", "P_computed_tokens")})
            data["tool_s"] += job.turn.tool_gap_ms / 1000
            data["output"] += job.turn.output
        stats[cap] = grouped
    return summary, stats


def workload_info(sessions, offers):
    outputs = sum(session.output for session in sessions)
    return {"sessions": [{"id": s.name, "root_offered_s": offers[s.name], "requests": len(s.turns),
                          "output": s.output, "tool_s": sum(t.tool_gap_ms for t in s.turns) / 1000} for s in sessions],
            "requests": sum(len(s.turns) for s in sessions), "output": outputs,
            "adjacent_full_read_GB": sum(s.source_read_bytes for s in sessions) / GB,
            "finish_reasons": dict(Counter(turn.finish for session in sessions for turn in session.turns))}


def build_candidates(sessions, pilots, threshold):
    gaps = {s.name: sum(turn.tool_gap_ms for turn in s.turns) / 1000 for s in sessions}
    eligible = [s for s in sessions if gaps[s.name] <= 600 and len(s.turns) >= 4 and s.source_read_bytes >= GB]
    density = sorted(eligible, key=lambda s: (-s.source_read_bytes / s.output, s.name))
    short = [s for s in density if gaps[s.name] <= 300]
    tight = [s for s in short if gaps[s.name] <= 120 and s.source_read_bytes >= threshold * s.output]
    witnessed = []
    for session in density:
        rows = [rows[session.name] for rows in pilots.values() if session.name in rows]
        if len(rows) == len(pilots):
            floor = min(r["GET_bytes"] / r["output"] for r in rows)
            if floor:
                witnessed.append((-floor, session.name, session))
    ready = [s for _, _, s in sorted(witnessed)[:24]]

    groups = defaultdict(set)
    for session in short:
        for turn in session.turns:
            if len(turn.pages) >= 64:
                groups[turn.pages[63]].add(session.name)
    shared = sorted((members for members in groups.values() if len(members) > 1),
                    key=lambda members: (-len(members), sorted(members)))
    related = set().union(*shared[:4])
    families = [s for s in short if s.name in related]
    members = [*families, *(s for s in short if s.name not in related)][:32]

    definitions = (
        ("tight_120", tight, 0),
        ("compact_16", short[:16], 0),
        ("short_32", short[:32], 0),
        ("pressure_48", density[:48], 0),
        ("pressure_64", density[:64], 0),
        ("witnessed_store_24", ready, 0),
        ("witnessed_store_24_paced", ready, 1),
        ("shared_64K_32", members, 0),
        ("shared_64K_32_paced", members, 1),
        ("pressure_48_paced", density[:48], 1),
    )
    candidates = {}
    for name, selected, interval in definitions:
        ordered = sorted(selected, key=lambda s: (s.source_start_ms, s.name))
        if len(ordered) >= 2:
            candidates[name] = (ordered, {s.name: rank * interval for rank, s in enumerate(ordered)})
    return candidates, {"compatible_with_600s_gap_and_at_least_4_turns_and_1GB": len(eligible),
                        "among_these_with_at_most_300s_total_original_gap": len(short),
                        "among_these_with_at_most_120s_gap_and_original_density_gate": len(tight),
                        "short_sessions_sharing_selected_64K_prefix_families": len(families),
                        "max_tool_gap_definition": "sum of the original EOS-to-next-send intervals per full session"}


def rescreen(corpus, pilots, threshold, models, trace_dest):
    plans, eligibility = build_candidates(corpus, pilots, threshold)
    service = models["forward"][0]
    runs = {}
    for name, (sessions, offers) in plans.items():
        test = report(Replay(sessions, offers, service, 100).run(), detailed=False)
        duration = test["finish_s"]
        wire = test["stage"]["D_egress_wire_busy_s"]
        longest = test["stage"]["longest_continuous_D_egress_wire_s"]
        runs[name] = {"workload": workload_info(sessions, offers), "100_screen": test,
                      "100_link_duty": wire / duration,
                      "100_longest_continuous_get_s": longest,
                      "min_duration_from_one_session_tool_and_decode_s": max(
                          offers[s.name] + sum(t.tool_gap_ms for t in s.turns) / 1000 + s.output * service.decode_first_s
                          for s in sessions)}
        print("screen", name, round(test["GET_MB_per_output"], 2), round(100 * wire / duration, 2),
              round(longest, 2), flush=True)
    best = sorted(runs, key=lambda name: (-runs[name]["100_link_duty"],
                                          -runs[name]["100_longest_continuous_get_s"], name))[:3]
    tested = {}
    for name in best:
        sessions, offers = plans[name]
        tested[name] = {}
        for order, (service, _) in models.items():
            result = paired(sessions, offers, service, order, workload=name, dest=trace_dest)
            tested[name][order] = result
            print("rescreen", name, order, [round(result["arms"][cap]["throughput_tokens_s"], 2) for cap in (100, 200)],
                  round(100 * result["100_loss_relative_to_200"], 3), flush=True)
    strongest = best[0]
    sessions, offers = plans[strongest]
    sensitivity = paired(sessions, offers, replace(models["reverse"][0], checkpoint_stride=4), "reverse",
                         workload=f"{strongest}_sparse_checkpoint", dest=trace_dest)
    return {
        "eligibility": eligibility,
        "rule": "Rank native 100 Gbps pilots by link duty, then uninterrupted GET duration. Repeat the leading "
        "three in both bandwidth orders. Roots are offered once, simultaneously or one second apart in source order.",
        "screened": runs, "ABBA": tested,
        "checkpoint_spacing_sensitivity": {"workload": strongest, "alignment_stride_in_1024_token_pages": 4,
                                            "order": "reverse", "pair": sensitivity},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    header, sessions = load(args.corpus)
    models = calibration(args.results)
    selected = [s for s in sessions if s.source_read_bytes >= header["high_read_bytes_per_output"] * s.output]
    if len(selected) != 98:
        raise ValueError("The requested 98-session cohort did not match the source gate")
    offers = dict.fromkeys((s.name for s in selected), 0.0)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    result = {"baseline_revision": PUBLISHED_BASELINE, "corpus": header, "corpus_sha256": sha256(args.corpus),
              "prior_screen_sha256": sha256(args.results / SOURCE_GATE / "summary.json"),
              "scope": "CPU only. Provider-recorded full-session completion lengths, 16-token hash chains, incoming "
              "size proxy, original tool gaps, cold lease-accounted P/D APC, and a shared 128 GiB Store. "
              "Store fill, hit, checkpoint, and egress outcomes are simulated. Paired arms reset all caches. "
              "Only D eight-port egress changes inside a pair; archive calibrations differ across orders.",
              "complete_compatible_sessions": len(sessions),
              "models": {k: {"service": asdict(s), **m} for k, (s, m) in models.items()},
              "high98": {"workload": workload_info(selected, offers), "orders": {}}}
    with gzip.open(args.trace, "wt", encoding="utf-8", compresslevel=4) as dest:
        pilots = {}
        for order, (service, _) in models.items():
            pair, stats = paired(selected, offers, service, order, workload="high98", dest=dest, collect=True)
            result["high98"]["orders"][order] = pair
            for cap, observations in stats.items():
                pilots[order, cap] = observations
            print("high98", order, [round(pair["arms"][cap]["throughput_tokens_s"], 4) for cap in (100, 200)],
                  round(100 * pair["100_loss_relative_to_200"], 3), flush=True)
        if not all(p["100_loss_relative_to_200"] >= .30 and
                   p["D_supply_pair"]["longest_100_fewer_D_active_with_P_pending_s"] >= 30
                   for p in result["high98"]["orders"].values()):
            result["rescreen"] = rescreen(sessions, pilots, header["high_read_bytes_per_output"], models, dest)
    result["request_trace_sha256"] = sha256(args.trace)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
