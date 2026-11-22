"""Audit cache feedback and replay a wall-clock Terminal-Bench arrival plan."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import gzip
from math import ceil, isclose
from pathlib import Path

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BYTES
from pd_kv_transfer_opt.simulator.agent_replay import EPS, GB, PAGE, Replay
from pd_kv_transfer_opt.simulator.agent_trace_gap import read_bytes
from pd_kv_transfer_opt.simulator.freeinference_ab import calibration, supply_pair, trace
from pd_kv_transfer_opt.simulator.terminal_prefill_calibration import clock_profiles
from pd_kv_transfer_opt.simulator.terminal_pressure import peak_window
from pd_kv_transfer_opt.simulator.terminal_rotation_pressure import Source, audit_run, workload


PRIOR = "2026-11-22-cpu-terminal-rotation-pressure"
R78 = "2026-11-22-r78-terminal-source-restore-stable-tail"
ORDERS = {"forward": (100, 200), "reverse": (200, 100)}
TARGET_LOSS = .30


def design_plan(domains, models, previous, profiles):
    """Size complete waves with room at the slower 200 Gbps reference."""
    residual_span = max(max(later) - min(later) for _, later in profiles.values())
    gap = ceil(1000 * (residual_span + .002)) / 1000
    rates = [100 * service.rail_efficiency / (64 * max(service.shares))
             for service, _ in models.values()]
    offer = previous["design"]["target_offered_GB_s_if_both_P_need_GET"]
    full_wave_GB = 2 * sum(read_bytes(source.pages) for source in domains) / GB
    period = full_wave_GB / offer
    # The first sweep is deliberately slow so that the cold preparation's
    # asymmetric cache hits do not determine the first regular wave's routing.
    worst_pair_s = max(
        2 * read_bytes(source.pages) / GB / rate + service.get_setup_s
        + (source.prompt % PAGE) / service.warm_tokens_s + service.decode_ingress_s
        + service.pull_setup_s + ceil(source.prompt / PAGE) * ATTENTION_BYTES / GB / service.pull_GB_s
        + service.decode_first_s + (service.slots - 1) * service.decode_add_s
        for (service, _), rate in zip(models.values(), rates, strict=True) for source in domains
    )
    first_gap = ceil(20 * (worst_pair_s + residual_span + .25)) / 20
    start = previous["paired_clients"]["design"]["pressure_start_s"]
    regular_start = start + len(domains) * first_gap
    # If every wave really reads the full set, grant enough waves to
    # amortize cold preparation even at the faster low-cap calibration.
    budget = (1 - TARGET_LOSS) * full_wave_GB / max(rates) - period
    if budget <= 0 or not max(rates) < offer < 2 * min(rates):
        raise ValueError("The proposed wall-clock period has no usable link service window")
    drain_budget_s = 5
    regular_waves = ceil(TARGET_LOSS * (regular_start + drain_budget_s) / budget) + 2
    plan = workload(domains, rounds=regular_waves + 1, pressure_start=start,
                    read_offer_GB_s=offer, preparation_copies=1)
    _, offers, _, rows = plan
    sorted_slots = sorted(range(len(domains)), key=lambda slot: (domains[slot].body_bytes, slot))
    rank = {slot: i for i, slot in enumerate(sorted_slots)}
    for row in rows:
        if row["phase"] == "restore":
            cycle = row["cycle"]
            at = (start + rank[row["slot"]] * first_gap if cycle == 0 else
                  regular_start + (cycle - 1) * period + rank[row["slot"]] * gap)
            row["root_offered_s"] = offers[row["name"]] = at
    return plan, {
        "pressure_start_s": start, "first_sweep_gap_s": first_gap, "regular_start_s": regular_start,
        "admission_residual_span_s": residual_span, "regular_pair_gap_s": gap,
        "regular_window_period_s": period, "regular_windows": regular_waves,
        "domain_slots_ordered_by_body": sorted_slots, "full_wave_GB": full_wave_GB,
        "offered_GB_s_if_both_P_need_GET": offer, "effective_100_GB_s_by_reference": rates,
        "optimistic_seconds_per_wave_towards_30pct_loss": budget,
        "design_drain_s": drain_budget_s, "full_work_sizing_extra_windows": 2,
    }


def route_credits(requests):
    """Reconstruct every native P choice from admissions and first-token returns."""
    events = []
    for key, row, score, ordinal in requests:
        events.append((row["P_admit_s"], 1, ordinal, key, row["P"], score))
        events.append((row["D_first_s"], 0, ordinal, key, row["P"], score))
    balance = [0., 0.]
    before = {}
    for _, admitted, _, key, role, score in sorted(events, key=lambda event: event[:3]):
        if admitted:
            before[key] = tuple(balance)
            if role != int(balance[1] + EPS < balance[0]):
                raise RuntimeError(f"Could not reproduce the original native route for {key}")
            balance[role] += score
        else:
            balance[role] -= score
    if any(abs(credit) > 1e-6 for credit in balance):
        raise RuntimeError("P scores did not return to zero after all first tokens")
    return before


def cache_feedback(replay, rows, domains):
    restore = [row for row in rows if row["phase"] == "restore"]
    credits = route_credits((job.key, job.row, job.p_score, job.session.source_start_ms)
                            for job in replay.requests.values())
    by_pair, by_role = defaultdict(list), defaultdict(list)
    fields = ("P_local_pages", "P_external_pages", "P_lookup_pages", "P_computed_tokens",
              "P_store_capacity_lost_pages", "GET_states", "GET_bytes")
    work = Counter(dict.fromkeys(fields, 0))
    for row in restore:
        job = replay.requests[f"{row['name']}:0"]
        by_pair[row["cycle"], row["slot"]].append(job)
        by_role[job.row["P"]].append((row["slot"], job))
        work.update({field: job.row[field] for field in fields})
    pair_keys = list(by_pair)
    split_failure = next((key for key, pair in by_pair.items()
                          if pair[0].row["P"] == pair[1].row["P"]), None)
    example = None
    if split_failure is not None:
        index = pair_keys.index(split_failure)
        prior = by_pair[pair_keys[index - 1]] if index else ()
        example = {
            "cycle": split_failure[0], "domain_slot": split_failure[1],
            "routing": [{"key": job.key, "P": job.row["P"], "P_scores_before": credits[job.key],
                         "score": job.p_score, "P_admit_s": job.row["P_admit_s"],
                         "P_local_pages": job.row["P_local_pages"], "GET_bytes": job.row["GET_bytes"]}
                        for job in by_pair[split_failure]],
            "previous_pair": [{"key": job.key, "P": job.row["P"], "score": job.p_score,
                               "D_first_s": job.row["D_first_s"], "GET_bytes": job.row["GET_bytes"]}
                              for job in prior],
        }
    distances = {"local_only": [], "with_GET": []}
    for placements in by_role.values():
        placements.sort(key=lambda item: (item[1].row["P_dispatch_s"], item[1].row["P_admit_s"],
                                          item[1].session.source_start_ms))
        last = {}
        for i, (slot, job) in enumerate(placements):
            if slot in last:
                between = {key for key, _ in placements[last[slot] + 1:i]}
                distances["with_GET" if job.row["GET_bytes"] else "local_only"].append(
                    sum(domains[key].pages for key in between))
            last[slot] = i
    return {
        "audited_P_routes": len(credits), "restore_work": dict(work),
        "pairs_routed_to_one_P": sum(pair[0].row["P"] == pair[1].row["P"] for pair in by_pair.values()),
        "first_pair_routed_to_one_P": example,
        "intervening_distinct_complete_domain_pages": {
            label: {"requests": len(values), "p10_p50_p90": [percentile(values, q) for q in (.1, .5, .9)]}
            for label, values in distances.items()
        },
    }


def wave_profile(replay, rows, design):
    """Measure whether GET, delivery and decode drain between fixed waves."""
    regular = defaultdict(list)
    for row in rows:
        if row["phase"] == "restore" and row["cycle"] > 0:
            regular[row["cycle"]].append(replay.requests[f"{row['name']}:0"].row)
    start = design["regular_start_s"]
    period = design["regular_window_period_s"]
    waves = []
    for index, jobs in sorted(regular.items()):
        planned_start = start + (index - 1) * period
        waves.append({"cycle": index, "first_offer_s": planned_start, "end_of_period_s": planned_start + period,
                      "GET_GB": sum(job["GET_bytes"] for job in jobs) / GB,
                      "GET_last_done_s": max((job["GET_done_s"] for job in jobs if job["GET_bytes"]), default=None),
                      "last_P_reply_s": max(job["P_reply_s"] for job in jobs),
                      "last_EOS_s": max(job["EOS_s"] for job in jobs)})
    end = max(wave["last_P_reply_s"] for wave in waves)
    span = end - start
    tx = [(max(a, start), min(b, end), rates) for a, b, rates in replay.tx_epochs if b > start and a < end]
    busy = sum(b - a for a, b, rates in tx if any(rates))
    port_GB = [sum((b - a) * rates[i] / 8 for a, b, rates in tx) for i in range(8)]
    supply = Counter(dict.fromkeys(("mean_running_sequences", "both_D_full_s",
                                    "both_D_idle_and_unfed_with_GET_s"), 0.))
    for a, b, widths, _, _, gets, backlog in replay.supply_epochs:
        dt = max(0, min(b, end) - max(a, start))
        supply["mean_running_sequences"] += dt * sum(widths) / span
        if sum(widths) == 2 * replay.service.slots:
            supply["both_D_full_s"] += dt
        if gets and not any(widths) and not any(backlog):
            supply["both_D_idle_and_unfed_with_GET_s"] += dt
    cleared = {field: sum(wave[field] is None or wave[field] <= wave["end_of_period_s"] for wave in waves[:-1])
               for field in ("GET_last_done_s", "last_P_reply_s", "last_EOS_s")}
    return {"start_s": start, "last_P_reply_s": end, "wire_busy_fraction": busy / span,
            "wire_peak_30s_Gbps": peak_window(tx, 30), "effective_wire_ceiling_Gbps": 8 * replay.GB_s,
            "observed_payload_GB_by_port": port_GB, "mean_payload_Gbps_by_port": [gb * 8 / span for gb in port_GB],
            "cleared_before_next_offer": cleared, "intervals_with_next_offer": len(waves) - 1,
            "last_offer_to_EOS_s": replay.now - max(job.offered for job in replay.requests.values()),
            "D_supply": dict(supply), "waves": waves}


def compare(plan, service, order, start, name, *, residuals=None, output=None, domains=None, design=None):
    sessions, offers, sizes, rows = plan
    runs, arms = {}, {}
    output_tokens = sum(row["output_tokens"] for row in rows)
    for cap in ORDERS[order]:
        run = Replay(sessions, offers, service, cap, observed_body_bytes=sizes, ingress_residual_s=residuals).run()
        if run.done != len(rows) or any(job.offered != offers[job.session.name] or job.turn.finish != "stop"
                                        for job in run.requests.values()):
            raise RuntimeError("Replay changed the wall-clock plan or natural EOS reference")
        arm = audit_run(run, rows, start)
        if arm["output_tokens"] != output_tokens:
            raise RuntimeError("Replay changed the frozen output work")
        if domains is not None:
            arm["routing_feedback"] = cache_feedback(run, rows, domains)
        if design is not None:
            arm["regular_waves"] = wave_profile(run, rows, design)
            arm["regular_phase_read_GB"] = sum(wave["GET_GB"] for wave in arm["regular_waves"]["waves"])
        runs[cap], arms[cap] = run, arm
        trace(output, name, order, cap, run)
    low, high = arms[100], arms[200]
    deficit_s = max(0, high["finish_s"] / (1 - TARGET_LOSS) - low["finish_s"])
    return {"arms": arms, "100_loss_relative_to_200": 1 - low["throughput_tokens_s"] / high["throughput_tokens_s"],
            "100_additional_completion_s_for_30pct": deficit_s,
            "100_additional_GB_if_all_new_wire_service_delays_completion": deficit_s * runs[100].GB_s,
            "same_P_placement_requests": sum(job.row["P"] == runs[200].requests[key].row["P"]
                                             for key, job in runs[100].requests.items()),
            "D_supply_comparison": supply_pair(runs[100], runs[200], start_s=start)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    previous_path = args.results / PRIOR / "summary.json"
    previous = read_json(previous_path)
    archived_plan = previous_path.with_name("workload.json")
    if sha256(archived_plan) != previous["workload_sha256"]:
        raise ValueError("Previous arrival plan digest did not match the calibration record")
    domains = [Source(**row) for row in previous["design"]["ordered_sources"]]
    models = {name: (replace(service, release_suffix_first=True), meta)
              for name, (service, meta) in calibration(args.results).items()}
    fit = read_json(args.results / R78 / "manifest.json")["workload"]["arrival_order_screen"]["admission_fit"]
    profiles = clock_profiles(args.results, fit)
    plan, design = design_plan(domains, models, previous, profiles)
    pressure_start = design["pressure_start_s"]
    matched = workload(domains, rounds=previous["paired_clients"]["design"]["cycles"],
                       pressure_start=pressure_start,
                       read_offer_GB_s=design["offered_GB_s_if_both_P_need_GET"], preparation_copies=1)
    if matched[3] != read_json(archived_plan)["paired_clients_earliest_offer_if_unblocked"]:
        raise ValueError("The earlier fixed-arrival plan changed")
    ingress_name, (root, later) = max(profiles.items(), key=lambda item: max(item[1][1]) - min(item[1][1]))
    residuals = {}
    for i, row in enumerate(plan[3]):
        delays = root if row["phase"] == "prepare" else later
        residuals[f"{row['name']}:0"] = delays[i % len(delays)]
    frozen = args.output.with_name("workload.json")
    scope = "CPU only. Exact Terminal-Bench snapshot fingerprints and EOS-reference counts, reused in separate cache domains."
    write_json(frozen, {"scope": scope, "design": design, "source_workload_sha256": previous["workload_sha256"],
                        "ordered_sources": previous["design"]["ordered_sources"], "requests": plan[3],
                        "fixed_ingress_residual_profile": ingress_name, "fixed_ingress_residual_s_by_request": residuals})
    out = {"scope": scope, "baseline_revision": previous["baseline_revision"],
           "previous_summary_sha256": sha256(previous_path), "workload_sha256": sha256(frozen),
           "source_repo": previous["source_repo"], "source_revision": previous["source_revision"],
           "ledger_sha256": previous["ledger_sha256"], "eos_receipt_sha256": previous["eos_receipt_sha256"],
           "models": previous["models"], "design": design, "ingress_residual_profile": ingress_name,
           "matched_945_fixed_arrivals": {}, "preplanned_arrivals": {"nominal": {}, "archive_ingress": {}}}
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.trace, "wt", encoding="utf-8", compresslevel=4) as dest:
        for order, (service, _) in models.items():
            original = compare(matched, service, order, pressure_start, "matched_945", output=dest, domains=domains)
            for cap in ORDERS[order]:
                reference = previous["paired_clients"]["same_requests_without_first_token_gate"][order]["arms"][str(cap)]
                if any(not isclose(original["arms"][cap][key], reference[key], abs_tol=1e-8)
                       for key in ("finish_s", "GET_GB", "output_tokens")):
                    raise RuntimeError("The archived fixed-arrival result could not be reproduced")
            out["matched_945_fixed_arrivals"][order] = original
            for label, ingress in (("nominal", None), ("archive_ingress", residuals)):
                value = compare(plan, service, order, pressure_start, label, residuals=ingress, output=dest, design=design)
                out["preplanned_arrivals"][label][order] = value
                print(label, order, value["100_loss_relative_to_200"], flush=True)
                if order == "forward" and label == "nominal":
                    rerun = compare(plan, service, "reverse", pressure_start, "same_service_ABBA", design=design)
                    out["same_service_ABBA_reproduced"] = rerun["arms"] == value["arms"]
                    if not out["same_service_ABBA_reproduced"]:
                        raise RuntimeError("Wall-clock results changed with simulation order alone")
    out["trace_sha256"] = sha256(args.trace)
    write_json(args.output, out)


if __name__ == "__main__":
    main()
