"""Budget the additional real reads for 30% loss using archived fixed arrivals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
from itertools import pairwise
import json
from math import floor, isclose
from pathlib import Path

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.simulator.agent_cache import STATE_BYTES_PER_CHECKPOINT, STORE_ATTENTION_BYTES
from pd_kv_transfer_opt.simulator.agent_replay import GB, PAGE, prefill_score
from pd_kv_transfer_opt.simulator.agent_trace_gap import read_bytes
from pd_kv_transfer_opt.simulator.terminal_fixed_arrival import TARGET_LOSS, route_credits
from pd_kv_transfer_opt.simulator.terminal_rotation_pressure import Source


def merge_gets(rows):
    """Integrate wire time once where two synchronous readers overlap."""
    intervals = []
    for start, end in sorted({(row["GET_submit_s"], row["GET_done_s"])
                              for row in rows if row["GET_bytes"]}):
        if intervals and start <= intervals[-1][1]:
            prev, stop = intervals[-1]
            intervals[-1] = prev, max(stop, end)
        else:
            intervals.append((start, end))
    return intervals


def trace_cases(path, summary, manifest, by_key):
    if sha256(path) != summary["trace_sha256"]:
        raise ValueError("The request trace digest does not match the fixed-arrival summary")
    out = defaultdict(dict)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["workload"] == "matched_945":
                continue
            key = row["key"]
            group = out[row["workload"], row["order"], str(row["cap"])]
            request = by_key[key]
            source = manifest["ordered_sources"][request["slot"]]
            if (key in group or row["offered_s"] != request["root_offered_s"]
                    or row["output"] != source["output"] or row["prompt"] != source["prompt"]
                    or row["finish"] != "stop"):
                raise ValueError(f"Repeated request or changed input in the fixed trace for {key}")
            group[key] = row
    expected = {(name, order, cap) for name, orders in summary["preplanned_arrivals"].items()
                for order in orders for cap in ("100", "200")}
    if out.keys() != expected or any(rows.keys() != by_key.keys() for rows in out.values()):
        raise ValueError("The trace does not cover the full set of fixed arrivals in both orders")
    return out


def wave_routes(by_wave, rows, by_key, ranks, domains, slots_in_body_order):
    """Bound once-per-P domain visits even with an oracle domain assignment."""
    scores = [prefill_score(source.body_bytes) for source in domains]
    credits = route_credits((key, row, scores[by_key[key]["slot"]], ranks[key]) for key, row in rows.items())
    full_bytes = sorted(read_bytes(d.pages) for d in domains)
    counts, score_skew, missing_cost, repeat_upper_bytes = [], [], 0, 0
    for keys in by_wave.values():
        ordered = sorted(keys, key=lambda key: (rows[key]["P_admit_s"], ranks[key]))
        roles = Counter(rows[key]["P"] for key in ordered)
        slots = defaultdict(list)
        passes = set()
        for i, key in enumerate(ordered):
            row, slot = rows[key], by_key[key]["slot"]
            slots[row["P"], slot].append(row["GET_bytes"])
            passes.add((row["P"], slots_in_body_order[i % len(domains)]))
        first_pair_seen = set()
        for key in ordered:
            slot = by_key[key]["slot"]
            if slot not in first_pair_seen:
                left, right = credits[key]
                score_skew.append(abs(left - right) / scores[slot])
                first_pair_seen.add(slot)
        missing = abs(roles[0] - len(domains))
        missing_cost += sum(full_bytes[:missing])
        repeat_upper_bytes += sum(min(values) for values in slots.values() if len(values) == 2)
        counts.append({"P_admissions": [roles[i] for i in range(2)],
                       "unavoidable_missing_role_domain_visits": missing,
                       "missing_adjacent_pair_visits": 2 * len(domains) - len(slots),
                       "missing_two_consecutive_pass_visits_with_frozen_roles": 2 * len(domains) - len(passes)})
    return {"regular_waves": counts, "skew_over_request_score_p50_p90_p95_p99_max": [
                percentile(score_skew, q) for q in (.5, .9, .95, .99, 1.)],
            "unavoidable_missing_role_domain_visits": sum(v["unavoidable_missing_role_domain_visits"] for v in counts),
            "missing_adjacent_pair_visits": sum(v["missing_adjacent_pair_visits"] for v in counts),
            "missing_two_consecutive_pass_visits_with_frozen_roles": sum(
                v["missing_two_consecutive_pass_visits_with_frozen_roles"] for v in counts),
            "full_once_per_role_domain_GB": 2 * sum(full_bytes) * len(counts) / GB,
            "optimistic_once_per_role_domain_ceiling_GB_at_frozen_role_counts":
                (2 * sum(full_bytes) * len(counts) - missing_cost) / GB,
            "archived_extra_read_upper_bound_GB": repeat_upper_bytes / GB}


def arm_audit(rows, by_wave, archived, last_offer, boundaries, *, route_inputs):
    request_rows = list(rows.values())
    rate = archived["regular_waves"]["effective_wire_ceiling_Gbps"] / 8
    output = sum(row["output"] for row in request_rows)
    eos = max(row["EOS_s"] for row in request_rows)
    reads = sum(row["GET_bytes"] for row in request_rows) / GB
    busy = merge_gets(request_rows)
    if (output != archived["output_tokens"] or eos != archived["finish_s"]
            or not isclose(reads, archived["GET_GB"], abs_tol=1e-7)
            or not isclose(reads, sum(archived["D_to_P_GB_by_port"]), abs_tol=1e-7)
            or not isclose(sum(end - start for start, end in busy) * rate, reads, abs_tol=1e-7)):
        raise ValueError("Output, EOS, or merged GET wire time disagrees with the archived arm")
    regular = [rows[key] for keys in by_wave.values() for key in keys]
    regular_GB = sum(row["GET_bytes"] for row in regular) / GB
    regular_ready = [(row["D_KV_ready_s"], row["output"]) for row in regular]
    return {"output_tokens": output, "finish_s": eos, "throughput_tokens_s": output / eos,
            "actual_GET_GB": reads, "actual_regular_GET_GB": regular_GB,
            "actual_regular_GB_per_planned_s": regular_GB / (boundaries[-1] - boundaries[0]),
            "effective_GB_s": rate, "last_GET_to_EOS_s": eos - busy[-1][1],
            "wire_GB_after_last_offer": rate * sum(max(0, end - max(start, last_offer)) for start, end in busy),
            "wire_GB_after_last_period": rate * sum(max(0, end - max(start, boundaries[-1])) for start, end in busy),
            "not_yet_D_ready_at_last_period_output_tokens": sum(q for t, q in regular_ready if t > boundaries[-1]),
            "D_ready_output_tokens_by_fixed_window": [sum(q for t, q in regular_ready if left < t <= right)
                                                       for left, right in pairwise(boundaries)],
            "waves_all_D_KV_ready_before_next_offer": sum(
                max(rows[key]["D_KV_ready_s"] for key in keys) <=
                archived["regular_waves"]["waves"][cycle - 1]["end_of_period_s"]
                for cycle, keys in by_wave.items() if cycle < len(by_wave)),
            "max_port_Gbps_averaged_over_regular_phase":
                max(archived["regular_waves"]["mean_payload_Gbps_by_port"]),
            "max_aggregate_30s_Gbps": archived["regular_waves"]["wire_peak_30s_Gbps"],
            "routing": wave_routes(by_wave, rows, *route_inputs)}


def read_budget(low, high, wave_count, period, last_offer, max_GB_per_new_page, cold_tokens_s_per_P):
    keep = 1 - TARGET_LOSS
    target_finish = high["finish_s"] / keep
    extra_s = max(0., target_finish - low["finish_s"])
    line = low["effective_GB_s"]
    extra = extra_s * line
    actual = low["actual_regular_GET_GB"]
    target = actual + extra
    route = low["routing"]
    full = route["full_once_per_role_domain_GB"]
    optimistic = route["optimistic_once_per_role_domain_ceiling_GB_at_frozen_role_counts"]
    extra_cold_pages = extra / max_GB_per_new_page
    return {
        "target_100_finish_s_with_same_EOS": target_finish,
        "additional_completion_s": extra_s,
        "additional_GET_GB_at_unit_exposure": extra,
        "additional_GET_GB_per_wave_at_unit_exposure": extra / wave_count,
        "target_regular_GET_GB_at_unit_exposure": target, "target_GB_per_wave": target / wave_count,
        "target_GB_per_planned_s": target / (wave_count * period),
        "steady_state_30pct_min_GB_s_with_equal_read_mix": line / keep,
        "target_fraction_of_full_regular_reads": target / full,
        "minimum_exposure_even_with_full_regular_reads": extra / (full - actual) if full > actual else None,
        "target_busy_wire_debt_GB_at_last_offer_with_observed_post_GET_tail":
            line * max(0., target_finish - last_offer - low["last_GET_to_EOS_s"]),
        "target_minus_oracle_once_per_role_domain_ceiling_GB": target - optimistic,
        "target_minus_oracle_plus_archived_repeat_upper_GB": target - optimistic - route["archived_extra_read_upper_bound_GB"],
        "minimum_cold_prefill_s_if_new_domains_supply_the_extra_reads":
            extra_cold_pages * PAGE / (2 * cold_tokens_s_per_P),
    }


def construction_envelope(domains, summary):
    services = [ref["service"] for ref in summary["models"].values()]
    inventories = {(s["p_pages"], s["store_bytes"], s["checkpoint_stride"]) for s in services}
    if len(inventories) != 1:
        raise ValueError("The capacity comparison requires a common pool configuration")
    local, store_bytes, stride = inventories.pop()
    if stride != 1:
        raise ValueError("Dense Store geometry requires a checkpoint at every eligible full page")
    page_bytes = STORE_ATTENTION_BYTES + STATE_BYTES_PER_CHECKPOINT
    store_pages = store_bytes // page_bytes
    full = sum(d.pages for d in domains)
    original_two = sorted((d.pages for d in domains), reverse=True)[:2]
    unique_two = sorted((d.pages for d in {d.key: d for d in domains}.values()), reverse=True)[:2]
    retain_after_one_skipped = local + sum(unique_two)
    best_after_skip = store_pages - sum(unique_two)
    # Even the favorable long-run once-per-P ring has a joint decode/wire cadence limit.
    emit_cap = min(2 * s["slots"] / (s["decode_first_s"] + (s["slots"] - 1) * s["decode_add_s"]) for s in services)
    full_wave_GB = 2 * sum(read_bytes(d.pages) for d in domains) / GB
    rates = summary["design"]["effective_100_GB_s_by_reference"]
    upper_period = (1 - TARGET_LOSS) * full_wave_GB / max(rates)
    output_per_copy = sum(d.output for d in domains)
    return {"P_complete_pages": local, "dense_Store_bytes_per_complete_page": page_bytes,
            "store_bytes": store_bytes, "dense_Store_max_complete_pages": store_pages,
            "current_complete_pages": full, "unused_Store_GiB_in_current_ring": (store_bytes - full * page_bytes) / 2**30,
            "minimum_new_domain_pages": min(d.pages for d in domains),
            "worst_intervening_pages_in_full_current_ring": full - original_two[0],
            "maximum_worst_intervening_pages_after_one_skip_with_all_seven_sources": best_after_skip,
            "minimum_Store_GiB_for_one_skip_tolerant_full_ring_with_all_seven_sources":
                retain_after_one_skipped * page_bytes / 2**30,
            "minimum_Store_GiB_for_one_skip_tolerant_full_ring_with_two_longest_original_domains":
                (local + sum(original_two)) * page_bytes / 2**30,
            "optimistic_GB_per_new_page_with_one_restore_on_each_P":
                2 * max(read_bytes(d.pages) / d.pages for d in domains) / GB,
            "favorable_regular_period_s_for_full_once_per_role_domain_reads": [
                full_wave_GB / (2 * min(rates)), upper_period],
            "smaller_peak_D_output_tokens_s": emit_cap, "one_copy_output_tokens_per_wave": output_per_copy,
            "uniform_copies_upper_bound_by_D_at_current_period":
                floor(emit_cap * summary["design"]["regular_window_period_s"] / output_per_copy),
            "uniform_copies_upper_bound_by_D_even_at_optimistic_steady_state_period":
                floor(emit_cap * upper_period / output_per_copy)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = read_json(args.fixed / "summary.json")
    manifest_path = args.fixed / "workload.json"
    manifest = read_json(manifest_path)
    if sha256(manifest_path) != summary["workload_sha256"]:
        raise ValueError("Fixed workload digest mismatch")
    by_key = {f"{row['name']}:0": row for row in manifest["requests"]}
    ranks = {key: i for i, key in enumerate(by_key)}
    cases = trace_cases(args.trace, summary, manifest, by_key)
    domains = [Source(**source) for source in manifest["ordered_sources"]]
    by_wave = defaultdict(list)
    for key, row in by_key.items():
        if row["phase"] == "restore" and row["cycle"] > 0:
            by_wave[row["cycle"]].append(key)
    design = summary["design"]
    start, period = design["regular_start_s"], design["regular_window_period_s"]
    regular_end = start + period * len(by_wave)
    boundaries = [start + 60 * i for i in range(1 + floor((regular_end - start) / 60)) if start + 60 * i < regular_end]
    boundaries.append(regular_end)
    last_offer = max(row["root_offered_s"] for row in by_key.values())
    route_inputs = (by_key, ranks, domains, design["domain_slots_ordered_by_body"])
    envelope = construction_envelope(domains, summary)
    pairs = {}
    for name, orders in summary["preplanned_arrivals"].items():
        for order, case in orders.items():
            arms = {cap: arm_audit(cases[name, order, cap], by_wave, case["arms"][cap],
                                   last_offer, boundaries, route_inputs=route_inputs) for cap in ("100", "200")}
            budget = read_budget(arms["100"], arms["200"], len(by_wave), period, last_offer,
                                 envelope["optimistic_GB_per_new_page_with_one_restore_on_each_P"],
                                 summary["models"][order]["service"]["cold_tokens_s"])
            pairs[name, order] = {"100_loss_relative_to_200": case["100_loss_relative_to_200"],
                                  "arms": arms, "read_budget": budget}
    write_json(args.output, {
        "baseline_revision": summary["baseline_revision"], "fixed_arrival_summary_sha256": sha256(args.fixed / "summary.json"),
        "fixed_workload_sha256": summary["workload_sha256"], "trace_sha256": summary["trace_sha256"],
        "target_100_loss_relative_to_200": TARGET_LOSS, "regular_waves": len(by_wave),
        "regular_period_s": period, "last_offer_s": last_offer, "D_ready_windows_s": boundaries,
        "construction_envelope": envelope,
        "cases": {f"{name}/{order}": value for (name, order), value in pairs.items()},
        "scope": "Unit-exposure budgets hold the archived high-cap finish and non-wire costs fixed. "
        "Route visit bounds condition on the archived allocation counts rather than claiming an invariant under new clocks. "
        "Ring robustness calculations count released complete pages, without an allowance for temporary HBM leases. "
        "The optimistic guard and cadence envelope assumes at most one distinct useful restore per (P, domain, wave).",
    })


if __name__ == "__main__":
    main()
