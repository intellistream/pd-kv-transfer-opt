"""Verify the CPU replay on r69/r78 and size a sustained native GET wave."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, read_jsonl, sha256, write_json
from pd_kv_transfer_opt.profiling.screen_bidirectional_workload import peak_demand
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS, ATTENTION_BYTES, STATE_BYTES
from pd_kv_transfer_opt.simulator.agent_replay import GB, STORE_BYTES, Replay
from pd_kv_transfer_opt.simulator.decode_supply import simulate_decode
from pd_kv_transfer_opt.simulator.freeinference_ab import calibration, report, supply_pair
from pd_kv_transfer_opt.simulator.rdma_replay import compare_pair, egress_epochs
from pd_kv_transfer_opt.simulator.terminal_supply import native
from pd_kv_transfer_opt.workloads.freeinference import Session, Turn


ANCHORS = ("2026-11-22-r69-terminal-source-restore-short-guard", "2026-11-22-r78-terminal-source-restore-stable-tail")
R69_METHODS = "2026-11-22-r69-terminal-source-restore-short-guard-methods"


def ledger_pages(path):
    by_task = defaultdict(list)
    for row in read_jsonl(path):
        by_task[row["task"]].append(row)
    pages, next_id = {}, 1
    for rows in by_task.values():
        previous = ()
        for row in sorted(rows, key=lambda r: r["step_id"]):
            shared = row["source_prompt_lcp_to_previous"] // ATTENTION_BLOCK_TOKENS
            length = row["prompt_tokens"] // ATTENTION_BLOCK_TOKENS
            common = min(length, shared, len(previous))
            keys = (*previous[:common], *range(next_id, next_id + length - common))
            next_id += length - common
            pages[row["id"]] = (row, keys)
            previous = keys
    return pages


def archived_plan(workload, reference, pages):
    rows = {row["id"]: row for row in workload["request_sources"]}
    bodies = workload["arrival_order_screen"]["wire_body_bytes"]
    origin = min(row["offered_s"] for row in rows.values())
    roots, planned, sizes, map_keys, sessions = {}, {}, {}, {}, []
    for pair in workload["selection"]["pairs"]:
        keys = (pair["root"], pair["successor"])
        turns = []
        roots[keys[0]] = rows[keys[0]]["offered_s"] - origin
        for step, key in enumerate(keys):
            spec, observed = rows[key], reference[key]
            ledger, hashes = pages[key]
            if (spec["prompt_tokens"], spec["prompt_sha256"]) != (
                    ledger["prompt_tokens"], ledger["token_ids_uint32le_sha256"]):
                raise ValueError(f"Archive and source prompt disagree for {key}")
            if observed["finish_reason"] != "stop":
                raise ValueError(f"Archive needs natural EOS for {key}")
            adjacent = pair["checkpoint_blocks"] if step else 0
            turns.append(Turn(round(1000 * spec["tool_gap_s"]), spec["prompt_tokens"], observed["output_tokens"],
                              hashes, adjacent, "stop"))
            sim_key = f"{keys[0]}:{step}"
            planned[sim_key] = spec["offered_s"] - origin
            sizes[sim_key] = bodies[key]
            map_keys[sim_key] = key
        sessions.append(Session(keys[0], round(1000 * roots[keys[0]]), 0, 0, tuple(turns)))
    return sessions, roots, planned, sizes, map_keys


def early_deliveries(rows, wave_start, deps):
    ordered = [r for key, r in rows.items() if deps[key]["predecessor"]]
    return {"successor_D_first_in_first_10s": sum(r["first_s"] <= wave_start + 10 for r in ordered),
            "successor_first_D_s": min(r["first_s"] for r in ordered)}


def successor_egress(arm):
    start = min(row["offered_s"] for row in arm[1].values() if row["predecessor"])
    return start, sum(max(0, end - max(begin, start)) * sum(rates) / 8
                      for begin, end, rates in egress_epochs(arm))


def settled_grid(anchors, target=.30):
    comparable = []
    for name, cases in anchors.items():
        for order, row in cases.items():
            measure = row["conditional"]["hardware"]
            high = measure["200"]
            low_time = measure["100"]["duration_s"]
            seconds = high["duration_s"]
            rate = row["effective_100_GB_s"]
            reads = row["high_successor_egress_GB"]
            low_reads = row["low_successor_egress_GB"]
            actual = row["conditional"].get("measured_P_only_fixed_reference_output_and_D_finish_gap_s")
            ready = row["wave_start_s"]
            cases = {"T200_s": seconds, "observed_100_s": low_time, "effective_100_GB_s": rate,
                     "observed_wave_reads_100_GB": low_reads, "observed_wave_reads_200_GB": reads,
                     "observed_200_post_P_tail_s": seconds - high["last_P_reply_s"],
                     "observed_200_first_successor_offer_s": ready,
                     "measured_P_only_same_EOS_gap_s": actual}
            if actual is not None:
                cases["exposed_fraction_of_ideal_fixed_read_speedup"] = actual / (reads / (2 * rate))
                comparable.append(cases)
            row["size_receipt"] = cases
    exposure = min(x["exposed_fraction_of_ideal_fixed_read_speedup"] for x in comparable)
    best_exposure = max(x["exposed_fraction_of_ideal_fixed_read_speedup"] for x in comparable)
    rates = [row["effective_100_GB_s"] for cases in anchors.values() for row in cases.values()]
    errors = [abs(row["conditional"]["short_burst_sensitivity"]["finish_gap_s"]
                  - row["conditional"]["measured_P_only_fixed_reference_output_and_D_finish_gap_s"])
              for cases in anchors.values() for row in cases.values()
              if "measured_P_only_fixed_reference_output_and_D_finish_gap_s" in row["conditional"]]
    error = max(errors)
    fixed_high_scenarios = []
    for seconds in (60, 70, 80, 90, 120):
        delta = target / (1 - target) * seconds
        fixed_high_scenarios.append({"compatible_T200_s": seconds,
                                    "needed_finish_gap_s": delta,
                                    "ideal_needed_GB": [2 * rate * delta for rate in (min(rates), max(rates))],
                                    "observed_exposure_and_error_GB": [
                                        2 * rate * (delta + error) /
                                        case["exposed_fraction_of_ideal_fixed_read_speedup"]
                                        for rate in (min(rates), max(rates)) for case in comparable]})
    tail_grid = {}
    for cold in (20, 44.5, 56):
        tail_grid[str(cold)] = {
            str(work): {str(theta): [((theta / (2 * target)) - 1) * work / rate - cold
                                    for rate in (max(rates), min(rates))]
                        for theta in (1.0, .8, exposure)}
            for work in (600, 800, 1000, 1200)}
    return {"target_loss_relative_to_200": target, "media": "D egress payload GB; GB is decimal",
            "effective_100_GB_s_range": [min(rates), max(rates)],
            "steady_30s_60s_GB": {str(t): [t * min(rates), t * max(rates)] for t in (30, 60, 120)},
            "max_abs_aligned_short_burst_error_s": error,
            "exposure_fraction_range": [exposure, best_exposure],
            "fixed_high_duration": fixed_high_scenarios,
            "serial_cold_wire_tail": {"relation": "T100 = cold + W/R100 + tail; delta <= exposure*W/(2*R100)",
                                      "maximum_tail_seconds_by_cold_seconds_and_GB_and_exposure": tail_grid},
            "ideal_exposed_fraction_of_T100_needed": 2 * target,
            "maximum_100_Gbps_payload_equivalent_pages_per_30s": [
                30 * r * GB / ATTENTION_BYTES for r in (min(rates), max(rates))]}


def verify(results, ledger):
    pages = ledger_pages(ledger)
    models = calibration(results)
    anchors, digests = {}, {}
    for name in ANCHORS:
        path = results / name
        manifest, digest, recorded = native(path)
        digests[name] = digest
        anchors[name] = {}
        for (order, archive), (service, receipt) in zip(
                (("forward", "b1"), ("reverse", "b2")), models.values(), strict=True):
            archives = {cap: recorded[f"{archive}-cap{cap}"] for cap in (100, 200)}
            start, high_wave_GB = successor_egress(archives[200])
            _, low_wave_GB = successor_egress(archives[100])
            reference = archives[200][1]
            decode = receipt["decode_fit"]
            robust = compare_pair(archives[100], archives[200], decode, service.slots, (), include_requests=True)
            model = robust["short_burst_sensitivity"]
            supply = {str(cap): {"observed": early_deliveries(archives[cap][1], start, reference),
                                 "conditional": early_deliveries(model[str(cap)]["per_request"], start, reference)}
                      for cap in (100, 200)}
            absolute_D = {}
            for cap, (_, requests, _) in archives.items():
                d_fit = simulate_decode(requests, decode)
                actual = max(row["end_s"] for row in requests.values())
                absolute_D[str(cap)] = {"predicted_finish_s": d_fit["finish_s"],
                                       "observed_finish_s": actual, "error_s": d_fit["finish_s"] - actual}

            sessions, offers, planned, sizes, ids = archived_plan(manifest["workload"], reference, pages)
            runs = {cap: Replay(sessions, offers, service, cap, turn_offers=planned,
                                observed_body_bytes=sizes).run() for cap in (100, 200)}
            loop = {}
            for cap, run in runs.items():
                measured = archives[cap][1]
                loop[str(cap)] = {**report(run),
                                  "same_P_as_observed": sum(measured[ids[key]]["P"] == f"P{j.row['P']}"
                                                            for key, j in run.requests.items()),
                                  "same_D_as_observed": sum(measured[ids[key]]["D"] == f"D{j.row['D']}"
                                                            for key, j in run.requests.items())}
            for prediction in ("prediction", "short_burst_sensitivity"):
                for cap in ("100", "200"):
                    robust[prediction][cap].pop("per_request")

            anchors[name][order] = {
                "calibration": receipt, "conditional": robust, "first_D_10_s": supply,
                "D_observed_inputs_check": absolute_D, "closed_loop": loop,
                "closed_loop_loss_fraction": 1 - runs[200].now / runs[100].now,
                "closed_loop_supply_pair": supply_pair(runs[100], runs[200]),
                "effective_100_GB_s": runs[100].GB_s,
                "wave_start_s": start, "high_successor_egress_GB": high_wave_GB,
                "low_successor_egress_GB": low_wave_GB}
    return {"anchor_trace_sha256": digests, "terminal_ledger_sha256": sha256(ledger),
            "scope": "Conditional replay supplies archived HBM/Store outcomes through P HTTP and rail telemetry. "
            "Closed-loop replay supplies verified prompt lengths, within-task lower-bound prefix keys, observed "
            "serialized request-body sizes, planned arrivals, and the high arm's natural output lengths. "
            "All closed-loop routes, cache events and bytes are computed. The release tools and stage costs are unchanged.",
            "calibration": anchors, "pressure_target": settled_grid(anchors)}


def put_windows(results):
    location = results / R69_METHODS
    methods = location / "native-method-events.jsonl.gz"
    digest = sha256(methods)
    if digest != read_json(location / "native-method-summary.json")["method_events_sha256"]:
        raise ValueError("r69 Store method archive digest mismatch")
    put = defaultdict(lambda: defaultdict(list))
    with gzip.open(methods, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            event = row["data"]
            if event["type"] == "store_end" and event["operation"] == "batch_put_from_multi_buffers":
                side = event["instance"][0]
                put[row["case"]][side].append((event["monotonic_ns"] / GB, event["instance"],
                                               event["payload_bytes"], ""))
    cases = {}
    for case, sides in put.items():
        if sides.keys() != {"P", "D"}:
            raise ValueError("Expected PUT method data on both physical hosts")
        peaks = {side: {str(t): peak_demand(sorted(rows), t)["Gbps"] * t / 8 for t in (30, 60)}
                 for side, rows in sides.items()}
        cases[case] = {"total_put_descriptor_GB_by_host": {
            side: sum(row[2] for row in rows) / GB for side, rows in sides.items()},
            "peak_put_descriptor_GB_by_host": peaks,
            "peak_put_descriptor_GB": {str(t): sum(v[str(t)] for v in peaks.values()) for t in (30, 60)}}
    return {"trace_sha256": digest, "by_case": cases,
            "clocks": "Separate host monotonic clocks. Sum independent host maxima to upper bound a common window."}


def feasibility(results, receipt):
    sizing = receipt["pressure_target"]
    loss = sizing["target_loss_relative_to_200"]
    min_R, max_R = rates = sizing["effective_100_GB_s_range"]
    lower, upper = sizing["exposure_fraction_range"]
    exposure = sorted({1.0, lower, upper, *(r["size_receipt"]["exposed_fraction_of_ideal_fixed_read_speedup"]
                         for order in receipt["calibration"].values() for r in order.values()
                         if "exposed_fraction_of_ideal_fixed_read_speedup" in r["size_receipt"])})
    # Quantities count scheduled, physically required GET payload. A single
    # rank-duplicated Attention load may transfer twice its unique Store bytes.
    capacity_GB = STORE_BYTES / GB
    put = put_windows(results)
    best_put_60 = max(r["peak_put_descriptor_GB"]["60"] for r in put["by_case"].values())
    work = (128 * ATTENTION_BYTES + STATE_BYTES) / GB
    services = [s for s, _ in calibration(results).values()]
    peak_D = [2 * s.slots / (s.decode_first_s + (s.slots - 1) * s.decode_add_s)
              for s in services]
    p_scenarios = [s.cold_tokens_s for s in services]
    cold_range = [min(p_scenarios), max(p_scenarios)]
    min_high_demand, max_high_demand = [r / (1 - loss) for r in rates]
    high_load_at_target = 1 / (2 * (1 - loss))
    tail = {
        str(cold): {str(volume): {str(theta): [
            (theta / (2 * loss) - 1) * volume / rate - cold for rate in (max_R, min_R)]
            for theta in exposure}
            for volume in (600, 800, 1000, 1200, 1500)}
        for cold in (20, 44.5, 56)}
    min_volume = {}
    for cold, drain in ((20, 5), (44.5, 5), (44.5, 12.7), (56, 5)):
        min_volume[f"{cold}+{drain}"] = {
            str(theta): [(cold + drain) * rate / (theta / (2 * loss) - 1) for rate in rates]
            if theta > 2 * loss else None for theta in exposure}
    example_R, example_W, example_C, example_Z = 9.4, 1250, 44.5, 5
    link_time = example_W / example_R
    example_T100 = example_C + link_time + example_Z
    example_T200 = example_T100 - upper * link_time / 2
    example_A = example_W / (example_T200 - example_C - example_Z)

    return {
        "throughput_loss_definition": "1 - X100/X200 with the same output-token work",
        "target": loss,
        "effective_rates_GB_s": {"100": rates, "200": [2 * r for r in rates]},
        "sustained_pressure_sizing": {
            "minimum_ready_GB_s_to_saturate_100": rates,
            "minimum_ready_GB_s_for_30pct_steady_loss_under_unsaturated_200": [min_high_demand, max_high_demand],
            "minimum_ready_payload_GB": {str(t): {"100_merely_saturated": [t * r for r in rates],
                                                   "30pct_steady_loss": [t * min_high_demand, t * max_high_demand]}
                                         for t in (30, 60, 120)},
            "logic": "In the steady fixed-work regime, X100/X200 = R100 / offered_payload_rate "
            "when 100 throttles delivery, 200 has spare link capacity, and no other stage sets the low rate."},
        "required_100_wire_share_of_entire_round": {
            str(theta): 2 * loss / theta for theta in exposure},
        "serial_envelope": {"max_tail_s_at_each_preparation_and_payload": tail,
                            "needed_payload_GB_at_each_preparation_plus_tail": min_volume},
        "arithmetic_feasibility_example": {
            "assumed_effective_100_GB_s": example_R, "assumed_read_GB": example_W,
            "assumed_preparation_s": example_C, "assumed_drain_s": example_Z, "assumed_exposure": upper,
            "T100_s": example_T100, "T200_s": example_T200, "100_loss_fraction": 1 - example_T200 / example_T100,
            "required_200_active_wave_GB_s": example_A, "available_200_link_GB_s": 2 * example_R,
            "D_wave_output_tokens_s_if_300_outputs_per_full_128Ki_token_GET_equivalent": 300 * example_A / work},
        "per_full_128Ki_token_GET_equivalent": {
            "physical_GB_including_3_checkpoint_groups": work,
            "100_wire_limited_GETs_per_s": [r / work for r in rates],
            "minimum_200_GETs_per_s_for_steady_30pct": [min_high_demand / work, max_high_demand / work],
            "D_calibrated_peak_output_tokens_s": [min(peak_D), max(peak_D)],
            "maximum_mean_output_tokens_per_GET_equivalent_at_D_peak": [
                min(peak_D) * work / max_high_demand, max(peak_D) * work / min_high_demand],
            "maximum_mean_output_tokens_per_GET_equivalent_at_300_tokens_s": [
                300 * work / max_high_demand, 300 * work / min_high_demand],
            "P_cold_tokens_s_per_instance": cold_range,
            "maximum_mean_cold_input_tokens_per_GET_equivalent_by_concurrent_blocked_P_workers": {
                str(k): [(2 - k * high_load_at_target) * cold_range[0] * work / max_high_demand,
                         (2 - k * high_load_at_target) * cold_range[1] * work / min_high_demand]
                for k in (1, 2)}},
        "store_work_conservation": {
            "physical_capacity_GB": capacity_GB,
            "r69_put_descriptor_windows": put,
            "upper_one_logical_read_per_residency_GB_with_full_initial_store_and_max_recorded_60s_put": (
                2 * (capacity_GB + best_put_60)),
            "minimum_60s_new_unique_put_GB_if_a_store_residency_is_read_only_once": {
                "100_merely_saturated": [max(0., 30 * r - capacity_GB) for r in rates],
                "30pct_steady_loss": [max(0., 30 * r / (1 - loss) - capacity_GB) for r in rates]},
            "scope": "An optimistic bound using at most 2x wire duplication per unique exported Attention "
            "object for one logical recovery. Store PUT descriptors are an upper bound on new unique "
            "successful publication. Sustained re-reads of resident pages may exceed the single-load budget."},
        "scope": "CPU calculation from qualified historical bandwidth comparisons. Candidate workload "
        "feasibility depends on keeping the indicated native Store hit and P miss sequence, critical-path "
        "exposure, the 200-arm D service, and its natural tail at the same time."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feasibility-output", type=Path)
    args = parser.parse_args()
    result = verify(args.results, args.ledger)
    write_json(args.output, result)
    if args.feasibility_output:
        write_json(args.feasibility_output, {"calibration_sha256": sha256(args.output),
                                            **feasibility(args.results, result)})


if __name__ == "__main__":
    main()
