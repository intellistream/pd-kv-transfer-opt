"""CPU-only causal four-FIFO trace-conditioned timing simulator.

Recorded routes, accepted cache and durations are observations, not a new-policy
oracle. Sensitivities hold those decisions fixed and cannot prove a policy gain.
"""

import argparse
from collections import deque
import heapq
import itertools
import json
import math
from pathlib import Path
import statistics


def quantile(values, probability):
    return sorted(values)[math.ceil(len(values) * probability) - 1]


def simulate(rows, order, costs, *, mode="measured", arrival_scale=1.0,
             routing_scale=1.0, front_scale=1.0, duration_predictions=None):
    if mode not in ("measured", "calibration_given_observed_cache"):
        raise ValueError("Unknown timing model")
    if arrival_scale <= 0 or routing_scale < 0 or front_scale < 0:
        raise ValueError("Invalid timing sensitivity")
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != set(order) or len(by_id) != len(rows):
        raise ValueError("One complete unique frozen request stream required")
    positions = {key: position for position, key in enumerate(order)}
    queue, sequence, pending, active = [], itertools.count(), [], set()
    available, ends, results = [0.0] * 4, {}, {}
    router_busy = False

    def event(at, kind, key, payload=None):
        heapq.heappush(queue, (at, next(sequence), kind, key, payload))

    def duration(row, role):
        if duration_predictions is not None:
            first, complete = duration_predictions[row["id"]][role]
            if not all(math.isfinite(value) and value >= 0 for value in (first, complete)) or first > complete:
                raise ValueError("Finite first<=complete predicted durations required")
        elif mode == "measured":
            first = row[role + "_wire"]["first_seconds"]
            complete = row[role + "_wire"]["complete_seconds"]
        else:
            response = row[role]
            node = response["slot"] + (2 if role == "D" else 0)
            model = costs["service"][node]
            first = (model["intercept_seconds"] +
                     model["seconds_per_uncached_token"] * response["uncached_prompt_tokens"] +
                     model["seconds_per_rdma_byte"] * response["num_cached_tokens"] * costs["bytes_per_token"])
            complete = first + (costs["decode_seconds"][response["slot"]] if role == "D" else 0)
        return first, complete

    def dispatch(now):
        nonlocal router_busy
        if router_busy:
            return
        eligible = next((key for key in sorted(pending, key=positions.get)
                         if by_id[key]["task"] not in active), None)
        if eligible is None:
            return
        row = by_id[eligible]
        offered = row["offered_arrival_seconds"] * arrival_scale
        ready = max(offered, ends.get(row["task"], 0.0))
        assert now >= ready
        pending.remove(eligible)
        active.add(row["task"])
        router_busy = True
        event(now + row["route_seconds"] * routing_scale, "routed", eligible,
              {"offered": offered, "ready": ready, "dispatched": now})

    for key in order:
        event(by_id[key]["offered_arrival_seconds"] * arrival_scale, "arrival", key)
    while queue:
        now, _, kind, key, state = heapq.heappop(queue)
        row = by_id[key]
        timing = row["pipeline_timing"]
        if kind == "arrival":
            pending.append(key)
        elif kind == "routed":
            router_busy = False
            node = row["route"]["p"]
            start = max(now, available[node])
            front = (row["P_submission_seconds"] - timing["P_started_seconds"]) * front_scale
            _, complete = duration(row, "P")
            available[node] = start + front + complete
            state["P_queue"] = start - now
            event(available[node], "P_done", key, state)
        elif kind == "P_done":
            node = row["route"]["d"]
            start = max(now, available[node])
            front = (row["D_submission_seconds"] - timing["D_started_seconds"]) * front_scale
            first, complete = duration(row, "D")
            available[node] = start + front + complete
            state.update(first=start + front + first, D_queue=start - now)
            event(available[node], "D_done", key, state)
        elif kind == "D_done":
            active.remove(row["task"])
            ends[row["task"]] = now
            state.update(end=now, primary=state["first"] - state["offered"],
                         service_TTFT=state["first"] - state["ready"],
                         dependency=state["ready"] - state["offered"])
            results[key] = state
        dispatch(now)
    assert len(results) == len(order) and not pending and not active
    primary, ttft = ([results[key][field] for key in order] for field in ("primary", "service_TTFT"))
    observed = [by_id[key]["offered_to_first_seconds"] for key in order]
    return {"scope": "trace-conditioned simulated timing, not rerun-policy/hardware/output validation",
            "mode": mode, "arrival_scale": arrival_scale, "routing_scale": routing_scale,
            "duration_predictions_supplied": duration_predictions is not None,
            "front_copy_gate_scale": front_scale, "requests": len(order),
            "primary_p50_seconds": quantile(primary, .5), "primary_p99_seconds_descriptive": quantile(primary, .99),
            "service_TTFT_p99_seconds_descriptive": quantile(ttft, .99),
            "makespan_seconds": max(ends.values()),
            "mean_D_queue_seconds": statistics.mean(value["D_queue"] for value in results.values()),
            "dependency_wait_sum_seconds": sum(value["dependency"] for value in results.values()),
            "validation_against_original_primary_MAE_seconds": statistics.mean(abs(a - b) for a, b in zip(primary, observed)),
            "per_request": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.bundle / "measurements.jsonl").read_text(encoding="utf-8").splitlines()]
    protocol = json.loads((args.bundle / "protocol.json").read_text(encoding="utf-8"))
    costs = json.loads((args.bundle / "cost-model.json").read_text(encoding="utf-8"))["costs"]
    report = {"source": str(args.bundle), "planning_decisions_and_cache_state_frozen": True,
              "measured_mode_uses_post_hoc_service_durations": True,
              "calibrated_mode_still_conditions_on_observed_cache_and_routes": True,
              "not_a_counterfactual_algorithm_AB": True, "phases": {}}
    for phase in ("baseline", "treatment"):
        rows = [row for row in records if row["phase"] == phase]
        cases = {"measured_validation": {}, "frozen_calibration_cost": {"mode": "calibration_given_observed_cache"},
                 "zero_front_copy_gate_sensitivity": {"front_scale": 0},
                 "zero_router_wall_sensitivity": {"routing_scale": 0},
                 "double_offered_rate_fixed_decisions_sensitivity": {"arrival_scale": .5}}
        report["phases"][phase] = {name: simulate(rows, protocol["test_order"], costs, **settings)
                                   for name, settings in cases.items()}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({phase: {name: {key: value for key, value in result.items() if key != "per_request"}
                             for name, result in cases.items()} for phase, cases in report["phases"].items()}))


if __name__ == "__main__":
    main()
