"""Screen cold-prefill order on native natural runs and fixed-work diagnostics."""

import argparse
from bisect import bisect_left
from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
from itertools import accumulate, combinations, product
from pathlib import Path

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.experiments.prefill_order import GATE_S, MAX_PROMOTIONS, P_SECONDS_PER_TOKEN, select_promotions
from pd_kv_transfer_opt.profiling.analyze_native_prefill_path import batch_and_request_index, decode_progress
from pd_kv_transfer_opt.profiling.compare_session_runs import ARCHIVES
from pd_kv_transfer_opt.profiling.profile_closed_sessions import (
    NS, check_observed, phase_times, read_trace, replay, requests_for,
)


P_ROLES = ("P0", "P1")
D_ROLES = ("D0", "D1")
D_SLOTS = 16


@dataclass
class PrefillWindow:
    order: tuple
    arrival_order: tuple
    running_tokens_at_gate: int
    work: dict
    slot_tokens: list
    slot_ends_ns: list
    serial_ends_ns: dict
    serial_fit_error_s: float

    def completion_slots(self, promoted):
        order = (*promoted, *(key for key in self.order if key not in promoted))
        return [self.slot_ends_ns[bisect_left(self.slot_tokens, done)]
                for done in accumulate(self.work[key] for key in order)], order

    def shifts(self, promoted):
        ends, order = self.completion_slots(promoted)
        return {key: (end - self.serial_ends_ns[key]) / NS for key, end in zip(order, ends, strict=True)}


def get_windows(events, profiles):
    windows, queue_wait = {}, {}
    first_admission = {
        role: min(event["arrival_ns"] for event in events
                  if event["instance"] == role and event["type"] == "p_chunk_arrival")
        for role in P_ROLES
    }
    gate = max(first_admission.values()) + GATE_S * NS
    for role in P_ROLES:
        arrivals, batches, by_request, _, _, errors = batch_and_request_index(
            [event for event in events if event["instance"] == role])
        if errors:
            raise ValueError(f"Invalid {role} P batch events: {errors[:3]}")
        for key, slices in by_request.items():
            queue_wait[key] = (slices[0][0]["dispatch_ns"] - arrivals[key]["arrival_ns"]) / NS
        order = tuple(sorted(
            (key for key, item in arrivals.items() if profiles[key]["predecessor"] is None
             and item["arrival_ns"] <= gate <= by_request[key][0][0]["dispatch_ns"]),
            key=lambda key: by_request[key][0][0]["dispatch_ns"]))
        arrival_order = tuple(sorted(order, key=lambda key: arrivals[key]["arrival_ns"]))
        unfinished = 0
        for key, slices in by_request.items():
            if slices[0][0]["dispatch_ns"] < gate <= slices[-1][0]["dispatch_ns"]:
                accounted = sum(count for batch, _, count in slices if batch["dispatch_ns"] < gate)
                unfinished += max(0, arrivals[key]["prompt_tokens"] - accounted)
        work = {key: sum(count for _, _, count in by_request[key]) for key in order}
        if any(by_request[key][0][1] != 0 for key in order):
            raise ValueError(f"Candidate in {role} already had a reusable prefix")
        slots = []
        for batch in batches:
            used = sum(count for key, _, count in batch["rows"] if key in work)
            if used:
                if used > batch["budget"]:
                    raise ValueError(f"Budget exceeded in {role}")
                slots.append((used, batch["dispatch_ns"]))
        window = PrefillWindow(order, arrival_order, unfinished, work, list(accumulate(n for n, _ in slots)),
                               [t for _, t in slots], {}, 0)
        times, _ = window.completion_slots(())
        window.serial_ends_ns = dict(zip(order, times, strict=True))
        window.serial_fit_error_s = max(
            abs(window.serial_ends_ns[key] - by_request[key][-1][0]["dispatch_ns"]) / NS for key in order)
        windows[role] = window
    return windows, queue_wait


def actions(windows):
    order = [key for role in P_ROLES for key in windows[role].arrival_order]
    yield ()
    for size in range(1, MAX_PROMOTIONS + 1):
        yield from combinations(order, size)


def apply_action(windows, choice):
    shifts = {}
    for window in windows.values():
        shifts.update(window.shifts(tuple(key for key in choice if key in window.work)))
    return shifts


def throughput(profiles, ends):
    duration = max(ends.values()) - min(row["offered_s"] for row in profiles.values())
    count = sum(row["output_tokens"] for row in profiles.values())
    return {"seconds": duration, "output_tokens": count, "tokens_s": count / duration,
            "last_request": max(ends, key=ends.get),
            "D_end_s": {role: max(ends[k] for k, row in profiles.items() if row["D"] == role) for role in D_ROLES}}


def no_prefill_residence(profiles, requests):
    costs = phase_times(profiles)
    changed = {key: {**phases, "P_HTTP": 0.0} if key in requests else phases
               for key, phases in costs.items()}
    return throughput(profiles, replay(profiles, changed)[1])


def fixed_service(profiles, shifts):
    changed = {key: {**phases, "P_HTTP": phases["P_HTTP"] + shifts.get(key, 0)}
               for key, phases in phase_times(profiles).items()}
    if any(phases["P_HTTP"] <= 0 for phases in changed.values()):
        raise ValueError("Promotion crossed a request's P admission")
    return throughput(profiles, replay(profiles, changed)[1])


def model_decode(profiles, specs, shifts, d_ready_delay, costs, queued_p=None):
    """Recompute D slot queues and concurrent output using an empirical step curve."""
    events, waiting = [], {role: deque() for role in D_ROLES}
    running = {role: {} for role in D_ROLES}
    remaining = {key: float(row["output_tokens"]) for key, row in profiles.items()}
    successor = {row["predecessor"]: key for key, row in profiles.items() if row["predecessor"]}
    if len(successor) != sum(row["predecessor"] is not None for row in profiles.values()):
        raise ValueError("Expected serial sessions")

    def offer(key, release):
        row = profiles[key]
        ready = max(row["offered_s"], release)
        p_done = ready + row["phases_s"]["ready_to_P"] + row["phases_s"]["P_HTTP"] + shifts.get(key, 0)
        if queued_p is not None and row["predecessor"] and queued_p[key] >= 0.5:
            # Leave the old first dispatch slot in place for a delayed successor.
            p_done = max(p_done, row["P_end_s"])
        heapq.heappush(events, (p_done + d_ready_delay[key], key))

    for key, row in profiles.items():
        if row["predecessor"] is None:
            offer(key, row["offered_s"])
    t, started, finished = 0.0, {}, {}
    while events or any(running.values()):
        future = events[0][0] if events else float("inf")
        step = {}
        for role, batch in running.items():
            if batch:
                context = max(specs[key]["prompt_tokens"] for key in batch)
                step[role] = costs[len(batch), context >= 65536]
                future = min(future, t + min(remaining[key] for key in batch) * step[role])
        for role, period in step.items():
            batch = running[role]
            for key in tuple(batch):
                remaining[key] -= (future - t) / period
                if remaining[key] < 1e-7:
                    del batch[key]
                    finished[key] = future
                    if key in successor:
                        next_key = successor[key]
                        offer(next_key, future + profiles[next_key]["tool_gap_s"])
        t = future
        while events and events[0][0] <= t:
            _, key = heapq.heappop(events)
            waiting[profiles[key]["D"]].append(key)
        for role, batch in running.items():
            while waiting[role] and len(batch) < D_SLOTS:
                key = waiting[role].popleft()
                started[key] = t
                batch[key] = None
    if len(finished) != len(profiles):
        raise ValueError("Incomplete D replay")
    result = throughput(profiles, finished)
    result["D_slot_start_s"] = {key: start for key, start in started.items() if profiles[key]["predecessor"] is None}
    return result


def decode_costs(path):
    bins = read_json(path)["step_cost_by_requests_and_history"]
    result = {}
    singleton_ratio = (bins["1/32K-64K"]["stage_cost_ms"]["rank0_worker_gap_ms"]["p50"]
                       / bins["1/64K+"]["stage_cost_ms"]["rank0_worker_gap_ms"]["p50"])
    for width in range(1, D_SLOTS + 1):
        long = bins[f"{width}/64K+"]["stage_cost_ms"]["rank0_worker_gap_ms"]["p50"]
        short = bins.get(f"{width}/32K-64K")
        result[width, True] = long / 1000
        result[width, False] = short["stage_cost_ms"]["rank0_worker_gap_ms"]["p50"] / 1000 if short else long * singleton_ratio / 1000
    return result


def suffix_samples(profiles, specs):
    terminal = defaultdict(float)
    for row in profiles.values():
        terminal[row["session_id"]] = max(terminal[row["session_id"]], row["end_s"])
    return [(row["session_id"], specs[key]["prompt_tokens"], terminal[row["session_id"]] - row["P_end_s"])
            for key, row in profiles.items() if row["predecessor"] is None]


def predict_suffix(samples, session, prompt):
    others = [(length, seconds) for known, length, seconds in samples if known != session]
    neighbors = sorted(others, key=lambda item: abs(item[0] - prompt))[:12]
    return percentile([seconds for _, seconds in neighbors], 0.75)


def candidate_inventories(windows, candidates, profiles, before):
    records = {}
    for choice in candidates:
        shifts = apply_action(windows, choice)
        identity = tuple(shifts[key] for key in sorted(shifts))
        if identity not in records or len(choice) < len(records[identity][0]):
            records[identity] = choice, fixed_service(profiles, shifts)
    best_choice, _ = min(records.values(), key=lambda item: (item[1]["seconds"], len(item[0])))
    gains = [100 * (after["tokens_s"] / before["tokens_s"] - 1) for _, after in records.values()]
    single = min((item for item in records.values() if len(item[0]) <= 1), key=lambda item: item[1]["seconds"])
    return best_choice, {
        "distinct_slot_outcomes_including_unchanged": len(gains),
        "at_least_5pct": sum(gain >= 5 for gain in gains),
        "at_least_10pct": sum(gain >= 10 for gain in gains),
        "negative": sum(gain < -1e-7 for gain in gains),
        "best_with_at_most_one_promotion": {"promoted": single[0], "fixed_D_residence": single[1]},
    }


def route_sensitivity(profiles, specs, best, shifts, ready_delay, steps, queued_p):
    scenarios = []
    for routes in product(D_ROLES, repeat=len(best)):
        placement = dict(zip(best, routes, strict=True))
        scenario = {key: {**row, "D": placement[key]} if key in placement else row for key, row in profiles.items()}
        base = model_decode(scenario, specs, {}, ready_delay, steps, queued_p)
        revised = model_decode(scenario, specs, shifts, ready_delay, steps, queued_p)
        scenarios.append({"placement": placement, "original_order_model": base, "promoted_order_model": revised,
                          "gain_pct": (revised["tokens_s"] / base["tokens_s"] - 1) * 100})
    return scenarios


def evaluate_arm(raw, profiles, specs, original_specs, origin, samples, steps):
    windows, queued_p = get_windows(raw["p_native"], profiles)
    original = fixed_service(profiles, {})
    keys = [key for window in windows.values() for key in window.arrival_order]
    suffix = {key: predict_suffix(samples, profiles[key]["session_id"], specs[key]["prompt_tokens"]) for key in keys}
    d_stages, _ = decode_progress(raw, profiles, origin)
    d_ready_delay = {key: row["D_KV_ready_s"] - profiles[key]["P_end_s"] for key, row in d_stages.items()}
    candidates = list(actions(windows))
    by_earliest = {key: profiles[key]["P_start_s"] for key in keys}

    def rank_by(value, reverse=False):
        first = set(sorted(keys, key=value, reverse=reverse)[:MAX_PROMOTIONS])
        return tuple(key for key in keys if key in first)

    def shortest_first(window_size=None):
        ordered = []
        for window in windows.values():
            waiting = window.arrival_order
            size = window_size or len(waiting)
            for start in range(0, len(waiting), size):
                ordered.extend(sorted(waiting[start:start + size], key=lambda key: specs[key]["prompt_tokens"]))
        return tuple(ordered)

    final_by_session = defaultdict(float)
    for row in profiles.values():
        final_by_session[row["session_id"]] = max(final_by_session[row["session_id"]], row["end_s"])
    actual_suffix = {key: final_by_session[profiles[key]["session_id"]] - profiles[key]["P_end_s"] for key in keys}
    best, inventory = candidate_inventories(windows, candidates, profiles, original)
    snapshots = {role: {
        "running_tokens": window.running_tokens_at_gate,
        "waiting": [{"id": key, "prompt_tokens": original_specs[key]["prompt_tokens"], "suffix_s": suffix[key]}
                    for key in window.arrival_order],
    } for role, window in windows.items()}
    methods = {
        "unchanged_observed_order": (),
        "oldest_eligible": rank_by(by_earliest.get),
        "shortest_prefill": rank_by(lambda key: specs[key]["prompt_tokens"]),
        "shortest_prefill_entire_queue": shortest_first(),
        "shortest_prefill_windows_of_four": shortest_first(4),
        "shortest_prefill_windows_of_eight": shortest_first(8),
        "largest_request_budget": rank_by(lambda key: original_specs[key]["max_tokens"], reverse=True),
        "largest_predicted_suffix": rank_by(suffix.get, reverse=True),
        "longest_actual_suffix_oracle": rank_by(actual_suffix.get, reverse=True),
        "predicted_session_end": select_promotions(snapshots),
        "best_fixed_service_oracle": best,
    }
    values = {}
    for method, choice in methods.items():
        shifts = apply_action(windows, choice)
        values[method] = {
            "promoted": choice,
            "max_prefill_delay_s": max(shifts.values()),
            "prefill_end_shifts_s": shifts,
            "fixed_D_residence": fixed_service(profiles, shifts),
            "rebatch_D": model_decode(profiles, specs, shifts, d_ready_delay, steps),
            "rebatch_D_hold_queued_P": model_decode(profiles, specs, shifts, d_ready_delay, steps, queued_p),
        }
    return {"observed": original, "candidate_count": len(candidates),
            "queue_by_P": {role: window.order for role, window in windows.items()},
            "P_serial_max_fit_error_s": {role: window.serial_fit_error_s for role, window in windows.items()},
            "predicted_suffix_s": suffix, "actual_suffix_s": actual_suffix, "actions": inventory,
            "best_order_promoted_root_route_sensitivity": route_sensitivity(
                profiles, specs, best, apply_action(windows, best), d_ready_delay, steps, queued_p),
            "methods": values}


def analyze(results):
    source_digests, natural, detailed = {}, {}, []
    baseline_profiles = {}
    samples = []
    original_specs = None
    r34 = results / "2026-11-20-r34-native-decode-path" / "summary.json"
    steps = decode_costs(r34)
    source_digests["r34_step_summary"] = sha256(r34)
    if source_digests["r34_step_summary"] != read_json(r34.with_name("manifest.json"))["files"][r34.name]["sha256"]:
        raise ValueError("Unexpected r34 decode step calibration")
    for run in ("r28", "r33", "r39"):
        folder = results / ARCHIVES[run]
        manifest = read_json(folder / "manifest.json")
        spec = manifest["workload"]["request_sources"]
        specs = {row["id"]: row for row in spec}
        if run == "r28":
            original_specs = specs
        for key, source in original_specs.items():
            if any(value != specs[key][field] for field, value in source.items() if field != "max_tokens"):
                raise ValueError(f"Benchmark schedule or input changed in {run}: {key}")
        trace = folder / f"{run}-events.jsonl.gz"
        source_digests[run] = sha256(trace)
        files = manifest.get("files", manifest.get("artifacts", {}))
        expected = files[trace.name]["sha256"] if files else manifest["trace_sha256"]
        if source_digests[run] != expected:
            raise ValueError(f"Unexpected input trace for {run}")
        for case, raw in read_trace(trace).items():
            if not case.endswith("cap100"):
                continue
            origin, = (event["monotonic_s"] for event in raw["client"] if event["type"] == "client_origin")
            profiles = requests_for(raw, spec, origin)
            check_observed(profiles)
            label = f"{run}/{case}"
            if run == "r28":
                samples.extend(suffix_samples(profiles, specs))
                roots = {key for key, row in profiles.items() if row["predecessor"] is None}
                baseline_profiles[label] = profiles
                natural[label] = {
                    "observed": fixed_service(profiles, {}),
                    "all_cold_root_P_HTTP_set_to_zero": no_prefill_residence(profiles, roots),
                    "all_successor_P_HTTP_set_to_zero": no_prefill_residence(profiles, profiles.keys() - roots),
                }
            else:
                detailed.append((label, raw, profiles, specs, origin))
    measured_order = {label: evaluate_arm(raw, profiles, specs, original_specs, origin, samples, steps)
                      for label, raw, profiles, specs, origin in detailed}
    r33_oracle_roots = set(measured_order["r33/b1-cap100"]["methods"]["best_fixed_service_oracle"]["promoted"])
    for label, profiles in baseline_profiles.items():
        natural[label]["r33_oracle_roots_P_HTTP_set_to_zero"] = no_prefill_residence(profiles, r33_oracle_roots)
    return {"source_sha256": source_digests, "r28_deployment_baseline": natural,
            "model_settings": {"gate_after_both_P_first_admissions_s": float(GATE_S),
                               "maximum_promoted_for_oracle": MAX_PROMOTIONS, "D_slots": D_SLOTS,
                               "score_prefill_s_per_token": P_SECONDS_PER_TOKEN,
                               "freeze_measured_outputs": True, "r39_fixed_length_and_placement": True},
            "r33_natural_instrumented_and_r39_controlled": measured_order}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(args.results))


if __name__ == "__main__":
    main()
