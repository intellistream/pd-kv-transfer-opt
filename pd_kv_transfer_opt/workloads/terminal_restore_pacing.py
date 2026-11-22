"""Calibrate offered times for the unchanged native proxy using a measured original-input replay."""

import argparse
from collections import defaultdict
from pathlib import Path
from random import Random
from statistics import median

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.profile_closed_sessions import read_trace, requests_for
from pd_kv_transfer_opt.workloads.terminal_restore_order import prefill_weights, routes


def calibrate(archive, *, use_recorded_first_P=False, screen_successor_output=None):
    manifest = read_json(archive / "manifest.json")
    event_path = archive / manifest["event_trace"]
    digest = sha256(event_path)
    if digest != manifest["artifacts"][event_path.name]["sha256"]:
        raise ValueError("The reference event trace differs from the recorded replay")
    workload = manifest["workload"]
    rows = {row["id"]: row for row in workload["request_sources"]}
    offered = {key: row["offered_s"] for key, row in rows.items()}
    roots = sorted((key for key, row in rows.items() if row["predecessor"] is None), key=offered.get)
    children = [key for key, row in rows.items() if row["predecessor"]]
    child_start = min(offered[key] for key in children)
    root_sequence, profile_cases = {}, {}
    first_P_by_case, screened = {}, {}
    delays = defaultdict(list)
    for name, arm in read_trace(event_path).items():
        origin, = [event["monotonic_s"] for event in arm["client"] if event["type"] == "client_origin"]
        profile = requests_for(arm, list(rows.values()), origin)
        if max(profile[key]["end_s"] for key in roots) >= child_start:
            raise ValueError(f"The reference has a root unfinished at the burst in {name}")
        root_sequence[name] = sorted(roots, key=lambda key: profile[key]["P_start_s"])
        profile_cases[name] = {key: profile[key]["P_start_s"] - offered[key] for key in rows}
        first = min(children, key=lambda key: profile[key]["P_start_s"])
        first_P_by_case[name] = profile[first]["P"]
        for key in children:
            delays[key].append(profile_cases[name][key])
            if screen_successor_output is not None:
                count = profile[key]["output_tokens"]
                reason = profile[key]["finish_reason"]
                if count > screen_successor_output or reason != "stop":
                    screened.setdefault(key, {})[name] = {"output_tokens": count, "reason": reason}
    children = [key for key in children if key not in screened]
    for key in screened:
        del rows[key]
    weights = prefill_weights(workload["arrival_order_screen"]["wire_body_bytes"])
    root_route = routes(roots, weights)
    source_route = {key: root_route[rows[key]["predecessor"]] for key in children}
    potential = {pair["successor"]: pair["potential_GET_bytes"] for pair in workload["selection"]["pairs"]}
    median_delay = {key: median(delays[key]) for key in children}
    # In the recorded cold-start controls the largest body needed as much as 286 ms.
    root_offer = {roots[0]: 0.0}
    root_offer.update({key: 0.36 + index * 0.07 for index, key in enumerate(roots[1:])})
    if any(sorted(roots, key=lambda key: root_offer[key] + lag[key]) != roots for lag in profile_cases.values()):
        raise ValueError("The new root offsets do not cover the observed startup variations")

    first_rank = 0
    if use_recorded_first_P:
        choices = set(first_P_by_case.values())
        if len(choices) != 1:
            raise ValueError("The native initial P choice was inconsistent across reference cases")
        first_rank = int(choices.pop()[-1])
    # Lead with a substantial miss on each P so releases do not alter early assigned load.
    first = max((key for key in children if source_route[key] != first_rank), key=potential.get)
    second = max((key for key in children if source_route[key] == first_rank), key=potential.get)
    remaining = sorted(set(children) - {first, second})
    first_proxy_s = child_start + median_delay[first]
    slots = [first_proxy_s, *(first_proxy_s + 0.12 + index * 0.028 for index in range(len(children) - 1))]
    rng = Random(20261122)
    best = None
    for _ in range(24000):
        order = [first, second, *rng.sample(remaining, len(remaining))]
        child_offer = {key: slot - median_delay[key] for key, slot in zip(order, slots, strict=True)}
        projected = {key: rank ^ first_rank for key, rank in routes(order, weights).items()}
        projected_bytes = sum(potential[key] for key in children if projected[key] != source_route[key])
        realized = {}
        for case, lag in profile_cases.items():
            actual = sorted(children, key=lambda key: child_offer[key] + lag[key])
            routed = {key: rank ^ first_rank for key, rank in routes(actual, weights).items()}
            realized[case] = sum(potential[key] for key in children if routed[key] != source_route[key])
        score = (min(realized.values()), projected_bytes)
        if best is None or score > best[0]:
            best = score, order, child_offer, projected, realized
    score, order, child_offer, projected, realized = best
    schedule = {**root_offer, **child_offer}
    return {**workload, "requests": len(rows), "last_session_start_s": max(root_offer.values()),
            "arrival_policy": "Keep exact inputs and predecessor pairs. Extend the first root offer gap and compensate each successor offer using recorded serialization and native proxy processing delay.",
            "arrival_order_screen": {**workload["arrival_order_screen"], "new_order_estimated_GET_bytes": score[1],
                "projected_root_P": {key: f"P{value}" for key, value in root_route.items()},
                "projected_successor_P": {key: f"P{value}" for key, value in projected.items()}},
            "arrival_gap_calibration": {"input_trace_sha256": digest, "recorded_root_orders": root_sequence,
                "median_offered_to_P_proxy_s": median_delay, "desired_proxy_order": order,
                "observed_delay_pattern_projection_bytes": realized,
                "observed_first_P": first_P_by_case, "projected_first_successor_P": f"P{first_rank}",
                "excluded_successors_after_reference_output_screen": screened,
                "assumptions": "Proxy order uses calibrated admission time. Selecting the same first P and retaining prefill reservations until all successors are selected are required for the projection."},
            "request_sources": [{**row, "offered_s": schedule[key]} for key, row in rows.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--use-recorded-first-P", action="store_true", dest="use_recorded_first_P")
    parser.add_argument("--screen-successor-output", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("The frozen arrival plan already exists")
    write_json(args.output, calibrate(args.archive, use_recorded_first_P=args.use_recorded_first_P,
                                     screen_successor_output=args.screen_successor_output))


if __name__ == "__main__":
    main()
