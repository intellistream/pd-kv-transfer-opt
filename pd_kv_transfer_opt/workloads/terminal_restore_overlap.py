"""Find common ready successors while that run's native proxy still has reserved P work."""

import argparse
from pathlib import Path
from random import Random
from statistics import median
from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, read_trace, requests_for
from pd_kv_transfer_opt.workloads.terminal_restore_order import prefill_weights, routes


def screen(folder):
    manifest = read_json(folder / "manifest.json")
    trace_path = folder / manifest["event_trace"]
    trace_hash = sha256(trace_path)
    if trace_hash != manifest["artifacts"][trace_path.name]["sha256"]:
        raise ValueError("The archived trace failed verification")
    workload = manifest["workload"]
    rows = workload["request_sources"]
    weights = prefill_weights(workload["arrival_order_screen"]["wire_body_bytes"])
    parent = {row["id"]: row["predecessor"] for row in rows if row["predecessor"] is not None}
    roots = sorted({row["id"] for row in rows} - parent.keys())
    payloads = {pair["successor"]: pair["potential_GET_bytes"] for pair in workload["selection"]["pairs"]}
    cases = {}
    for name, arm in read_trace(trace_path).items():
        origin, = [e["monotonic_s"] for e in arm["client"] if e["type"] == "client_origin"]
        requests = requests_for(arm, rows, origin)
        release = {row["id"]: row["monotonic_ns"] / NS - origin for row in arm["proxy"]
                   if row["type"] == "decode_first_chunk" and row["id"] in roots}
        if release.keys() != set(roots):
            raise ValueError("Incomplete native P reservation release markers")
        cases[name] = requests, release

    def replay(order, requests, active):
        scores = list(active)
        assignments = {}
        for key in order:
            role = int(scores[1] < scores[0])
            assignments[key] = f"P{role}"
            scores[role] += weights[key]
        volume = sum(payloads[key] for key, role in assignments.items() if requests[parent[key]]["P"] != role)
        return volume, assignments

    rng = Random(20261123)
    findings = []
    last = max(release for _, end in cases.values() for release in end.values())
    # Scan observable common lifetime windows before any first-wave successor arrived.
    original_release = min(row["offered_s"] for row in rows if row["predecessor"] is not None)
    for tick in range(4, round(4 * min(last + 1, original_release))):
        t = tick / 4
        ready = sorted(key for key, prev in parent.items()
                       if all(req[prev]["end_s"] <= t - .15 for req, _ in cases.values()))
        if len(ready) < 4:
            continue
        scores = {}
        for name, (req, release) in cases.items():
            if any(t - .15 < value < t + .85 for value in release.values()):
                break
            active = [sum(weights[key] for key in roots
                          if req[key]["P"] == f"P{rank}" and req[key]["P_start_s"] <= t < release[key])
                      for rank in range(2)]
            if not any(active):
                break
            scores[name] = active
        if len(scores) != len(cases):
            continue
        best = None
        for _ in range(2500):
            order = rng.sample(ready, len(ready))
            modeled = {name: replay(order, cases[name][0], active) for name, active in scores.items()}
            worst_case = min(value[0] for value in modeled.values())
            if best is None or worst_case > best[0]:
                best = worst_case, order, modeled
        volume, order, modeled = best
        cold = {}
        for name, (req, _) in cases.items():
            cold[name] = {"P_HTTP_in_flight_at_start": sum(req[key]["P_start_s"] <= t < req[key]["P_end_s"] for key in roots),
                          "P_to_D_in_next_8_s": sum(t <= req[key]["D_start_s"] < t + 8 for key in roots),
                          "D_running_at_start": sum(req[key]["first_s"] <= t < req[key]["end_s"] for key in roots)}
        findings.append({"earliest_s": t, "ready_requests": len(ready), "launch_order": order,
                         "min_predicted_GET_bytes": volume,
                         "per_case_GET_bytes": {name: data[0] for name, data in modeled.items()},
                         "projected_P": {name: data[1] for name, data in modeled.items()},
                         "initial_P_weight": scores, "observed_cold_work": cold,
                         "later_requests": sorted(parent.keys() - set(ready))})
    prior_P = {key: sorted({req[key]["P"] for req, _ in cases.values()}) for key in roots}
    median_lag = {key: median(req[key]["P_start_s"] - req[key]["dispatch_s"] for req, _ in cases.values())
                  for key in parent}
    return {"reference_trace_sha256": trace_hash, "readiness_margin_s": .15, "guard_after_start_s": .85,
            "reference_P_by_root": prior_P, "successor_median_dispatch_to_P_s": median_lag,
            "candidates": sorted(findings, key=lambda row: row["min_predicted_GET_bytes"], reverse=True)}


def make_plan(workload, screening):
    if not screening["candidates"]:
        raise ValueError("No common ready successor burst with outstanding native P work")
    early = screening["candidates"][0]
    rows = workload["request_sources"]
    lag = screening["successor_median_dispatch_to_P_s"]
    P_root = screening["reference_P_by_root"]
    if any(len(placement) != 1 for placement in P_root.values()):
        raise ValueError("The two-burst plan needs matched reference root placement")
    P_root = {key: placement[0] for key, placement in P_root.items()}
    parents = {row["id"]: row["predecessor"] for row in rows if row["predecessor"] is not None}
    first_placements = list(early["projected_P"].values())
    if any(plan != first_placements[0] for plan in first_placements[1:]):
        raise ValueError("The early burst's predicted P assignments differ across reference arms")
    weights = prefill_weights(workload["arrival_order_screen"]["wire_body_bytes"])
    payloads = {pair["successor"]: pair["potential_GET_bytes"] for pair in workload["selection"]["pairs"]}
    rng = Random(20261124)
    remaining = early["later_requests"]
    if not remaining:
        raise ValueError("Choose an overlapping wave with a second group still to send")
    best = None
    for _ in range(24_000):
        order = rng.sample(remaining, len(remaining))
        guessed = routes(order, weights)
        starts = {
            first: sum(payloads[key] for key, rank in guessed.items() if P_root[parents[key]] != f"P{rank ^ first}")
            for first in (0, 1)
        }
        floor = min(starts.values())
        if best is None or floor > best[0]:
            best = floor, order, starts
    late_floor, late_order, late_bytes = best
    late_assignments = routes(late_order, weights)
    late_placements = {f"P{first}": {key: f"P{rank ^ first}" for key, rank in late_assignments.items()}
                       for first in (0, 1)}

    def offered(order, start, spacing, first_gap):
        # Offset body-processing time while keeping the leading offer at start.
        base = start + lag[order[0]]
        times = {order[0]: start}
        for index, key in enumerate(order[1:]):
            times[key] = base + first_gap + index * spacing - lag[key]
        if min(times.values()) != start:
            raise ValueError("The transport delay calibration would predate the leading offer")
        return times

    early_start, late_start = early["earliest_s"], 95.0
    arrivals = {**offered(early["launch_order"], early_start, .041, .105),
                **offered(late_order, late_start, .029, .115)}
    if arrivals.keys() != parents.keys():
        raise ValueError("The partition does not cover every original successor")
    return {**workload, "pressure_window_s": [early_start, 108.0],
            "arrival_policy": "Keep the reference roots. Offer one burst of ready original successors during live cold work, then release the other unchanged successors in a second burst.",
            "arrival_order_screen": {**workload["arrival_order_screen"],
                                     "projected_successor_P": {**first_placements[0], **late_placements["P0"]},
                                     "new_order_estimated_GET_bytes": early["min_predicted_GET_bytes"] + late_floor},
            "overlap_arrival_calibration": {
                "reference_trace_sha256": screening["reference_trace_sha256"], "early": early,
                "late_release_s": late_start, "late_order": late_order,
                "late_GET_bytes_by_first_P": {f"P{key}": value for key, value in late_bytes.items()},
                "late_projected_P_by_first_P": late_placements,
                "anticipated_limitations": "The early projection assumes outstanding root reservations remain through its short arrival window. The late projection assumes the remaining roots have released their P reservations.",
            },
            "request_sources": [{**row, "offered_s": arrivals.get(row["id"], row["offered_s"])} for row in rows]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--write-plan", type=Path)
    args = parser.parse_args()
    findings = screen(args.archive)
    write_json(args.output, findings)
    for candidate in findings["candidates"][:4]:
        print(candidate["earliest_s"], candidate["ready_requests"], candidate["min_predicted_GET_bytes"] / 1e9)
    if args.write_plan:
        if args.write_plan.exists():
            raise ValueError("The frozen overlap plan already exists")
        manifest = read_json(args.archive / "manifest.json")
        write_json(args.write_plan, make_plan(manifest["workload"], findings))
