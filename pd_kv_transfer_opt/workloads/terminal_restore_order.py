"""Choose a successor arrival order using traces of the unchanged native proxy."""

import argparse
from pathlib import Path
from random import Random

from pd_kv_transfer_opt.common import read_json, read_jsonl, sha256, write_json
from pd_kv_transfer_opt.profiling.profile_closed_sessions import read_trace, requests_for
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS, ATTENTION_BYTES, STATE_BYTES


def prefill_weights(body_bytes):
    # P priority weights active_kv_cache uniformly, so the common 0.3 factor cancels.
    return {key: size / 4 * 0.0345 + 120.0745 for key, size in body_bytes.items()}


def routes(order, weights):
    load = [0.0, 0.0]
    assignments = {}
    for key in order:
        role = int(load[1] < load[0])
        assignments[key] = role
        load[role] += weights[key]
    return assignments


def choose_order(keys, weights, source_roles, potential, mandatory=(), candidates=24000):
    rng = Random(20261122)
    best = None
    for _ in range(candidates):
        order = rng.sample(keys, len(keys))
        selected = routes(order, weights)
        moving = {key for key in order if selected[key] != source_roles[key]}
        if all(key in moving for key in mandatory):
            volume = sum(potential[key] for key in moving)
            if best is None or volume > best[0]:
                best = volume, order, selected
    return best


def selection(directory, ledger_file, body_file, candidates=24000):
    manifest = read_json(directory / "manifest.json")
    workload = manifest["workload"]
    trace_file = directory / manifest["event_trace"]
    digest = sha256(trace_file)
    if digest != manifest["artifacts"][trace_file.name]["sha256"]:
        raise ValueError("Source trace checksum mismatch")
    specs = workload["request_sources"]
    originals = {row["id"]: row for row in specs}
    initial_order = sorted((row["id"] for row in specs if row["predecessor"]), key=lambda key: originals[key]["offered_s"])
    planned_times = [originals[key]["offered_s"] for key in initial_order]
    if len(set(planned_times)) != len(initial_order):
        raise ValueError("The reference needs distinct frozen successor arrivals")
    measured_body = read_json(body_file)
    if measured_body["parent_sha256"] != workload["parent_r42_sha256"] or measured_body["wire_body_bytes"].keys() != originals.keys():
        raise ValueError("Request body sizes do not match the pinned workload")
    weights = prefill_weights(measured_body["wire_body_bytes"])
    estimated = routes(initial_order, weights)
    anchors = {}
    tails = set()
    modeled_matches = {}
    for case, arm in read_trace(trace_file).items():
        origin, = [row["monotonic_s"] for row in arm["client"] if row["type"] == "client_origin"]
        profile = requests_for(arm, specs, origin)
        tail = max(profile, key=lambda key: profile[key]["end_s"])
        if tail not in initial_order:
            raise ValueError(f"Reference tail is not one of the selected successors in {case}")
        tails.add(tail)
        for key in initial_order:
            root = profile[originals[key]["predecessor"]]
            route = int(root["P"][-1])
            if root["end_s"] >= planned_times[0] or anchors.setdefault(key, route) != route:
                raise ValueError(f"Reference predecessor readiness or P route changed in {case}")
        modeled_matches[case] = sum(f"P{estimated[key]}" == profile[key]["P"] for key in initial_order)

    ledger = {row["id"]: row for row in read_jsonl(ledger_file)}
    potential = {}
    for key in initial_order:
        row = ledger[key.split("-", 1)[1]]
        if row["token_ids_uint32le_sha256"] != originals[key]["prompt_sha256"]:
            raise ValueError(f"Source token hash does not match {key}")
        blocks = row["source_prompt_lcp_to_previous"] // ATTENTION_BLOCK_TOKENS
        potential[key] = blocks * ATTENTION_BYTES + STATE_BYTES

    best = choose_order(initial_order, weights, anchors, potential, tails, candidates)
    old_bytes = sum(potential[key] for key in initial_order if estimated[key] != anchors[key])
    if best is None or best[0] <= old_bytes:
        raise ValueError("No candidate increases estimated bytes and includes the original tail")

    volume, order, guesses = best
    schedule = dict(zip(order, planned_times, strict=True))
    return {**workload,
            "arrival_policy": "Keep the original root arrivals and the same successor release times. Reassign successor offers to those times using the native proxy arrival-order screen. All predecessors must still finish.",
            "arrival_order_screen": {
                "reference_trace_sha256": digest, "source_ledger_sha256": sha256(ledger_file),
                "body_length_receipt_sha256": sha256(body_file), "candidate_orders": candidates,
                "reference_route_matches_by_arm": modeled_matches,
                "prior_tail_requests": sorted(tails), "initial_order_estimated_GET_bytes": old_bytes,
                "new_order_estimated_GET_bytes": volume, "full_restore_upper_bytes": sum(potential.values()),
                "successors": [{"id": key, "source_P": f"P{anchors[key]}",
                                "estimated_P": f"P{guesses[key]}", "potential_GET_bytes": potential[key],
                                "body_bytes": measured_body["wire_body_bytes"][key], "offered_s": schedule[key]}
                               for key in order],
            },
            "request_sources": [{**row, "offered_s": schedule.get(row["id"], row["offered_s"])} for row in specs]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("body_sizes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan = selection(args.reference, args.ledger, args.body_sizes)
    write_json(args.output, plan)
    screen = plan["arrival_order_screen"]
    print(screen["initial_order_estimated_GET_bytes"], screen["new_order_estimated_GET_bytes"], flush=True)


if __name__ == "__main__":
    main()
