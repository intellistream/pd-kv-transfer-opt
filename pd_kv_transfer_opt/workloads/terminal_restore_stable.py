"""Plan a short-output shared-prefix replay from existing native bandwidth arms."""

import argparse
from collections import defaultdict, deque
from heapq import heapify, heappop, heappush
from pathlib import Path
from random import Random
from statistics import median

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.profile_closed_sessions import read_trace, requests_for
from pd_kv_transfer_opt.workloads.terminal_restore_checkpoints import plan
from pd_kv_transfer_opt.workloads.terminal_restore_order import prefill_weights


def recoverable(pairs, profiles, limit):
    return [pair for pair in pairs if all(rows[pair[role]]["finish_reason"] == "stop"
                                         and 0 < rows[pair[role]]["output_tokens"] <= limit
                                         for rows in profiles.values() for role in ("root", "successor"))]


def initial_reservations(pairs, rows, weights, root_roles, start):
    return [(rows[pair["root"]]["P_credit_release_s"], root_roles[pair["root"]], weights[pair["root"]])
            for pair in pairs if rows[pair["root"]]["P_credit_release_s"] > start]


def predict(order, pairs, by_child, view, weights, root_roles, start, *, idle_role=None):
    arrivals = []
    for index, key in enumerate(order):
        offer = start if index == 0 else start + .14 + (index - 1) * .049
        parent = by_child[key]["root"]
        arrivals.append((max(offer, view[parent]["end_s"]), index, key))
    release = initial_reservations(pairs, view, weights, root_roles, start)
    heapify(release)
    active = [sum(weight for _, role, weight in release if role == r) for r in (0, 1)]
    volume = 0
    for at, _, key in sorted(arrivals):
        while release and release[0][0] <= at:
            _, rank, mass = heappop(release)
            active[rank] -= mass
        equal = idle_role is not None and abs(active[1] - active[0]) < 1e-7
        route = idle_role if equal else int(active[1] < active[0])
        pair = by_child[key]
        active[route] += weights[key]
        delay = view[key]["P_credit_release_s"] - view[key]["P_start_s"]
        heappush(release, (at + delay, route, weights[key]))
        if route != root_roles[pair["root"]]:
            volume += pair["potential_GET_bytes"]
    return volume


def proposed_orders(pairs, weights, root_roles, profiles, start, initial, *, ready_first=False):
    yield initial
    rng = Random(20261126)
    families = defaultdict(list)
    for pair in pairs:
        families[pair["task"]].append(pair)
    known_ready = {p["successor"] for p in pairs
                   if all(rows[p["root"]]["end_s"] + .15 <= start for rows in profiles.values())}
    for rows in families.values():
        rows.sort(key=lambda row: (ready_first and row["successor"] not in known_ready,
                                   -row["checkpoint_blocks"], row["successor"]))
    for view in profiles.values():
        pending = initial_reservations(pairs, view, weights, root_roles, start)
        base = [sum(weight for _, role, weight in pending if role == r) for r in (0, 1)]
        for _ in range(120):
            queues = {task: deque(rows) for task, rows in families.items()}
            active = list(base)
            order = []
            while queues:
                rank = int(active[1] < active[0])
                heads = [entries[0] for entries in queues.values()]
                ready = [pair for pair in heads if pair["successor"] in known_ready]
                matching = [pair for pair in (ready or heads) if root_roles[pair["root"]] != rank]
                choice = rng.choice(matching or ready or heads)
                task, key = choice["task"], choice["successor"]
                order.append(key)
                active[rank] += weights[key]
                queues[task].popleft()
                if not queues[task]:
                    del queues[task]
            yield order


def guard_cold_arrivals(workload):
    screen = workload["arrival_order_screen"]
    order = screen["planned_root_order"]
    sizes = screen["wire_body_bytes"]
    fit = screen["admission_fit"]
    times = {order[0]: 0.0}
    earliest = .72
    for index, key in enumerate(order[1:]):
        correction = fit["s_per_byte"] * (sizes[order[0]] - sizes[key]) if fit else 0.0
        times[key] = earliest + correction
        earliest += .12 if index < 3 else .068
    workload["request_sources"] = [{**row, "offered_s": times.get(row["id"], row["offered_s"])}
                                  for row in workload["request_sources"]]
    workload["last_session_start_s"] = max(times.values())
    screen["cold_arrival_guard"] = {"first_gap_s": .72, "first_three_remaining_gaps_s": .12,
                                    "subsequent_gap_s": .068, "reason": "Leave room for the first large HTTP request and the initial request-body preparation delays observed in the parent replay."}


def read_reference(archive):
    manifest = read_json(archive / "manifest.json")
    trace = archive / manifest["event_trace"]
    trace_sha256 = sha256(trace)
    if trace_sha256 != manifest["artifacts"][trace.name]["sha256"]:
        raise ValueError("The parent arms no longer match their receipt")
    parent = manifest["workload"]
    profiles = {}
    for name, trace_arm in read_trace(trace).items():
        origin, = (row["monotonic_s"] for row in trace_arm["client"] if row["type"] == "client_origin")
        rows = requests_for(trace_arm, parent["request_sources"], origin)
        released = {event["id"]: event["monotonic_ns"] / 1e9 - origin
                    for event in trace_arm["proxy"] if event["type"] == "decode_first_chunk"}
        if released.keys() != rows.keys():
            raise ValueError(f"The native P reservation release was not recorded for every request in {name}")
        profiles[name] = {key: {**row, "P_credit_release_s": released[key]} for key, row in rows.items()}
    if {int(name.rsplit("cap", 1)[1]) for name in profiles} != {100, 200}:
        raise ValueError("The screen needs complete native arms at both bandwidths")
    return parent, profiles, trace_sha256


def make_plan(archive, limit, *, release_s=None, guard_roots=False, exclude_source_tasks=(), additional_references=()):
    parent, profiles, trace_sha256 = read_reference(archive)
    reference_hashes = {archive.name: trace_sha256}
    first_reference_release = parent["pressure_window_s"][0]
    for path in additional_references:
        other, additional, digest = read_reference(path)
        fields = ("id", "prompt_sha256", "predecessor", "max_tokens")
        original = {tuple(row[k] for k in fields) for row in parent["request_sources"]}
        comparable = {tuple(row[k] for k in fields) for row in other["request_sources"]}
        if original != comparable or other["parent_requests_sha256"] != parent["parent_requests_sha256"]:
            raise ValueError("Arrival references must replay exactly the same original requests")
        if path.name in reference_hashes:
            raise ValueError("Arrival reference names must be distinct")
        reference_hashes[path.name] = digest
        first_reference_release = min(first_reference_release, other["pressure_window_s"][0])
        profiles.update({f"{path.name}/{case}": rows for case, rows in additional.items()})
    excluded = set(exclude_source_tasks)
    parent_pairs = parent["selection"]["pairs"]
    if not excluded.issubset({pair["task"] for pair in parent_pairs}):
        raise ValueError("Every excluded task must occur in the parent workload")
    pairs = [pair for pair in recoverable(parent_pairs, profiles, limit) if pair["task"] not in excluded]
    if not pairs:
        raise ValueError("The reference arms have no original windows meeting the output screen")
    keys = {pair[role] for pair in pairs for role in ("root", "successor")}
    sizes = {key: size for key, size in parent["arrival_order_screen"]["wire_body_bytes"].items() if key in keys}
    weights = prefill_weights(sizes)
    selected = {**parent["selection"], "pairs": pairs}
    spec = {"source_repo": parent["source_repo"], "source_revision": parent["source_revision"],
            "tokenizer_sha256": parent["tokenizer_fingerprint"], "selection": selected,
            "requests": [{**row, "offered_s": row["original_offered_s"], "token_ids_uint32le_sha256": row["prompt_sha256"]}
                         for row in parent["request_sources"] if row["id"] in keys]}
    start = first_reference_release if release_s is None else release_s
    if not 0 < start <= first_reference_release:
        raise ValueError("Check release times before the parent experiment dispatches any successors")
    fit = parent["arrival_order_screen"].get("admission_fit")
    receipt = {"wire_body_bytes": sizes, "sha256": parent["parent_requests_sha256"]}
    provisional = plan(spec, receipt, start, admission_fit=fit)
    base = provisional["arrival_order_screen"]
    root_roles = {key: int(role[1]) for key, role in base["projected_root_P"].items()}
    by_child = {pair["successor"]: pair for pair in pairs}

    def estimated(order):
        idle_roles = (0, 1) if guard_roots else (None,)
        return {name: min(predict(order, pairs, by_child, view, weights, root_roles, start, idle_role=role)
                          for role in idle_roles) for name, view in profiles.items()}

    def score(amounts):
        low, high = min(amounts.values()), max(amounts.values())
        return low - .5 * (high - low), low

    initial = base["planned_successor_order"]
    alternatives = ((order, estimated(order)) for order in proposed_orders(pairs, weights, root_roles, profiles, start, initial))
    chosen, volumes = max(alternatives, key=lambda item: score(item[1]))
    frozen = plan(spec, receipt, start, admission_fit=fit, ordered_successors=chosen)
    if guard_roots:
        guard_cold_arrivals(frozen)
    frozen["stable_checkpoint_calibration"] = {
        "reference_trace_sha256": trace_sha256, "max_natural_observed_tokens": limit,
        "additional_reference_trace_sha256": {name: digest for name, digest in reference_hashes.items() if name != archive.name},
        "excluded_source_tasks": sorted(excluded),
        "reference_cases": list(profiles), "reference_projected_GET_bytes": volumes,
        "reference_neutral_GET_bytes": estimated(initial),
        "screened_both_idle_role_choices": guard_roots,
        "reference_on_time_full_read_bytes_by_case": {name: sum(pair["potential_GET_bytes"] for pair in pairs
                                                                if view[pair["root"]]["end_s"] <= start - .15)
                                                        for name, view in profiles.items()},
        "observed_successor_output_tokens_by_case": {name: sum(view[pair["successor"]]["output_tokens"] for pair in pairs)
                                                       for name, view in profiles.items()},
        "observed_median_successor_tokens": median(view[pair["successor"]]["output_tokens"]
                                                   for pair in pairs for view in profiles.values()),
        "forecast_assumptions": "Reference parent readiness and P reservation release delays are reused; the native proxy selects destinations from the measured-style weighted reservations. Actual pacing, locality and effective GET bytes must be verified on the frozen replay.",
    }
    frozen["arrival_policy"] = "Replay only original windows observed to stop within the screening threshold. Freeze the successor burst order from the native proxy credit replay. Stop and generation budgets remain those of the original client."
    return frozen


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--max-observed-output", type=int, default=800)
    cli.add_argument("--release-s", type=float)
    cli.add_argument("--guard-root-arrivals", action="store_true")
    cli.add_argument("--exclude-task", action="append", default=[])
    cli.add_argument("--reference", type=Path, action="append", default=[])
    args = cli.parse_args()
    if args.output.exists():
        raise FileExistsError("The selected plan already exists")
    result = make_plan(args.archive, args.max_observed_output, release_s=args.release_s, guard_roots=args.guard_root_arrivals,
                       exclude_source_tasks=args.exclude_task, additional_references=args.reference)
    write_json(args.output, result)
    print(result["sessions"], result["stable_checkpoint_calibration"])
