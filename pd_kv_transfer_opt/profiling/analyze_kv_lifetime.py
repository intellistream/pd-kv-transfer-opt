"""Join physical page observations to EOS replay completion and session releases."""

import argparse
from collections import defaultdict
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, phase_times, read_trace, replay, requests_for


def index_events(events):
    indexed = defaultdict(dict)
    for row in events:
        name = row["type"]
        if name == "p_allocation_first_dispatch":
            for key in row["request_ids"]:
                indexed[key].setdefault(name, row)
        elif "request_id" in row:
            key = row.get("p_request_id", row["request_id"])
            if name == "p_pool_denied":
                indexed[key].setdefault(name, []).append(row)
            else:
                indexed[key].setdefault(name, row)
    return indexed


def change_finish(profiles, advance, phase):
    costs = {key: dict(durations) for key, durations in phase_times(profiles).items()}
    for key, seconds in advance.items():
        costs[key][phase] -= min(costs[key][phase], seconds)
    _, done, chain = replay(profiles, costs)
    before = max(row["end_s"] for row in profiles.values())
    after = max(done.values())
    return {"original_finish_s": before, "counterfactual_finish_s": after,
            "advance_s": before - after, "resulting_tail_chain": chain,
            "fixed_service_and_EOS": True}


def seconds_between(begin, end):
    return (end["monotonic_ns"] - begin["monotonic_ns"]) / NS


def report_case(arm, workload, clock_guard_ns):
    origin, = (row["monotonic_s"] for row in arm["client"] if row["type"] == "client_origin")
    profiles = requests_for(arm, workload, origin)
    index = index_events(arm["physical"])
    if index.keys() - profiles.keys():
        raise ValueError("Physical observations include unknown workload requests")
    expected = {"p_handoff_retain", "p_handoff_release", "d_paged_receive_arrival",
                "d_paged_receive_allocated", "d_paged_receive_ready"}
    missing = {key: sorted(expected - index[key].keys()) for key in profiles if expected - index[key].keys()}
    if missing:
        raise ValueError(f"Incomplete page lifecycle coverage: {missing}")

    by_p = defaultdict(list)
    by_request = {}
    denials = []
    attributable_wait = {}
    for key, events in index.items():
        p_hold, p_free = (events[k] for k in ("p_handoff_retain", "p_handoff_release"))
        d_arrival, d_alloc, d_ready = (events[k] for k in
            ("d_paged_receive_arrival", "d_paged_receive_allocated", "d_paged_receive_ready"))
        if not (p_hold["instance"] == p_free["instance"] == profiles[key]["P"]
                and d_arrival["instance"] == d_alloc["instance"] == d_ready["instance"] == profiles[key]["D"]):
            raise ValueError(f"Native and proxy routes disagree for {key}")
        hold = seconds_between(p_hold, p_free)
        d_wait = seconds_between(d_arrival, d_alloc)
        d_comm = seconds_between(d_alloc, d_ready)
        if min(hold, d_wait, d_comm) < 0 or p_free["cause"] != "connector_finished_sending":
            raise ValueError(f"Invalid release or receive for {key}")
        metrics = {"P": p_hold["instance"], "D": d_arrival["instance"],
                   "p_hold_s": hold, "P_referenced_blocks": p_hold["referenced_blocks"],
                   "P_just_this_reference_blocks_on_retain": p_hold["exclusively_owned_blocks"],
                   "P_physically_returned_blocks": p_free["returned_blocks"], "D_arrival_to_alloc_s": d_wait,
                   "D_alloc_to_KV_ready_s": d_comm}
        by_p[metrics["P"]].append(metrics)
        by_request[key] = metrics
        physically_attributable = False
        for row in events.get("p_pool_denied", []):
            missing_blocks = row["need_blocks"] - row["free_blocks"]
            holders = [donor for donor, blocks in row["pending_holders"].items() if blocks]
            # Only certify the conservative bound if every contributing reference is still before D allocation.
            all_waiting = bool(holders) and all(
                index[donor]["d_paged_receive_allocated"]["unix_ns"] - row["unix_ns"] >= clock_guard_ns
                for donor in holders)
            physically_caused = 0 < missing_blocks <= row["detachable_outside_prefix"] and all_waiting
            physically_attributable |= physically_caused
            denials.append({"request": key, "instance": row["instance"], "gate": row["gate"],
                            "free_blocks": row["free_blocks"], "needed_blocks": row["need_blocks"],
                            "reserved_blocks": row["reserved_blocks"], "missing_physical_blocks": max(0, missing_blocks),
                            "detachable_outside_prefix": row["detachable_outside_prefix"],
                            "all_contributing_holders_still_waiting_for_D_pages": all_waiting,
                            "ref_release_would_fit_at_observed_check": physically_caused, "holders": holders})
        if physically_attributable and (resumed := events.get("p_pool_allocation_resumed")) is not None:
            attributable_wait[key] = (resumed["monotonic_ns"] - resumed["first_fail_ns"]) / NS

    _, _, tail = replay(profiles, phase_times(profiles))
    d_alloc_window = {key: row["D_alloc_to_KV_ready_s"] for key, row in by_request.items()}
    samples = defaultdict(list)
    for row in arm["physical"]:
        if row["type"] == "p_pool_sample":
            samples[row["instance"]].append(row)
    physical = {}
    for role, rows in samples.items():
        rows.sort(key=lambda row: row["monotonic_ns"])
        attempted = [right for left, right in zip(rows, rows[1:]) if right["calls"] > left["calls"]]
        if not attempted:
            raise ValueError(f"Missing exact allocation checkpoints for {role}")
        physical[role] = {"observed_calls_lower_bound": rows[-1]["calls"] - rows[0]["calls"],
                          "unobserved_checks_in_sample_range":
                              rows[-1]["unobserved_checks"] - rows[0]["unobserved_checks"],
                          "min_free_at_observed_check": min(row["min_attempt_free"] for row in attempted),
                          "min_slack_at_observed_check": min(row["min_attempt_slack"] for row in attempted),
                          "peak_sampled_reclaimable_held_blocks": max(row["detachable_blocks"] for row in rows),
                          "peak_sampled_held_requests": max(row["holder_count"] for row in rows)}
    if set(physical) != {"P0", "P1"}:
        raise ValueError("Both P instances need allocator coverage")

    outputs = sum(row["output_tokens"] for row in profiles.values())
    finish = max(row["end_s"] for row in profiles.values())
    release_by_p = {role: {
        "requests": len(rows), "held_s": distribution(row["p_hold_s"] for row in rows),
        "returned_physical_block_refs": sum(row["P_physically_returned_blocks"] for row in rows)}
        for role, rows in by_p.items()}
    return {
        "whole_replay": {"requests": len(profiles), "output_tokens": outputs,
                         "drain_s": finish, "output_tokens_per_s": outputs / finish},
        "allocator_checkpoints": physical, "releases": release_by_p,
        "sampled_denials": denials, "requests_with_denials": len({row["request"] for row in denials}),
        "requests_with_fitting_detach_before_D_allocation": len({row["request"] for row in denials
            if row["ref_release_would_fit_at_observed_check"]}),
        "matched_receive_phase_s": {metric: distribution(row[metric] for row in by_request.values())
                                    for metric in ("D_arrival_to_alloc_s", "D_alloc_to_KV_ready_s")},
        "D_arrival_to_alloc_over_5s": sum(row["D_arrival_to_alloc_s"] > 5 for row in by_request.values()),
        "counterfactuals": {
            "remove_observed_attributable_allocator_episodes": change_finish(profiles, attributable_wait, "P_HTTP"),
            "remove_entire_D_alloc_to_ready_at_fixed_admission": change_finish(profiles, d_alloc_window, "D_to_first"),
        },
        "observed_tail_chain": tail, "per_request": by_request,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("workload_spec", type=Path, help="Replay workload receipt or published archive manifest")
    parser.add_argument("output", type=Path)
    parser.add_argument("--clock-guard-ms", type=int, default=150)
    args = parser.parse_args()
    manifest = read_json(args.workload_spec)
    digest = sha256(args.trace)
    if "files" in manifest and manifest["files"][args.trace.name]["sha256"] != digest:
        raise ValueError("Trace does not match the archive manifest")
    inputs = manifest.get("workload", manifest)["request_sources"]
    cases = read_trace(args.trace)
    report = {case: report_case(arm, inputs, args.clock_guard_ms * 10**6) for case, arm in cases.items()}
    write_json(args.output, {"trace_sha256": digest, "clock_guard_ms": args.clock_guard_ms, "cases": report})


if __name__ == "__main__":
    main()
