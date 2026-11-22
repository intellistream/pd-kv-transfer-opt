"""Select original Terminal-Bench turns with measured D-to-P restore demand."""

import argparse
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import median

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.screen_bidirectional_workload import D_RATE, P_RATE
from pd_kv_transfer_opt.workloads.terminal_eos_cohort import (
    MAX_OBSERVED_OUTPUT, MIN_OBSERVATIONS, ROOTS_PER_SECOND, observed_outputs,
)


def select(method_archive, eos_plan, target_release_s, max_sessions, max_eos=MAX_OBSERVED_OUTPUT, stagger_s=0):
    if max_sessions < 1 or not isfinite(target_release_s) or target_release_s <= (max_sessions - 1) / ROOTS_PER_SECOND:
        raise ValueError("Need at least one session and a finite successor release after the last root")
    if max_eos < 1 or not isfinite(stagger_s) or stagger_s < 0:
        raise ValueError("Need a positive EOS selection cutoff and a nonnegative arrival spacing")
    manifest = read_json(method_archive / "manifest.json")
    parent = manifest["workload"]
    parent_rows = {row["id"]: row for row in parent["request_sources"]}
    evidence_paths = [method_archive.parent / name for name in eos_plan["selection"]["evidence"]]
    observations, evidence = observed_outputs(evidence_paths, parent, parent_rows)
    summary_file = method_archive / "summary.json"
    method = read_json(summary_file)["cases"]

    def stable(key):
        history = observations[key]
        return len(history) >= MIN_OBSERVATIONS and all(
            state == "ok" and reason == "stop" and type(count) is int and 1 <= count <= max_eos
            for state, reason, count in history
        )

    candidates = defaultdict(list)
    for row in parent_rows.values():
        successor = row["id"]
        predecessor = row["predecessor"]
        if predecessor is None or row["prompt_tokens"] < 24_000:
            continue
        first = parent_rows[predecessor]
        if first["predecessor"] is not None or not (stable(predecessor) and stable(successor)):
            continue
        get_bytes = {arm: result["per_request"][successor]["P_direct_GET_bytes"] for arm, result in method.items()}
        if max(get_bytes.values()) < 200_000_000:
            continue
        predicted_output = sum(median(x[2] for x in observations[key]) for key in (predecessor, successor))
        new_tokens = max(result["per_request"][successor]["P_computed_prompt_tokens"] for result in method.values())
        predicted_compute_s = max((first["prompt_tokens"] + new_tokens) / P_RATE, predicted_output / D_RATE)
        candidates[row["session_id"]].append({
            "root": predecessor, "successor": successor, "root_prompt_tokens": first["prompt_tokens"],
            "observed_target_GET_bytes": get_bytes, "observed_median_output_tokens": predicted_output,
            "positive_in_both_arms": min(get_bytes.values()) >= 200_000_000,
            "GET_bytes_per_reference_second": sum(get_bytes.values()) / len(get_bytes) / predicted_compute_s,
        })

    def priority(candidate):
        return (not candidate["positive_in_both_arms"], -candidate["GET_bytes_per_reference_second"], candidate["successor"])

    winners = sorted((min(group, key=priority) for group in candidates.values()), key=priority)[:max_sessions]
    if len(winners) != max_sessions:
        raise ValueError("The source archive does not support the requested number of distinct sessions")
    if stagger_s:
        for item in winners:
            roots = [result["per_request"][item["root"]] for result in method.values()]
            item["historical_root_service_s"] = median(row["end_s"] - row["ready_s"] for row in roots)
        target_order = sorted(winners, key=lambda item: (item["historical_root_service_s"], item["successor"]))
        release = {picked["successor"]: target_release_s + index * stagger_s for index, picked in enumerate(target_order)}
    else:
        release = {picked["successor"]: target_release_s for picked in winners}
    schedule = {"successor_schedule": {"first_offer_s": target_release_s, "spacing_s": stagger_s,
                                       "order": "historical root residence"}} if stagger_s else {}
    release_description = "their planned release" if stagger_s else "the common release"
    requests = []
    for index, picked in enumerate(sorted(winners, key=lambda item: parent_rows[item["root"]]["offered_s"])):
        root, successor = (parent_rows[picked[key]] for key in ("root", "successor"))
        for row, offered in ((root, index / ROOTS_PER_SECOND), (successor, release[successor["id"]])):
            requests.append({**row, "original_offered_s": row["offered_s"], "offered_s": offered})

    fields = ("source_repo", "source_revision", "source_workload_sha256", "tokenizer_fingerprint", "ignore_eos",
              "temperature", "tool_gap_policy", "max_tokens_policy", "route_policy")
    return {
        **{key: parent[key] for key in fields}, "parent_r42_sha256": parent["workload_sha256"],
        "requests": len(requests), "sessions": len(winners), "last_session_start_s": (len(winners) - 1) / ROOTS_PER_SECOND,
        "arrival_policy": f"Independent original roots at 8/s; original successors eligible on {release_description} or real predecessor finish plus tool gap, whichever is later. All requests are in the throughput denominator.",
        "pressure_window_s": [target_release_s, target_release_s + 10],
        "selection": {
            "max_historical_eos_tokens": max_eos, "min_historical_observations": MIN_OBSERVATIONS,
            "method_trace_sha256": manifest["trace_sha256"], "method_summary_sha256": sha256(summary_file),
            "outputs": evidence, "selected_pairs": winners,
            "historical_target_GET_bytes": {arm: sum(w["observed_target_GET_bytes"][arm] for w in winners) for arm in method},
            **schedule,
            "reference_root_prefill_s": sum(w["root_prompt_tokens"] for w in winners) / P_RATE,
        },
        "request_sources": requests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method_archive", type=Path)
    parser.add_argument("eos_plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-release-s", type=float, default=20)
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--max-historical-eos", type=int, default=MAX_OBSERVED_OUTPUT)
    parser.add_argument("--successor-spacing-s", type=float, default=0)
    args = parser.parse_args()
    plan = select(args.method_archive, read_json(args.eos_plan), args.target_release_s, args.sessions,
                  args.max_historical_eos, args.successor_spacing_s)
    write_json(args.output, plan)
    print(plan["selection"]["historical_target_GET_bytes"], flush=True)


if __name__ == "__main__":
    main()
