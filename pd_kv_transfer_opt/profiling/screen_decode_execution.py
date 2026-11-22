"""Estimate how a faster D singleton would affect the natural r28 throughput."""

import argparse
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.compare_session_runs import (
    ARCHIVES, DECODE, D_ROLES, decode_tails, singleton_samples, timeline,
)
from pd_kv_transfer_opt.profiling.profile_closed_sessions import (
    check_observed, phase_times, read_trace, requests_for,
)


def sampled_singletons(profiles, telemetry, origin):
    """Count consecutive sole-D windows confirmed by clients and native counters."""
    seconds = {}
    for role in D_ROLES:
        boundaries = []
        for key, row in profiles.items():
            if row["D"] == role:
                boundaries.extend(((row["D_start_s"], key, True), (row["end_s"], key, False)))
        active, left = set(), 0.0
        for right, key, entering in sorted(boundaries):
            if len(active) == 1:
                only, = active
                begin = max(left, profiles[only]["first_s"])
                if right > begin:
                    sample = singleton_samples(telemetry, origin, role, begin, right)
                    seconds[only] = seconds.get(only, 0.0) + sample["covered_s"]
            if entering:
                active.add(key)
            else:
                active.remove(key)
            left = right
    return {key: value for key, value in seconds.items() if value > 0}


def scale(profiles, before, singleton):
    phases = phase_times(profiles)
    estimates = {}
    for fraction in (0.1, 0.2, 0.3):
        costs = {key: {**row, DECODE: row[DECODE] - fraction * singleton.get(key, 0)}
                 for key, row in phases.items()}
        after = timeline(profiles, costs)
        estimates[str(fraction)] = {**after,
                                   "gain_pct": 100 * (after["throughput_tokens_s"] / before["throughput_tokens_s"] - 1)}
    return estimates


def assess(root, stem, accept):
    path = root / f"{stem}-events.jsonl.gz"
    manifest = read_json(root / "manifest.json")
    digest = sha256(path)
    artifacts = manifest.get("artifacts", manifest.get("files"))
    if digest != artifacts[path.name]["sha256"]:
        raise ValueError(f"Unexpected {stem} input")
    arms = {}
    for name, raw in read_trace(path).items():
        if not accept(name):
            continue
        origin, = (row["monotonic_s"] for row in raw["client"] if row["type"] == "client_origin")
        profiles = requests_for(raw, manifest["workload"]["request_sources"], origin)
        check_observed(profiles)
        before = timeline(profiles, phase_times(profiles))
        tails = decode_tails(profiles, raw["telemetry"], origin)
        singleton = {v["terminal_request"]: v["singleton_s"] for v in tails.values()}
        covered = sampled_singletons(profiles, raw["telemetry"], origin)
        arms[name] = {"observed": before, "D_terminal_observations": tails,
                      "only_D_singleton_scaled": scale(profiles, before, singleton),
                      "sampled_single_D": covered,
                      "all_observed_single_D_windows_scaled": scale(profiles, before, covered)}
    return digest, arms


def analyze(results):
    digest, arms = assess(results / ARCHIVES["r28"], "r28", lambda name: name.endswith("cap100"))
    base_ms = 13.591
    estimates = {str(cycle_ms): {
        "break_even_acceptance": cycle_ms / base_ms - 1,
        "acceptance_for_twenty_percent_less_time_per_token": cycle_ms / (0.8 * base_ms) - 1,
    } for cycle_ms in (16, 18)}
    output = {"r28_trace_sha256": digest, "r28_natural_100": arms,
              "one_draft_cost_scenarios": {"r34_native_singleton_step_p50_ms": base_ms,
                                          "estimates_by_assumed_cycle_ms": estimates}}
    recent = results / "2026-11-20-r40-prefill-order"
    if (recent / "manifest.json").is_file():
        receipt, repeats = assess(recent, "r40", lambda name: name.endswith("-native"))
        output.update(r40_trace_sha256=receipt, r40_native_100=repeats)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(args.results))


if __name__ == "__main__":
    main()
