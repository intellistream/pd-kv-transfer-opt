"""Attribute byte savings and drained throughput in the r38 long-session replay."""

import argparse
from collections import Counter
from itertools import combinations
from math import ceil, comb, isclose
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution, interval_union
from pd_kv_transfer_opt.profiling.profile_closed_sessions import (
    compare_arms, measure_arms, read_trace, replay, requests_for,
)


def summarize_store(events):
    gets = [row for row in events if row["type"] == "r38_physical_store_get"]
    decisions = [row for row in events if row["type"] == "r18_restore_decision"]
    plans = [row for row in events if row["type"] == "r18_packed_preflight"]
    copies = [row for row in events if row["type"] == "same_head_local_copy"]
    checks = [row for row in events if row["type"] == "same_head_verify"]
    actual = sum(row["payload_bytes"] for row in gets)
    check_bytes = sum(row["extra_get_bytes"] for row in checks)
    observed_bytes = actual - check_bytes
    native_bytes = sum(row["native_bytes"] for row in decisions) if decisions else actual
    if not 0 <= observed_bytes <= native_bytes or any(not row["ok"] for row in gets):
        raise ValueError("Physical GET and native-equivalent descriptors do not reconcile")
    get_spans = [(row["monotonic_ns"] - row["elapsed_ns"], row["monotonic_ns"]) for row in gets]
    copy_spans = [(row["copy_begin_ns"], row["copy_end_ns"]) for row in copies]
    return {
        "physical_GET_calls": len(gets),
        "physical_GET_bytes": actual,
        "verification_GET_bytes": check_bytes,
        "verification_samples": len(checks),
        "native_equivalent_bytes_for_same_descriptors": native_bytes,
        "observed_reads_after_excluding_verification_bytes": observed_bytes,
        "byte_saving_on_same_descriptors_pct": 100 * (native_bytes - observed_bytes) / native_bytes if native_bytes else 0,
        "GET_window_union_s": interval_union(get_spans) / 1e9,
        "copy_window_union_s": interval_union(copy_spans) / 1e9,
        "GET_wall_ms": distribution(row["elapsed_ns"] / 1e6 for row in gets),
        "packed_path_decisions": dict(Counter(row["mode"] for row in decisions)),
        "packed_plan_ms": distribution(row["plan_ns"] / 1e6 for row in plans),
        "packed_preflight_wait_ms": distribution(row["wait_ns"] / 1e6 for row in plans),
        "copy_ms": distribution(row["elapsed_ns"] / 1e6 for row in copies),
        "copy_device_ms": distribution(row["device_elapsed_ms"] for row in copies),
        "copy_pair_wait_ms": distribution(row["pair_wait_ns"] / 1e6 for row in copies),
        "broadcast_operations": sum(row["operations"] for row in copies),
    }


def compare_prefixes(first, second):
    earliest_difference = Counter()
    equal = 0
    for key, left in first.items():
        right = second[key]
        equal += left["output_token_ids_sha256"] == right["output_token_ids_sha256"]
        a = left["output_token_prefix_sha256"]
        b = right["output_token_prefix_sha256"]
        first_difference = next((length for length in sorted(a.keys() & b.keys(), key=int)
                                 if a[length] != b[length]), None)
        earliest_difference[first_difference or "none_at_shared_checkpoints"] += 1
    return {"same_full_sequence": equal, "first_differing_saved_prefix_tokens": dict(earliest_difference)}


def decode_work_and_cadence(first, second):
    """Split observed elapsed differences across first token, EOS work, and generation cadence."""
    phases = ("before_first", "output_tokens", "residence_per_output_token")

    def components(profiles):
        result = {}
        for key, row in profiles.items():
            tokens = row["output_tokens"]
            post_first = row["phases_s"]["first_to_end"]
            before_first = sum(row["phases_s"].values()) - post_first
            result[key] = dict(zip(phases, (before_first, tokens, post_first / tokens), strict=True))
        return result

    a, b = components(first), components(second)
    finish_by_subset = {}
    savings = Counter()
    for size in range(len(phases) + 1):
        for subset in map(frozenset, combinations(phases, size)):
            mix = {phase: b if phase in subset else a for phase in phases}
            costs = {key: {"before_first": mix["before_first"][key]["before_first"],
                           "first_to_end": mix["output_tokens"][key]["output_tokens"]
                           * mix["residence_per_output_token"][key]["residence_per_output_token"]}
                     for key in first}
            _, done, _ = replay(first, costs)
            finish = finish_by_subset[subset] = max(done.values())
            for phase in subset:
                weight = 1 / (len(phases) * comb(len(phases) - 1, size - 1))
                savings[phase] += weight * (finish_by_subset[subset - {phase}] - finish)
    expected = max(row["end_s"] for row in first.values()) - max(row["end_s"] for row in second.values())
    if not isclose(sum(savings.values()), expected, abs_tol=1e-6):
        raise ValueError("Generation work and cadence do not reconcile with measured time")
    return dict(savings)


def analyze(folder):
    manifest = read_json(folder / "manifest.json")
    trace = folder / "r38-events.jsonl.gz"
    digest = sha256(trace)
    if digest != manifest["trace_sha256"]:
        raise ValueError("Trace does not match the archive manifest")
    workload = manifest["workload"]["request_sources"]
    cases, profiles, outputs = {}, {}, {}
    for arm, raw in read_trace(trace).items():
        expected = manifest["runs"][arm]
        origin, = (row["monotonic_s"] for row in raw["client"] if row["type"] == "client_origin")
        profiles[arm] = requests_for(raw, workload, origin)
        outputs[arm] = {row["id"]: row for row in raw["client"] if row["type"] == "request_end"}
        if any(row["output_token_ids_hashed"] != row["completion_tokens"] for row in outputs[arm].values()):
            raise ValueError(f"Incomplete output digests in {arm}")
        result = measure_arms(raw, profiles[arm], origin, expected["cap_Gbps"], {"P": 16, "D": 16})
        result["P_store"] = summarize_store(raw.get("P_method", []))
        decisions = result["P_store"]["packed_path_decisions"]
        if (expected["mode"] == "baseline" and decisions
                or expected["mode"] == "packed" and not decisions.get("packed")):
            raise ValueError(f"Store execution mode did not match the plan in {arm}")
        result["finish_reasons"] = dict(Counter(row["finish_reason"] for row in profiles[arm].values()))
        ttft = sorted(row["first_s"] - row["ready_s"] for row in profiles[arm].values())
        result["TTFT_s"] = {**distribution(ttft), "p99": ttft[ceil(len(ttft) * 0.99) - 1]}
        receipt = expected["receipt"]
        if (result["output_tokens"] != receipt["output_tokens"]
                or abs(result["throughput_tokens_s"] - receipt["output_tokens_per_s"]) > 1e-6):
            raise ValueError(f"Throughput did not match the measured receipt in {arm}")
        cases[arm] = result

    def pair(first_arm, second_arm):
        first, second = cases[first_arm], cases[second_arm]
        stages = compare_arms(profiles[first_arm], profiles[second_arm], first, second)
        tails = {case["observed_dependencies"]["tail"] for case in (first, second)}
        return {
            "a": first_arm,
            "b": second_arm,
            "throughput_change_pct": 100 * (second["throughput_tokens_s"] / first["throughput_tokens_s"] - 1),
            "elapsed_a_minus_b_s": stages["elapsed_100_minus_200_s"],
            "log_workload_term": stages["log_volume_term"],
            "log_elapsed_term": stages["log_elapsed_term"],
            "stage_swap_attribution_s": stages["stage_swap_attribution_s"],
            "decode_work_vs_cadence_s": decode_work_and_cadence(profiles[first_arm], profiles[second_arm]),
            "matched_route_requests": stages["matched_route_requests"],
            "matched_output_lengths": stages["matched_output_lengths"],
            "generation": compare_prefixes(outputs[first_arm], outputs[second_arm]),
            "tails": {key: {arm: profiles[arm][key] for arm in (first_arm, second_arm)} for key in sorted(tails)},
        }

    comparisons = {"packed_vs_native": {}, "native_200_vs_100": {}, "packed_200_vs_100": {}, "same_cap_repeats": {}}

    def compare_when_complete(group, label, a, b):
        if a in cases and b in cases:
            comparisons[group][label] = pair(a, b)

    for block in ("b1", "b2"):
        for cap in (100, 200):
            suffix = f"{block}-cap{cap}"
            compare_when_complete("packed_vs_native", suffix, f"native-{suffix}", f"packed-{suffix}")
        for mode in ("native", "packed"):
            compare_when_complete(f"{mode}_200_vs_100", block, f"{mode}-{block}-cap100", f"{mode}-{block}-cap200")
    for mode in ("native", "packed"):
        for cap in (100, 200):
            compare_when_complete("same_cap_repeats", f"{mode}-cap{cap}", f"{mode}-b1-cap{cap}", f"{mode}-b2-cap{cap}")
    return {"trace_sha256": digest, "cases": cases, **comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(args.directory))


if __name__ == "__main__":
    main()
