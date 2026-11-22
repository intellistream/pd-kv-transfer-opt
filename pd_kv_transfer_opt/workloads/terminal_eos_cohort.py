"""Freeze Terminal-Bench chains with consistently short, natural EOS in earlier runs."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.profile_closed_sessions import read_trace


MAX_OBSERVED_OUTPUT = 1024
MIN_OBSERVATIONS = 4
ROOTS_PER_SECOND = 8
IDENTITY_FIELDS = ("session_id", "predecessor", "prompt_sha256", "prompt_tokens", "max_tokens", "tool_gap_s")


def observed_outputs(archives, parent, parent_rows):
    observations = defaultdict(list)
    evidence = {}
    for directory in archives:
        manifest = read_json(directory / "manifest.json")
        workload = manifest["workload"]
        if workload["source_revision"] != parent["source_revision"] or workload["ignore_eos"]:
            raise ValueError(f"Historical arm has different input or stopping policy: {directory.name}")
        for source in workload["request_sources"]:
            original = parent_rows.get(source["id"])
            if original and any(source[field] != original[field] for field in IDENTITY_FIELDS):
                raise ValueError(f"Historical request differs from the parent: {source['id']}")
        trace_name = manifest.get("event_trace")
        if trace_name is None:
            trace, = directory.glob("*events.jsonl.gz")
        else:
            trace = directory / trace_name
        arms = read_trace(trace)
        evidence[directory.name] = {"trace_sha256": sha256(trace), "cases": list(arms)}
        for arm in arms.values():
            for row in arm["client"]:
                if row["type"] == "request_end" and row["id"] in parent_rows:
                    observed = (row.get("status"), row.get("finish_reason"), row.get("completion_tokens"))
                    observations[row["id"]].append(observed)
    return observations, evidence


def selection(parent_path, archives):
    parent = read_json(parent_path)
    parent_rows = {row["id"]: row for row in parent["request_sources"]}
    observations, evidence = observed_outputs(archives, parent, parent_rows)
    chains = defaultdict(list)
    for row in parent["request_sources"]:
        chains[row["session_id"]].append(row)
    selected = []
    for chain in chains.values():
        prefix = []
        for row in chain:
            history = observations[row["id"]]
            if len(history) < MIN_OBSERVATIONS or any(
                    state != "ok" or reason != "stop" or not 1 <= tokens <= MAX_OBSERVED_OUTPUT
                    for state, reason, tokens in history):
                break
            prefix.append(row)
        if len(prefix) >= 2:
            selected.append(prefix)
    selected.sort(key=lambda chain: chain[0]["offered_s"])
    if not selected:
        raise ValueError("Historical evidence supports no predecessor-closed chains")
    requests = [dict(row, original_offered_s=row["offered_s"], offered_s=i / ROOTS_PER_SECOND)
                for i, chain in enumerate(selected) for row in chain]
    fields = ("source_repo", "source_revision", "source_workload_sha256", "tokenizer_fingerprint", "ignore_eos",
              "temperature", "tool_gap_policy", "max_tokens_policy", "route_policy")
    return {**{name: parent[name] for name in fields}, "parent_r42_sha256": parent["workload_sha256"],
            "requests": len(requests), "sessions": len(selected), "last_session_start_s": (len(selected) - 1) / ROOTS_PER_SECOND,
            "arrival_policy": "Release roots in parent arrival order at 8/s, then use observed predecessor completion",
            "pressure_window_s": [15, 35],
            "selection": {"max_historical_eos_tokens": MAX_OBSERVED_OUTPUT, "minimum_observations": MIN_OBSERVATIONS,
                          "rule": "Longest predecessor-closed prefix of at least two requests; require every saved outcome to end at EOS.",
                          "historical_median_tokens_summed": sum(median(x[2] for x in observations[row["id"]]) for row in requests),
                          "evidence": evidence}, "request_sources": requests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    plan = selection(args.parent, args.archives)
    write_json(args.output, plan)
    print(f"Frozen {plan['sessions']} original sessions and {plan['requests']} original requests")


if __name__ == "__main__":
    main()
