"""Compare drained native serving throughput and separate D graph diagnostics."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.profile_closed_sessions import compare_arms, measure_arms, read_trace, requests_for


def diagnostic_interval(row):
    """Use the same emitted token interval before starting the native profiler."""
    emitted = 0
    crossings = {}
    for event in row["chunks"]:
        emitted += event["n_tokens"]
        for boundary in (32, 104):
            if boundary not in crossings and emitted >= boundary:
                crossings[boundary] = (emitted, event["at_s"])
    left, right = (crossings[n] for n in (32, 104))
    return (right[1] - left[1]) * 1000 / (right[0] - left[0])


def decode_capacity_samples(telemetry):
    values = defaultdict(list)
    metrics = {"vllm:kv_cache_usage_perc": "KV_used_fraction", "vllm:num_preemptions_total": "preemptions"}
    for row in telemetry:
        for engine in row.get("engines", ()):
            if engine["role"] not in ("D0", "D1"):
                continue
            for name, value in engine.get("values", {}).items():
                metric = metrics.get(name.partition("{")[0])
                if metric is not None:
                    values[engine["role"], metric].append(value)
    return {role: {metric: {"samples": len(samples), "max": max(samples)}
                   for (label, metric), samples in values.items() if label == role}
            for role in ("D0", "D1")}


def analyze(directory):
    manifest = read_json(directory / "manifest.json")
    for name, entry in manifest["files"].items():
        if sha256(directory / name) != entry["sha256"]:
            raise ValueError(f"Unexpected digest for {name}")
    trace = directory / "r35-events.jsonl.gz"
    observed, requests, devices, diagnostic = {}, {}, {}, {}
    for name, records in sorted(read_trace(trace).items()):
        origin, = (row["monotonic_s"] for row in records["client"] if row["type"] == "client_origin")
        request = requests_for(records, manifest["workload"]["request_sources"], origin)
        requests[name] = request
        observed[name] = measure_arms(records, request, origin, 100, {"P": 16, "D": 16})
        observed[name]["D_capacity_samples"] = decode_capacity_samples(records["telemetry"])
        observed[name]["finish_reasons"] = dict(Counter(row["finish_reason"] for row in request.values()))
        devices[name] = read_json(directory / f"{name}-graph-steps.json")
        diagnostic[name] = read_json(directory / f"{name}-diagnostic.json")
    comparable = compare_arms(requests["a"], requests["b"], observed["a"], observed["b"])
    effects = {key: comparable[key] for key in (
        "log_throughput_ratio", "log_volume_term", "log_elapsed_term",
        "stage_swap_attribution_s", "matched_route_requests", "matched_output_lengths",
        "matched_route_but_different_output_length")}
    effects["elapsed_a_minus_b_s"] = comparable["elapsed_100_minus_200_s"]
    effects["largest_token_volume_changes"] = [
        {"id": row["id"], "a_tokens": row["tokens_100"], "b_tokens": row["tokens_200"]}
        for row in comparable["largest_token_volume_changes"]]
    tail_requests = {key for arm in observed.values() for key in arm["observed_dependencies"]["tail_chain"]}
    effects["paired_tail_requests"] = {key: {arm: rows[key] for arm, rows in requests.items()}
                                       for key in sorted(tail_requests)}
    effects["throughput_gain_pct"] = 100 * (observed["b"]["throughput_tokens_s"] /
                                            observed["a"]["throughput_tokens_s"] - 1)
    measures = sorted(devices["a"]["costs_ms"].keys() & devices["b"]["costs_ms"].keys())
    return {"arms": observed, "comparison": effects, "diagnostic": {
        "same_output_token_ids": diagnostic["a"]["token_ids_sha256"] == diagnostic["b"]["token_ids_sha256"],
        "pre_capture_client_ms_per_token": {name: diagnostic_interval(row) for name, row in diagnostic.items()},
        "matched_graph_steps": {name: row["aligned_successor_steps"] for name, row in devices.items()},
        "first_allreduce_in_graph_rank_steps": {name: row["first_allreduce_in_graph_rank_steps"]
                                                for name, row in devices.items()},
        "p50_ms": {k: {arm: devices[arm]["costs_ms"][k]["p50"] for arm in ("a", "b")} for k in measures},
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, analyze(args.directory))


if __name__ == "__main__":
    main()
