"""Quantify application batching and fixed-submit handoff slack in the archived r27c trace."""

import argparse
from collections import Counter
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_full_rdma import associate_loads, associate_transfers
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution, normalized_proxy_id, read_bundle
from pd_kv_transfer_opt.profiling.screen_transport_limits import handoff_timing


def store_groups(worker_calls, native_ids):
    selected = [row for row in worker_calls if row["proxy_ids"] and all(key in native_ids for key in row["proxy_ids"])]
    groups = Counter(len(row["proxy_ids"]) for row in selected)
    if sum(jobs * count for jobs, count in groups.items()) != len(native_ids) * 4:
        raise ValueError("Lost a worker-rank GET")
    return {
        "worker_rank_batches": len(selected),
        "worker_rank_batches_with_multiple_requests": len(selected) - groups[1],
        "requests_per_batch_histogram": dict(sorted(groups.items())),
        "request_level_GET_calls": len(native_ids) * 4,
        "potential_calls_with_one_GET_per_rank_batch": len(selected),
        "sum_native_GET_ms_per_batch": distribution(row["native_GET_sum_ms"] for row in selected),
        "worker_ms_outside_native_GET_per_batch": distribution(row["worker_other_ms"] for row in selected),
    }


def audit(directory):
    summary = read_json(directory / "summary.json")
    paths = {role: directory / f"r27c-{role}.jsonl.gz" for role in ("P", "D")}
    digests = {role: sha256(path) for role, path in paths.items()}
    if digests != {role: summary["source_sha256"][role] for role in paths}:
        raise ValueError("Native events differ from the r27c summary inputs")
    p, d = (read_bundle(paths[role]) for role in ("P", "D"))
    _, worker_calls, errors = associate_loads(p["native"])
    if errors:
        raise ValueError("P archive has unmatched GETs")
    handoffs = associate_transfers(d["native"])
    proxy = {row["id"]: row["proxy_request_id"] for row in d["proxy"] if row["type"] == "prefill_begin"}
    descriptor_submits = [row for row in d["native"] if row["type"] == "native_transfer_submit"]

    result = {}
    for case, measured in summary["cases"].items():
        requests = measured["per_request"]
        names = {key: proxy[f"{case}~{key}"] for key in requests}
        native_ids = set(names.values())
        vector_histogram = Counter()
        submissions = [row for row in descriptor_submits
                       if normalized_proxy_id(row["remote_request_id"]) in native_ids]
        if len(submissions) != len(requests) * 4:
            raise ValueError(f"Lost a P→D descriptor submission in {case}")
        for row in submissions:
            shape = {int(size): count for size, count in row["descriptor_size_histogram"].items()}
            if sum(shape.values()) != row["descriptor_count"]:
                raise ValueError("Incomplete descriptor histogram")
            vector_histogram.update(shape)
        result[case] = {
            "P_Store": store_groups(worker_calls, native_ids),
            "P_to_D": {
                "native_transfer_calls": len(submissions),
                "descriptors_per_submission": distribution(row["descriptor_count"] for row in submissions),
                "payload_bytes_per_submission": distribution(row["payload_bytes"] for row in submissions),
                "descriptor_size_histogram": dict(vector_histogram.most_common()),
                "fixed_admission_timing": handoff_timing(requests, {key: handoffs[pid] for key, pid in names.items()}),
            },
        }
    return {"source_sha256": digests, "cases": result}


def audit_apc_restore(directory):
    manifest = read_json(directory / "manifest.json")
    paths = {role: directory / f"r19-{role}.jsonl.gz" for role in ("P", "D")}
    digests = {role: sha256(path) for role, path in paths.items()}
    if digests != {role: manifest["artifacts"][path.name]["sha256"] for role, path in paths.items()}:
        raise ValueError("Native events differ from the r19 archive inputs")
    p, d = (read_bundle(paths[role]) for role in ("P", "D"))
    _, worker_calls, _ = associate_loads(p["native"])
    proxy = {row["id"]: row["proxy_request_id"] for row in d["proxy"] if row["type"] == "prefill_begin"}
    cases = {}
    for row in d["client_manifest"]:
        if row["case"] != "warm" and row.get("phase") == "target" and row.get("arm_mode") == "baseline":
            cases[row["case"]] = store_groups(worker_calls, {proxy[item["id"]] for item in row["request_sources"]})
    if len(cases) != 4:
        raise ValueError("Expected four r19 native test arms")
    return {"source_sha256": digests, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--apc-high-restore", type=Path, help="Optional r19 archive with APC-enabled native arms")
    args = parser.parse_args()
    report = {"r27c": audit(args.directory)}
    if args.apc_high_restore:
        report["r19"] = audit_apc_restore(args.apc_high_restore)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
