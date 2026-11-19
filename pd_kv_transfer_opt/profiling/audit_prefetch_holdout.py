"""Screen original-session and matched-cohort traces for repeatable early-read choices."""

import argparse
from pathlib import Path

from pd_kv_transfer_opt.common import sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_cross_batch_choices import (
    STAGING_BYTES,
    free_windows,
    inspect,
    ms,
    observation_windows,
    restore_jobs,
)
from pd_kv_transfer_opt.profiling.analyze_cross_batch_prefetch import CASE, clock_margin_ns, index_restores, port_window
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import normalized_proxy_id, read_bundle
from pd_kv_transfer_opt.profiling.analyze_packed_readiness import client_ids, join_forwards, request_names
from pd_kv_transfer_opt.profiling.analyze_same_head_ab import cap_readback
from pd_kv_transfer_opt.profiling.audit_restore_worker_costs import get_case_nodes, group_case_nodes


def r18_natural(prefill, decode):
    batches, nodes, errors, unmatched = join_forwards(prefill)
    if errors or unmatched or len(batches) != len(nodes):
        raise ValueError("Incomplete r18 forwards")
    names = client_ids(decode)
    for node in nodes:
        node["rows"] = [(names[normalized_proxy_id(rid)], offset, tokens)
                        for rid, offset, tokens in node["batch"]["rows"]]
        node["names"] = {name for name, _, _ in node["rows"]}
        node["reads"] = {rank: dict(zip(request_names(load, names), load.reads, strict=True))
                         for rank, load in node["loads"].items()}
    return {"natural-sessions": nodes}


def r21_synchronous(prefill, decode):
    names = client_ids(decode)
    cases = {req["id"]: row["case"] for row in decode["client_manifest"] if row["phase"] == "target"
             and CASE.fullmatch(row["case"]) for req in row["request_sources"]}
    timed = {rid for rid, name in names.items() if name in cases}
    _, nodes, early = index_restores(prefill, timed)
    active = {cases[names[rid]] for _, _, rid in early if rid in timed}
    stock = {name: case for name, case in cases.items() if case not in active}
    return group_case_nodes(nodes, names, stock), sorted(active)


def r21_counters(case, node, prior, busy, prefill, decode):
    """Summarize D MAC counters inside the largest measured gap for each deadline."""
    batch = node["batch"]
    offset = batch["unix_ns"] - batch["monotonic_ns"]
    samples = sorted((row for row in decode["cap_port_sample"] if row["case"] == case),
                     key=lambda row: row["unix_ns"])
    cap = int(case.rsplit("cap", 1)[1])
    readback = cap_readback(decode, case, cap, list(range(8)))
    if readback is None:
        raise ValueError(f"No D shaping receipt for {case}")
    margin = clock_margin_ns(prefill, decode)
    result = {}
    for label, window in observation_windows(node, prior, restore_jobs(node)).items():
        gaps = list(free_windows(*window, busy))
        if not gaps:
            continue
        left, right = max(gaps, key=lambda item: item[1] - item[0])
        counters = port_window(samples, left + offset, right + offset, readback, margin)
        rows = counters["ports"]
        result[label] = {
            "clock_margin_ms": counters["clock_margin_ms"],
            "ports_with_interval": counters["ports_with_interval"],
            "covered_ms_by_port": {port: sum(bin_["coverage_ms"] for bin_ in row["bins"]) for port, row in rows.items()},
            "tx_bytes_by_port": {port: sum(bin_["tx_bytes"] for bin_ in row["bins"]) for port, row in rows.items()},
            "nominal_unused_capacity_bytes_by_port": {
                port: row.get("nominal_unused_capacity_bytes") for port, row in rows.items()},
        }
    return result


def screen(run, prefill, decode, cases):
    result = []
    for case, nodes in sorted(cases.items()):
        ends = [row for row in decode["client"] if row["type"] == "request_end" and row["case"] == case]
        if run != "r18" and (len(ends) != 10 or any(row["status"] != "ok" for row in ends)):
            raise ValueError(f"Missing or unsuccessful formal targets in {case}")
        tails = {"ttft": max(ends, key=lambda row: row["first_s"] - row["ready_s"])["id"],
                 "drain": max(ends, key=lambda row: row["end_s"])["id"]}
        busy = sorted((r.begin, r.end) for n in nodes for rank in n["reads"].values() for r in rank.values())
        prior_by_instance = {}
        for node in sorted(nodes, key=lambda row: row["batch"]["dispatch_ns"]):
            instance = node["batch"]["instance"]
            prior = prior_by_instance.get(instance)
            prior_by_instance[instance] = node
            if len(node["reads"].get(0, ())) < 2:
                continue
            if prior is None or any(load.verified for load in node["loads"].values()):
                raise ValueError(f"Unusable multi-restore observation in {run}/{case}")
            row = inspect(case, node, prior, busy, tails, cap=1600 if run == "r18" else None)
            row["stock_last_forward_rank"] = max(
                node["forwards"], key=lambda rank: node["forwards"][rank]["forward_enter_ns"])
            for job, (_, reads) in zip(row["jobs"], restore_jobs(node), strict=True):
                first = min(read.begin for read in reads)
                job["stock_get_by_rank"] = [
                    {"launch_offset_ms": ms(read.begin - first), "duration_ms": ms(read.end - read.begin)}
                    for read in reads
                ]
            row["predecessor_rank0_device_ms"] = prior["batch"].get("device", {}).get("forward_device_ms")
            row["predecessor_dispatch_to_result_ms"] = (
                prior["batch"]["ready_ns"] - prior["batch"]["dispatch_ns"]) / 1e6
            row["replay_sensitivity"] = [p for p in row["replay_sensitivity"] if p["slot_bytes_per_rank"] == STAGING_BYTES]
            if run == "r18":
                row["contains_TTFT_tail_chunk"] = row["contains_drain_tail_chunk"] = None
            if run.startswith("r21"):
                row["D_port_gap_counters"] = r21_counters(case, node, prior, busy, prefill, decode)
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("r18_directory", "r19_directory", "r21_directory", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    report = {}
    inputs = (("r18", args.r18_directory, "r18"), ("r19", args.r19_directory, "r19"),
              ("r21", args.r21_directory, "r21"), ("r21_earlier", args.r21_directory, "baseline"))
    for run, directory, stem in inputs:
        paths = [directory / f"{stem}-{role}.jsonl.gz" for role in "PD"]
        prefill, decode = map(read_bundle, paths)
        if prefill["scope"][0]["run_id"] != decode["scope"][0]["run_id"]:
            raise ValueError(f"Expected a matching trace pair for {run}")
        excluded = []
        if run == "r18":
            cases = r18_natural(prefill, decode)
        elif run == "r19":
            cases = {case: nodes for case, nodes in get_case_nodes(prefill, decode).items() if case.startswith("b-")}
        else:
            cases, excluded = r21_synchronous(prefill, decode)
        report[run] = {
            "source_sha256": {role: sha256(path) for role, path in zip("PD", paths, strict=True)},
            "cases": len(cases), "traced_batches": sum(map(len, cases.values())), "excluded_active_early_cases": excluded,
            "multi_restore": screen(run, prefill, decode, cases),
        }
    write_json(args.output, {
        "scope": "Frozen synchronous paths with fixed bytes, releases, staging and publish order. "
                 "The r18 cap is the sum of configured port rates, not measured available bandwidth. "
                 "Slot eligibility uses payload, before any alignment padding. D MAC bins report observed "
                 "traffic; they do not identify destination leases, directional source placement, "
                 "predecessor interference or deliverable per-read credit.",
        "runs": report,
    })


if __name__ == "__main__":
    main()
