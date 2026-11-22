"""Account for original requests during a same-rate workload expansion."""

import argparse
from pathlib import Path
from statistics import median

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.profile_closed_sessions import read_trace, requests_for


def archive(path):
    manifest = read_json(path / "manifest.json")
    trace = path / manifest["event_trace"]
    if sha256(trace) != manifest["artifacts"][trace.name]["sha256"]:
        raise ValueError("Cohort archive trace checksum mismatch")
    return manifest, read_trace(trace), read_json(path / "summary.json")["arms"]


def describe(rows, shared, arm):
    roots = [r for k, r in rows.items() if k in shared and r["predecessor"] is None]
    followers = [r for k, r in rows.items() if k in shared and r["predecessor"] is not None]
    return {"total_throughput_tokens_s": arm["throughput_tokens_s"], "total_output_tokens": arm["output_tokens"],
            "total_time_s": arm["duration_s"], "requests": len(rows),
            "P_compute_tokens": arm["P_prompt_token_sources"]["local_compute"],
            "P_external_restore_tokens": arm["P_prompt_token_sources"]["external_kv_transfer"],
            "D_external_handoff_tokens": arm["D_prompt_token_sources"]["external_kv_transfer"],
            "P_waiting_peak": [arm["P_D_resource_samples"][role]["peak_waiting"] for role in ("P0", "P1")],
            "P_TX_Gbps_30_90_s": arm["P_egress"]["primary_window"]["P_MAC_TX_Gbps"],
            "D_TX_Gbps_30_90_s": arm["observed_arrival_window"]["D_MAC_TX_Gbps"],
            "common_root_P_HTTP_p50_s": median(r["phases_s"]["P_HTTP"] for r in roots),
            "common_successor_P_HTTP_p50_s": median(r["phases_s"]["P_HTTP"] for r in followers),
            "common_root_replies_by_s": {t: sum(r["P_end_s"] < t for r in roots) for t in (30, 60, 90, 120)},
            "common_successors_ready_by_s": {t: sum(r["ready_s"] < t for r in followers) for t in (30, 60, 90, 120)},
            "store_counters": arm["store_counter_delta"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("small", type=Path)
    parser.add_argument("large", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    inputs = [archive(path) for path in (args.small, args.large)]
    plan = [{r["id"]: r for r in m["workload"]["request_sources"]} for m, _, _ in inputs]
    if not plan[0].keys() < plan[1].keys():
        raise ValueError("Require a strict same-origin workload expansion")
    fields = ("session_id", "prompt_sha256", "prompt_tokens", "predecessor", "tool_gap_s", "max_tokens", "offered_s")
    if any(any(spec[k] != plan[1][rid][k] for k in fields) for rid, spec in plan[0].items()):
        raise ValueError("An original request changed")
    arms = {}
    for case in inputs[1][0]["case_order"]:
        pair = {}
        for (manifest, trace, report), specs in zip(inputs, plan, strict=True):
            origin, = [r["monotonic_s"] for r in trace[case]["client"] if r["type"] == "client_origin"]
            rows = requests_for(trace[case], [*specs.values()], origin)
            pair[manifest["run_id"]] = describe(rows, plan[0].keys(), report[case])
        arms[case] = pair
    write_json(args.output, {"unchanged_original_requests": len(plan[0]), "arms": arms,
                            "scope": "Compare the larger offered workload with the smaller one. Routes, actual EOS, "
                                     "run length, and cache state evolve naturally in each run."})


if __name__ == "__main__":
    main()
