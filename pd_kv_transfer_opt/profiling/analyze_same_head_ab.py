"""Compare named fixed cohorts from baseline and same-head experiment event bundles."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from pd_kv_transfer_opt.common import write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import (
    classify_layout,
    normalized_proxy_id,
    read_bundle,
)
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import stats
from pd_kv_transfer_opt.profiling.analyze_rdma_hard_caps import case_gets, port_rates


GET = "batch_get_into_multi_buffers"
COPY = "same_head_local_copy"
TX = "mac_tx_total_oct_num"
RX = "mac_rx_total_oct_num"
PROTOCOL = ("workload_sha256", "prefix_ids", "prompt_lengths", "offered_rate",
            "selected_requests", "output_tokens", "aggregate_D_egress_cap_Gbps")
LAYOUT_FIELDS = ("raw_segments", "layout_boundaries", "local_only_breaks", "peer_only_breaks",
                 "both_breaks", "contiguous_both")


def index(rows, field):
    indexed = {row[field]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate {field} in trace")
    return indexed


def load_arm(prefill, decode):
    return arm_data(read_bundle(prefill), read_bundle(decode))


def arm_data(p, d):
    p_scope, = p["scope"]
    d_scope, = d["scope"]
    if p_scope["role"] != "P" or d_scope["role"] != "D" or p_scope["run_id"] != d_scope["run_id"]:
        raise ValueError("Expected a P and D bundle from one run")
    manifests = index(d["client_manifest"], "case")
    cases = {name: row for name, row in manifests.items()
             if row["aggregate_D_egress_cap_Gbps"] and not row.get("profiler_enabled")}
    return p_scope["run_id"], sorted(p["native"], key=lambda row: row["monotonic_ns"]), d, cases


def select_cases(decode, prefix):
    """Remove the arm prefix from case labels while preserving request IDs."""
    selected = decode.copy()
    for kind in ("client_manifest", "client", "client_summary", "cap_port_sample", "cap_control"):
        field = "name" if kind == "cap_control" else "case"
        selected[kind] = [
            dict(row, **{field: row[field].removeprefix(prefix)})
            for row in decode[kind] if row[field].startswith(prefix)
        ]
    return selected


def port_deltas(samples):
    if len(samples) < 2:
        raise ValueError("At least two D port samples are required")
    begin = index(samples[0]["ports"], "port")
    end = index(samples[-1]["ports"], "port")
    if len(begin) != 8 or begin.keys() != end.keys():
        raise ValueError("Unexpected D port set")
    rates = port_rates(samples)
    by_port = {}
    for port, first in begin.items():
        last = end[port]
        counters = {key: last["counters"][key] - first["counters"][key] for key in (TX, RX)}
        by_port[port] = {"tx_bytes": counters[TX], "rx_bytes": counters[RX],
                         "sample_window_s": (last["monotonic_ns"] - first["monotonic_ns"]) / 1e9}
    return by_port, rates["D_tx_Gbps"]["max"]


def copy_metrics(events, proxy_ids, copy_event):
    def matched_ids(row):
        ids = row.get("request_ids") or [row.get("request_id")]
        return {(normalized_proxy_id(rid) or rid) for rid in ids} & proxy_ids.keys()

    copies = []
    known_ids = set()
    for row in events:
        if row["type"] == copy_event and (ids := matched_ids(row)):
            copies.append(row)
            known_ids.update(ids)
    return {"events": len(copies), "requests_with_events": len(known_ids),
            "method_counts": dict(Counter(row["method"] for row in copies if "method" in row)),
            "reported_payload_bytes": sum(row["payload_bytes"] for row in copies),
            "elapsed_ms": stats(row["elapsed_ns"] / 1e6 for row in copies),
            "device_elapsed_ms": stats(row["device_elapsed_ms"] for row in copies
                                       if row.get("device_elapsed_ms") is not None),
            "pair_wait_ms": stats(row["pair_wait_ns"] / 1e6 for row in copies
                                  if row.get("pair_wait_ns") is not None),
            "broadcast_operations": stats(row["operations"] for row in copies
                                          if row.get("operations") is not None)}


def layout_profile(prefill, decode):
    """Join each layout breakdown to the preceding copy on the same worker."""
    clients = {row["id"]: row["case"] for row in decode["client"]
               if row["type"] == "request_end" and row["status"] == "ok" and row["case"].startswith("layout-")}
    if not clients:
        return {}
    proxies = {row["proxy_request_id"]: clients[row["id"]] for row in decode["proxy"]
               if row["type"] == "prefill_begin" and row["id"] in clients}
    copies, rows = {}, defaultdict(list)
    for row in sorted(prefill["native"], key=lambda item: item["monotonic_ns"]):
        if row["type"] not in (COPY, "same_head_copy_layout"):
            continue
        ids = tuple(row["request_ids"])
        worker_batch = row["pid"], ids
        if row["type"] == COPY:
            copies[worker_batch] = row
            continue
        cases = {proxies[req] for raw in ids if (req := normalized_proxy_id(raw)) in proxies}
        if len(cases) != 1:
            raise ValueError("A layout event must belong to a single diagnostic case")
        case, = cases
        copy = copies[worker_batch]
        cuts = sum(row[field] for field in LAYOUT_FIELDS[1:-1])
        if cuts != copy["operations"] or cuts + row["contiguous_both"] != row["raw_segments"]:
            raise ValueError(f"Inconsistent page gap and broadcast counts in {case}")
        rows[case].append(dict(row, broadcasts=cuts, copy_ms=copy["elapsed_ns"] / 1e6,
                               address_gap_breaks=cuts - row["layout_boundaries"]))
    output = {}
    for case, events in rows.items():
        requests = sum(name == case for name in clients.values())
        if len(events) != 4 * requests:
            raise ValueError(f"Missing rank observations in {case}")
        output[case] = {
            "requests": requests, "rank_events": len(events),
            **{field: stats(row[field] for row in events)
               for field in (*LAYOUT_FIELDS, "broadcasts", "copy_ms", "address_gap_breaks")},
        }
    return output


def cap_readback(decode, name, limit, ports):
    control = {row["name"]: row["data"] for row in decode["cap_control"]}
    receipt = control.get(f"{name}-shaping.json")
    if receipt is None:
        return None
    configured = {str(port): row["current_Mbps"] for port, row in receipt["readback"].items()}
    if (configured.keys() != {str(port) for port in ports} or receipt["requested_aggregate_Gbps"] != limit
            or sum(configured.values()) != limit * 1000):
        raise ValueError(f"D cap readback does not match manifest: {name}")
    return configured


def sync_load_windows(events):
    """Measure first worker entry to last worker exit for each restored request."""
    started = {}
    by_request = defaultdict(list)
    for row in events:
        kind = row["type"]
        worker = row["pid"], row["thread_id"]
        if kind == "worker_load_begin":
            started[worker] = row["monotonic_ns"]
        elif kind == "worker_load_end":
            begin = started.pop(worker)
            for request in row["request_ids"]:
                by_request[normalized_proxy_id(request)].append((worker, begin, row["monotonic_ns"]))

    worker_counts = Counter()
    elapsed_ms = []
    for spans in by_request.values():
        worker_counts[len({worker for worker, _, _ in spans})] += 1
        elapsed_ms.append((max(end for _, _, end in spans) - min(begin for _, begin, _ in spans)) / 1e6)
    return {"worker_counts": dict(sorted(worker_counts.items())), "elapsed_ms": stats(elapsed_ms)}


def analyze_case(p_events, decode, manifest, copy_event):
    name = manifest["case"]
    requests = manifest["selected_requests"]
    summary, = [row for row in decode["client_summary"] if row["case"] == name]
    if summary["counts"] != {"ok": requests} or summary["quality"]["errors"] or summary["fixed_output_deviations"]:
        raise ValueError(f"Incomplete or invalid cohort: {name}")
    ends = index([row for row in decode["client"] if row["case"] == name and row["type"] == "request_end"], "id")
    proxy_ids = {row["proxy_request_id"]: row["id"] for row in decode["proxy"]
                 if row["type"] == "prefill_begin" and row["id"] in ends}
    if len(ends) != requests or len(proxy_ids) != requests or set(proxy_ids.values()) != set(ends):
        raise ValueError(f"Run has incomplete client/proxy joins: {name}")
    selected = case_gets(p_events, proxy_ids)["native"]
    gets = [row for row in selected if row["type"] == "store_end" and row["operation"] == GET]
    if not gets or any(row["negative_returns"] for row in gets):
        raise ValueError(f"Missing or failed native GETs: {name}")
    observed_groups = [classify_layout(row) for row in gets]
    classified = [group for group in observed_groups if group is not None]
    ports, peak = port_deltas([row for row in decode["cap_port_sample"] if row["case"] == name])
    configured = cap_readback(decode, name, manifest["aggregate_D_egress_cap_Gbps"], ports)
    pd_bytes = sum(row["payload_bytes"] for row in decode["native"]
                   if row["type"] == "native_transfer_submit"
                   and normalized_proxy_id(row.get("remote_request_id")) in proxy_ids)
    finite = summary["finite_cohort"]
    drain = finite["drain_included_s"]
    tokens = sum(row["completion_tokens"] for row in ends.values())
    if tokens != requests * manifest["output_tokens"]:
        raise ValueError(f"Unexpected output token count: {name}")
    return {
        "requests": requests, "drain_s": drain, "requests_s": requests / drain,
        "output_tokens_s": tokens / drain, "ttft_p99_s": finite["ttft_from_ready"]["p99_s"],
        "get_calls": len(gets), "get_payload_bytes": sum(row["payload_bytes"] for row in gets),
        "get_elapsed_ms": stats(row["elapsed_ns"] / 1e6 for row in gets),
        "get_layout": {"classified_calls": len(classified), "attention_bytes": sum(group[0] for group in classified),
                       "align_state_bytes": sum(group[1] for group in classified)},
        "sync_load_window": sync_load_windows(selected),
        "local_copy": copy_metrics(p_events, proxy_ids, copy_event),
        "P_to_D_descriptor_bytes": pd_bytes,
        "D_mac_tx_bytes": sum(row["tx_bytes"] for row in ports.values()), "D_tx_sample_peak_Gbps": peak,
        "D_ports": ports, "D_port_readback_Mbps": configured,
    }


def delta_percent(baseline, prototype):
    return 100 * (prototype - baseline) / baseline if baseline else None


def deltas(baseline, prototype):
    fields = ("get_payload_bytes", "D_mac_tx_bytes", "requests_s", "output_tokens_s", "ttft_p99_s")
    return {key: delta_percent(baseline[key], prototype[key]) for key in fields}


def compare(baseline, prototype, selected_cases=None, copy_event=COPY):
    return compare_arms(load_arm(*baseline), load_arm(*prototype), selected_cases, copy_event)


def compare_arms(baseline, prototype, selected_cases=None, copy_event=COPY):
    a_id, a_p, a_d, a_cases = baseline
    b_id, b_p, b_d, b_cases = prototype
    chosen = sorted(a_cases.keys() & b_cases.keys()) if selected_cases is None else selected_cases
    if not chosen or any(name not in a_cases or name not in b_cases for name in chosen):
        raise ValueError("Each selected shaped case must occur in both arms")
    if len(chosen) != len(set(chosen)):
        raise ValueError("Duplicate selected case")

    pairs = {}
    groups = defaultdict(list)
    for name in chosen:
        a_manifest, b_manifest = a_cases[name], b_cases[name]
        if any(a_manifest.get(k) != b_manifest.get(k) for k in PROTOCOL):
            raise ValueError(f"Mismatch between the baseline and prototype protocol for {name}")
        a, b = (analyze_case(p, d, manifest, copy_event)
                for p, d, manifest in [(a_p, a_d, a_manifest), (b_p, b_d, b_manifest)])
        if a["D_ports"].keys() != b["D_ports"].keys():
            raise ValueError(f"Different D ports for {name}")
        if a["D_port_readback_Mbps"] and b["D_port_readback_Mbps"] and a["D_port_readback_Mbps"] != b["D_port_readback_Mbps"]:
            raise ValueError(f"Different D port caps for {name}")
        pairs[name] = {"baseline": a, "prototype": b, "prototype_minus_baseline_percent": deltas(a, b),
                       "D_port_delta_bytes": {port: {key: b["D_ports"][port][key] - row[key]
                                             for key in ("tx_bytes", "rx_bytes")}
                                              for port, row in a["D_ports"].items()}}
        groups[(a_manifest["aggregate_D_egress_cap_Gbps"], a_manifest["offered_rate"], a["requests"])].append(name)

    by_condition = []
    for (cap, rate, requests), names in sorted(groups.items()):
        row = {"D_egress_cap_Gbps": cap, "offered_requests_s": rate, "requests_per_round": requests,
               "cases": names}
        for arm in ("baseline", "prototype"):
            rounds = [pairs[name][arm] for name in names]
            row[arm] = {
                "requests_s": mean(r["requests_s"] for r in rounds),
                "output_tokens_s": mean(r["output_tokens_s"] for r in rounds),
                "ttft_p99_s": median(r["ttft_p99_s"] for r in rounds),
                "get_payload_bytes": median(r["get_payload_bytes"] for r in rounds),
                "D_mac_tx_bytes": median(r["D_mac_tx_bytes"] for r in rounds),
            }
        row["prototype_minus_baseline_percent"] = deltas(row["baseline"], row["prototype"])
        by_condition.append(row)
    return {"baseline_run_id": a_id, "prototype_run_id": b_id, "paired_cases": pairs,
            "baseline_cases_excluded": sorted(a_cases.keys() - set(chosen)),
            "prototype_cases_excluded": sorted(b_cases.keys() - set(chosen)),
            "by_condition": by_condition}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_p", type=Path)
    parser.add_argument("baseline_d", type=Path)
    parser.add_argument("prototype_p", type=Path, nargs="?")
    parser.add_argument("prototype_d", type=Path, nargs="?")
    parser.add_argument("--interleaved", action="store_true", help="Read a- and b- cases from a single P/D bundle pair.")
    parser.add_argument("--case", action="append", help="Compare a specific case name in both arms. Repeat for more.")
    parser.add_argument("--copy-event", default=COPY, help="Name of the request-tagged P local-copy observation.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.interleaved:
        if args.prototype_p or args.prototype_d:
            parser.error("--interleaved takes one P/D bundle pair")
        p, d = read_bundle(args.baseline_p), read_bundle(args.baseline_d)
        a, b = (arm_data(p, select_cases(d, prefix)) for prefix in ("a-", "b-"))
        result = compare_arms(a, b, args.case, args.copy_event)
        if profiles := layout_profile(p, d):
            result["layout_profile"] = profiles
    else:
        if not (args.prototype_p and args.prototype_d):
            parser.error("Provide both P/D bundle pairs or use --interleaved")
        result = compare((args.baseline_p, args.baseline_d), (args.prototype_p, args.prototype_d), args.case, args.copy_event)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
