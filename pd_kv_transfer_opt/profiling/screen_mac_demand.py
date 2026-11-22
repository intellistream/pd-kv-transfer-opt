"""Replay sampled 200 Gbps wire rates against candidate per-port limits."""

import argparse
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_multiwave import aligned_tx_intervals
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, read_trace, requests_for


def align_prefill(samples, origin):
    decode_offset = origin["unix_ns"] - origin["monotonic_ns"]
    result = []
    for sample in samples:
        offset = sample["unix_ns"] - sample["monotonic_ns"] - decode_offset
        result.append({"ports": [{**p, "monotonic_ns": p["monotonic_ns"] + offset} for p in sample["ports"]]})
    return result


def replay_port(intervals, port, line, stop):
    backlog = peak = 0.0
    waiting = []
    for left, right, _, rates in intervals:
        left, right = max(0, left), min(stop, right)
        if left >= right:
            continue
        rate = rates[port]
        exposed_until = right if rate >= line else min(right, left + backlog / (line - rate))
        if (backlog > 0 or rate > line) and exposed_until > left:
            if waiting and left == waiting[-1][1]:
                waiting[-1] = (waiting[-1][0], exposed_until)
            else:
                waiting.append((left, exposed_until))
        backlog = max(0, backlog + (right - left) * (rate - line))
        peak = max(peak, backlog)
    return {"port": port, "peak_queued_GB": peak / 8, "peak_single_port_queue_delay_s": peak / line,
            "queued_s": sum(right - left for left, right in waiting),
            "longest_queue_s": max((right - left for left, right in waiting), default=0)}, waiting


def read_flow_control(samples, origin, stop):
    rows = [row for row in samples if origin <= row["unix_ns"] <= stop]
    if len(rows) < 2:
        return None
    before, after = rows[0], rows[-1]
    ports = []
    for first, last in zip(before["ports"], after["ports"], strict=True):
        if first["port"] != last["port"]:
            raise ValueError("Port order changed during the capture")
        ports.append({"port": first["port"], "counts": {key: last["counters"][key] - first["counters"][key]
                      for key in first["counters"].keys() & last["counters"].keys() if not key.endswith("_oct_num")}})
    return {"covered_s": (after["unix_ns"] - before["unix_ns"]) / NS, "ports": ports}


def analyze(directory, effective_caps=(80, 100)):
    if any(not 0 < cap for cap in effective_caps):
        raise ValueError("Effective egress rates must be positive")
    manifest = read_json(directory / "manifest.json")
    path = directory / manifest["event_trace"]
    digest = sha256(path)
    if manifest["artifacts"][path.name]["sha256"] != digest:
        raise ValueError("Trace checksum mismatch")
    formal = read_json(directory / "summary.json")["arms"]
    arms = {}
    for case, arm in read_trace(path).items():
        report = formal[case]
        end = report["duration_s"]
        origin, = [row["anchor"] for row in arm["client"] if row["type"] == "client_origin"]
        source = {"P": arm["P-telemetry"], "D": arm["telemetry"]}
        flow_control = {role: read_flow_control(samples, origin["unix_ns"], origin["unix_ns"] + end * NS)
                        for role, samples in source.items()}
        result = {"formal_Gbps_each_direction": report["cap_Gbps"], "observed_total_tokens_s": report["throughput_tokens_s"],
                  "flow_control": flow_control}
        if report["cap_Gbps"] == 200:
            profiles = requests_for(arm, manifest["workload"]["request_sources"], origin["monotonic_ns"] / NS)
            tail = [profiles[key] for key in report["observed_dependencies"]["tail_chain"]]
            source["P"] = align_prefill(source["P"], origin)
            for role, samples in source.items():
                intervals = list(aligned_tx_intervals(samples, origin["monotonic_ns"]))
                probes = {}
                for cap in effective_caps:
                    rows = [replay_port(intervals, port, cap / 8, end) for port in range(8)]
                    phase = ("D_start_s", "first_s") if role == "P" else ("P_start_s", "P_end_s")
                    probes[cap] = {"per_port": [row for row, _ in rows],
                                   "tail_requests_with_any_port_queue_overlap": sum(
                                       any(max(row[phase[0]], left) < min(row[phase[1]], right)
                                           for _, waiting in rows for left, right in waiting) for row in tail)}
                result[f"{role}_replayed_egress"] = probes
        arms[case] = result
    return {"trace_sha256": digest, "arms": arms,
            "scope": "Rate samples from actual 200 Gbps departures are used as a piecewise-constant arrival proxy. "
                     "Per-port independent fluid FIFO. No inferred exact per-request traffic, or queue/routing feedback. "
                     "Flow-control frames counted in the timing window do not identify pause duration."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--effective-caps", type=float, nargs="+", default=(80, 100))
    args = parser.parse_args()
    write_json(args.output, analyze(args.archive, args.effective_caps))


if __name__ == "__main__":
    main()
