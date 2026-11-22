"""Screen directional KV demand and fixed-trace P egress sensitivity."""

import argparse
from collections import defaultdict, deque
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS, phase_times, read_trace, replay, requests_for


P_RATE = 44_000
D_RATE = 1_154
GET = "batch_get_into_multi_buffers"


def peak_demand(events, width):
    window = deque()
    pending = best = 0
    last = 0.0
    for at, _, volume, _ in events:
        while window and at - window[0][0] >= width:
            pending -= window.popleft()[1]
        pending += volume
        window.append((at, volume))
        if pending > best:
            best, last = pending, at
    return {"Gbps": best * 8 / NS / width, "end_s": last}


def capture(arm, profiles, origin, wall):
    handoff, writes, gets = [], [], []
    native_finish = {}
    ranks = defaultdict(set)
    for row in arm["d_native"]:
        kind, key = row["type"], row.get("remote_request_id")
        if kind not in ("native_transfer_submit", "native_handle_end"):
            continue
        if row["instance"] != profiles[key]["D"]:
            raise ValueError(f"Unexpected D route for {key}")
        at = row["monotonic_ns"] / NS - origin
        if kind == "native_transfer_submit":
            handoff.append((at, profiles[key]["P"], row["payload_bytes"], key))
            ranks[key].add(row["pid"])
        else:
            native_finish[key] = max(at, native_finish.get(key, 0))
    if ranks.keys() != profiles.keys() or native_finish.keys() != profiles.keys() or any(len(pids) != 4 for pids in ranks.values()):
        raise ValueError("Incomplete four-rank handoff coverage")
    for row in arm["p_native"]:
        if row["type"] == "adxl_transfer_sync" and row["operation"] == 1 and row["status"] == 0:
            at = (row["unix_ns"] + row["begin_ns"] - row["monotonic_ns"] - wall) / NS
            writes.append((at, row["instance"], row["payload_bytes"], ""))
        if row["type"] == "store_begin" and row["operation"] == GET:
            at = (row["unix_ns"] - wall) / NS
            gets.append((at, row["instance"], row["payload_bytes"], ""))
    return sorted(handoff), sorted(writes), sorted(gets), native_finish


def p_queue(events, native_finish, profiles, cap, ports_per_prefiller):
    available = defaultdict(float)
    finished = defaultdict(float)
    service_bytes_s = cap / 8 * ports_per_prefiller * NS / 8
    for at, source, size, request in events:
        available[source] = max(at, available[source]) + size / service_bytes_s
        if request and size:
            finished[request] = max(finished[request], available[source])
    extra = {key: max(0.0, finished[key] - end) for key, end in native_finish.items()}
    phases = phase_times(profiles)
    cost = {key: {**value, "D_to_first": value["D_to_first"] + extra[key]} for key, value in phases.items()}
    _, done, chain = replay(profiles, cost)
    duration = max(done.values()) - min(row["offered_s"] for row in profiles.values())
    outputs = sum(row["output_tokens"] for row in profiles.values())
    return {"overruns_ms": distribution(v * 1000 for v in extra.values() if v > 0),
            "duration_s": duration, "output_token_s": outputs / duration, "tail": chain[-1]}


def session_windows(profiles, method, handoff, writes):
    by_session, by_request = defaultdict(list), defaultdict(int)
    for key, row in profiles.items():
        by_session[row["session_id"]].append(key)
    for _, _, size, key in handoff:
        by_request[key] += size
    # Approximate a fresh first handoff by the upper observed bytes/token ratio of actual roots.
    max_cold_handoff = max(by_request[k] / method[k]["prompt_tokens"] for k, v in profiles.items() if v["predecessor"] is None)
    put_per_compute = sum(row[2] for row in writes) / sum(row["P_computed_prompt_tokens"] for row in method.values())
    windows = []
    for session, keys in by_session.items():
        for i, first in enumerate(keys):
            compute = method[first]["prompt_tokens"]
            output = profiles[first]["output_tokens"]
            handoff_bytes = round(compute * max_cold_handoff)
            get_bytes = 0
            for last in keys[i + 1:]:
                compute += method[last]["P_computed_prompt_tokens"]
                output += profiles[last]["output_tokens"]
                handoff_bytes += by_request[last]
                get_bytes += method[last]["P_direct_GET_bytes"]
                service = max(compute / P_RATE, output / D_RATE)
                windows.append({"session": session, "first": first, "last": last, "P_compute": compute, "D_output": output,
                                "P_to_D_handoff_Gbps": handoff_bytes * 8 / NS / service,
                                "P_to_D_with_background_proxy_Gbps": (handoff_bytes + compute * put_per_compute) * 8 / NS / service,
                                "D_to_P_GET_Gbps": get_bytes * 8 / NS / service})
    fields = ("P_to_D_handoff_Gbps", "P_to_D_with_background_proxy_Gbps", "D_to_P_GET_Gbps")
    return {"windows": len(windows), "cold_handoff_bytes_per_input_token": max_cold_handoff,
            "P_background_bytes_per_computed_token": put_per_compute,
            "strongest_by_path": {field: max(windows, key=lambda row: row[field]) for field in fields}}


def summarize(arm, case, specs):
    event, = [e for e in arm["client"] if e["type"] == "client_origin"]
    origin = event["monotonic_s"]
    profiles = requests_for(arm, specs, origin)
    handoff, writes, gets, native_finish = capture(arm, profiles, origin, event["anchor"]["unix_ns"])
    physical = case["physical_paths"]
    if sum(row[2] for row in handoff) != physical["P_to_D_Mooncake_descriptor_bytes"] or sum(row[2] for row in gets) != physical["D_to_P_Store_GET_descriptor_bytes"]:
        raise ValueError("The direction byte totals disagree with the independent method summary")
    combined = sorted(handoff + writes)
    series = {"handoff": handoff, "P_Store_PUT": writes, "P_combined": combined,
              "P0_combined": [x for x in combined if x[1] == "P0"], "P1_combined": [x for x in combined if x[1] == "P1"],
              "D_Store_GET": gets}
    peaks = {key: {str(width): peak_demand(value, width) for width in (1, 5, 10, 30)}
             for key, value in series.items()}
    duration = max(row["end_s"] for row in profiles.values()) - min(row["offered_s"] for row in profiles.values())
    outputs = sum(row["output_tokens"] for row in profiles.values())
    return {"observed": {"duration_s": duration, "output_tokens": outputs, "output_token_s": outputs / duration},
            "directional_submitted_bytes": {key: sum(e[2] for e in value) for key, value in series.items()},
            "peak_offers": peaks,
            "P_FIFO_replay": {str(ports): {str(cap): p_queue(combined, native_finish, profiles, cap, ports)
                                           for cap in (100, 200)} for ports in (4, 2)},
            "session_windows": session_windows(profiles, case["per_request"], handoff, writes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method_archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.method_archive / "manifest.json")
    path = args.method_archive / "r43-method-events.jsonl.gz"
    digest = sha256(path)
    if digest != manifest["trace_sha256"]:
        raise ValueError("The method trace does not match its manifest")
    summary = read_json(args.method_archive / "summary.json")["cases"]
    specs = manifest["workload"]["request_sources"]
    arms = {name: summarize(data, summary[name], specs) for name, data in read_trace(path).items()}
    write_json(args.output, {"source_trace_sha256": digest, "arms": arms,
        "scope": "Fixed observed input, route, upstream submit times and EOS; pooled FIFO by sending P instance; "
                 "successful P ADXL PUT is concurrent background; all modeled handoff overruns added to D-to-first; "
                 "outputs and remaining stages frozen. Numbers are sensitivity estimates, not new link measurements."})


if __name__ == "__main__":
    main()
