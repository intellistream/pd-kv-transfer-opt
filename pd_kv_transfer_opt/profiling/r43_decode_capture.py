"""Capture separate four-rank Decode windows beside a native r43 diagnostic replay."""

import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path
import re
import time
import urllib.request

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.experiments.run_suite import profile_rpc
from pd_kv_transfer_opt.experiments.serve import endpoints, model_commands, read_deployment, store_pool_layout
from pd_kv_transfer_opt.profiling.capture_decode_windows import metric


RANK_IN_NAME = re.compile(r"_rank(\d+)_")
TARGETS = {"1-4": range(1, 5), "9-16": range(9, 17)}
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
CANN_FILES = ("kernel_details.csv", "communication.json", "trace_view.json")


def target_band(running):
    return next((name for name, widths in TARGETS.items() if running in widths), None)


def plan(config_path, workload_path, shaping_path, run_dir, roles, stride):
    config = read_deployment(config_path)
    template = read_deployment(CONFIG_DIR / "ascend-2p2d-decode-apc.example.json")
    for role in ("P", "D"):
        for key in ("enable_prefix_caching", "max_num_seqs", "max_num_batched_tokens",
                    "gpu_memory_utilization", "compilation_config", "recompute_scheduler_enable"):
            if config[role].get(key, False) != template[role].get(key, False):
                raise ValueError(f"r43 requires {role}.{key} from the baseline template")
    workload = read_json(workload_path)
    if workload != read_json(CONFIG_DIR / "workload-r42-natural-sustained.json"):
        raise ValueError("The diagnostic requires the published r43 workload and natural EOS policy")
    shaping = read_json(shaping_path)
    readback = shaping["readback"]
    if (shaping["aggregate_Gbps"] != 200 or set(readback) != {str(i) for i in range(8)}
            or any(port["configured_Mbps"] != 25_000 or port["current_Mbps"] != 25_000
                   for port in readback.values())):
        raise ValueError("Expected an eight-port 200 Gbps readback from the native replay")
    store, commands = model_commands(config, "D", run_dir, profile_decode_steps=True, decode_native_profiler=True)
    return {"workload_sha256": workload["workload_sha256"], "D_readback_sha256": sha256(shaping_path),
            "step_stride": stride, "roles": roles, "target_width_bands": list(TARGETS),
            "expected_D_instances": commands, "expected_D_store": store, "store_pool_layout": store_pool_layout(config),
            "endpoints": {role: endpoints(config)[role] for role in roles}}


def validate_state(run_dir, capture_plan):
    state = read_json(run_dir / "state.json")
    expected = capture_plan["expected_D_instances"]
    running = [{key: row[key] for key in ("id", "argv", "devices", "port")} for row in state["instances"]]
    if (state["role"] != "D" or state["stage"] != "ready" or not state.get("decode_native_profiler")
            or not state.get("decode_step_observations") or state.get("observations_enabled")
            or state.get("decode_step_stride") != capture_plan["step_stride"]
            or state.get("store_pool_layout") != capture_plan["store_pool_layout"]
            or running != expected or state["store_config"] != capture_plan["expected_D_store"]):
        raise ValueError("The ready D service does not match the native r43 profiling plan")
    return state["run_id"]


def new_ranks(root, before, *, wait_s=8):
    deadline = time.monotonic() + wait_s
    while True:
        ranks = {}
        for path in sorted(root.glob("*_ascend_pt")):
            if path in before or (match := RANK_IN_NAME.search(path.name)) is None:
                continue
            rank = int(match[1])
            if rank in ranks or rank not in range(4):
                raise ValueError(f"Ambiguous rank captures under {root}")
            ranks[rank] = path
        if len(ranks) == 4 or time.monotonic() >= deadline:
            return ranks
        time.sleep(.2)


def capture_one(role, endpoint, run_dir, receipt, seconds):
    root = run_dir / "native-process-profile" / role
    before = set(root.glob("*_ascend_pt"))
    window = {"role": role, "request_ns": time.monotonic_ns()}
    try:
        window["start"] = profile_rpc(role, endpoint, "start")
        window["started_ns"] = time.monotonic_ns()
        if window["start"]["status"] != 200:
            raise RuntimeError(f"Profiler start failed on {role}")
        time.sleep(seconds)
    finally:
        try:
            window["stop_request_ns"] = time.monotonic_ns()
            window["stop"] = profile_rpc(role, endpoint, "stop")
        finally:
            window["stopped_ns"] = time.monotonic_ns()
            receipt.update(window)
    ranks = new_ranks(root, before)
    receipt["ranks"] = {str(rank): path.relative_to(run_dir).as_posix() for rank, path in ranks.items()}
    complete_export = len(ranks) == 4 and all(
        (path / "ASCEND_PROFILER_OUTPUT" / name).is_file() for path in ranks.values() for name in CANN_FILES
    )
    if len(ranks) != 4 or (window["stop"]["status"] != 200 and not complete_export):
        raise RuntimeError(f"Four worker profiles or profiler shutdown could not be verified on {role}")


def collect(run_dir, capture_plan, output, *, seconds, narrow_after_s, deadline_s):
    run_id = validate_state(run_dir, capture_plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Use a fresh capture receipt path")
    roles = capture_plan["roles"]
    pending = {(role, name) for role in roles for name in TARGETS}
    record = {"run_id": run_id, "workload_sha256": capture_plan["workload_sha256"],
              "D_readback_sha256": capture_plan["D_readback_sha256"], "step_stride": capture_plan["step_stride"],
              "stage": "collecting", "windows": []}
    write_json(output, record)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    previous = {}
    metric_failures = Counter()
    begin = time.monotonic()
    try:
        while pending and time.monotonic() - begin < deadline_s:
            for role in roles:
                if not any(target[0] == role for target in pending):
                    continue
                endpoint = capture_plan["endpoints"][role]
                try:
                    with opener.open(endpoint + "/metrics", timeout=3) as response:
                        reported = response.read().decode()
                    running = metric(reported, "num_requests_running")
                    tokens = metric(reported, "generation_tokens_total")
                except (OSError, ValueError):
                    previous.pop(role, None)
                    metric_failures[role] += 1
                    continue
                band = target_band(running)
                last_band, last_tokens = previous.get(role, (None, 0))
                previous[role] = band, tokens
                if (band != last_band or (role, band) not in pending or tokens <= last_tokens
                        or (band == "1-4" and time.monotonic() - begin < narrow_after_s)):
                    continue
                window = {"role": role, "target_width": band, "running_at_trigger": running}
                record["windows"].append(window)
                capture_one(role, endpoint, run_dir, window, seconds)
                pending.remove((role, band))
                write_json(output, record)
            if pending:
                time.sleep(.25)
        record["stage"] = "partial" if pending else "captured"
    except BaseException as error:
        record["stage"] = "interrupted_or_failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["missing_windows"] = [list(item) for item in sorted(pending)]
        record["metric_read_failures"] = dict(metric_failures)
        write_json(output, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--shaping", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--roles", nargs="+", choices=("D0", "D1"), default=("D0", "D1"))
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--narrow-after-s", type=float, default=93.)
    parser.add_argument("--deadline-s", type=float, default=540.)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    durations = (args.window_s, args.narrow_after_s, args.deadline_s)
    if (not all(isfinite(value) for value in durations) or args.window_s <= 0 or args.narrow_after_s < 0
            or args.deadline_s <= 0 or args.stride <= 0 or len(set(args.roles)) != len(args.roles)):
        parser.error("Invalid capture timing, stride or roles")
    run_dir = args.run_dir.resolve()
    capture_plan = plan(args.config, args.workload_manifest, args.shaping, run_dir, args.roles, args.stride)
    if args.dry_run:
        print(json.dumps({**capture_plan, "target_seconds": args.window_s, "deadline_s": args.deadline_s},
                         indent=2))
    else:
        if args.output is None:
            parser.error("--output is required for capture")
        collect(run_dir, capture_plan, args.output, seconds=args.window_s, narrow_after_s=args.narrow_after_s,
                deadline_s=args.deadline_s)


if __name__ == "__main__":
    main()
