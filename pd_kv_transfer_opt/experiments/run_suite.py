"""Run the r14 bandwidth matrix against a ready D-side 2P2D controller."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from pd_kv_transfer_opt.common import read_json, write_json
from pd_kv_transfer_opt.experiments.cohort import fixed_sources
from pd_kv_transfer_opt.experiments.hardware import Shaper
from pd_kv_transfer_opt.experiments.rdma_full_protocol import CASES, offered_rate, port_comparison
from pd_kv_transfer_opt.experiments.serve import endpoints, read_deployment


def check_health(urls):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for url in urls:
        with opener.open(url, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"Endpoint is not ready: {url}")


def profile_rpc(instance, endpoint, action):
    started = time.time_ns()
    request = urllib.request.Request(f"{endpoint}/{action}_profile", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=185 if action == "stop" else 45) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    return {"instance": instance, "status": status, "start_unix_ns": started, "end_unix_ns": time.time_ns()}


def worker_captures_closed(run_dir, number):
    path = run_dir / "logs/D0.log"
    raw = run_dir / "native-process-profile/D0"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        text = path.read_text(errors="replace")
        markers = re.findall(
            r"Worker_TP(\d+)[^\n]*(?:The profiling data cannot be parsed during the daemon process|All profiling data parsed)", text
        )
        counts = Counter(markers)
        if len(list(raw.glob("*_ascend_pt"))) >= 4 * number and all(counts[str(i)] >= number for i in range(4)):
            return {"captures_by_rank": dict(counts)}
        time.sleep(.35)
    raise RuntimeError("D0 model-worker capture close could not be verified.")


def run_case(args, config, state, shaper, folder, case):
    output = args.run_dir / f"natural-sessions-{case}"
    command = [sys.executable, "-m", "pd_kv_transfer_opt.experiments.cohort",
               "--case", case, "--run-id", state["run_id"], "--workload", str(args.workload),
               "--endpoint", f"http://127.0.0.1:{config['proxy_port']}", "--model", config["model_name"], "--output", str(output)]
    with (folder / f"{case}-ports.jsonl").open("x", buffering=1, encoding="utf-8") as counters, \
         (folder / f"{case}.log").open("x", encoding="utf-8") as log:
        def sample():
            counters.write(json.dumps(shaper.sample(), separators=(",", ":")) + "\n")

        sample()
        with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as client:
            deadline = time.monotonic() + 500
            try:
                while client.poll() is None:
                    try:
                        client.wait(timeout=2.5)
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Cohort did not drain: {case}")
                        sample()
            finally:
                if client.poll() is None:
                    client.terminate()
                    try:
                        client.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        client.kill()
                        client.wait()
            if client.returncode:
                raise RuntimeError(f"Cohort {case} exited with {client.returncode}. See {folder / (case + '.log')}.")
        sample()


def profile_case(args, config, state, shaper, folder, case, number):
    urls = endpoints(config)
    labels = ("P0", "D0")
    receipt = {"case": case, "stage": "starting"}
    with ThreadPoolExecutor(max_workers=2) as workers:
        def rpc(action):
            return list(workers.map(lambda label: profile_rpc(label, urls[label], action), labels))

        try:
            receipt["start"] = rpc("start")
            if any(row["status"] != 200 for row in receipt["start"]):
                raise RuntimeError(f"Profile start failed: {receipt}")
            receipt["stage"] = "collecting"
            write_json(folder / f"{case}-trace.json", receipt)
            run_case(args, config, state, shaper, folder, case)
        finally:
            receipt["stop"] = rpc("stop")
            write_json(folder / f"{case}-trace.json", receipt)
    if any(row["status"] not in (200, 500) for row in receipt["stop"]):
        raise RuntimeError(f"Profile stop failed: {receipt}")
    if any(row["status"] == 500 for row in receipt["stop"]):
        receipt["D_worker_stop"] = worker_captures_closed(args.run_dir, number)
    receipt["stage"] = "captured"
    write_json(folder / f"{case}-trace.json", receipt)


def run(args, cases, config, port_rates=None):
    state = read_json(args.run_dir / "state.json")
    if state["role"] != "D" or state["stage"] != "ready":
        raise RuntimeError("Run on the ready D host with a D-side run directory.")
    if any(config[role].get("enable_prefix_caching") is not False for role in ("P", "D")):
        raise ValueError("Set enable_prefix_caching to false on both roles for the r14 matrix.")
    if args.profiles and state.get("observations_enabled") is False:
        raise ValueError("Start both roles with --observe before requesting r14 profiler windows.")
    fixed_sources(args.workload)
    check_health([url + "/health" for url in endpoints(config).values()] +
                 [f"http://127.0.0.1:{config['proxy_port']}/docs"])
    folder = args.run_dir / "hard-cap-suite"
    folder.mkdir(exist_ok=False)
    shaper = Shaper(args.hccn_tool, args.sudo)
    before = shaper.read()
    if any(item != {"configured_Mbps": 200000, "maximum_Mbps": 200000, "current_Mbps": 200000} for item in before.values()):
        raise ValueError("The r14 baseline requires eight ports at the original 200 Gbps setting.")
    status = {"stage": "running", "completed_cases": [], "restored": False}
    if port_rates:
        weighted_case, = port_rates
        status["port_rate_comparison"] = {"weighted_case": weighted_case, "port_Mbps": port_rates[weighted_case]}
    write_json(folder / "shaping-initial.json", before)
    mutation_started = False
    profiles = 0

    def interrupt(signum, _frame):
        raise InterruptedError(f"Signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)
    try:
        for case, cap, count in cases:
            status["current"] = case
            write_json(folder / "progress.json", status)
            if cap is not None:
                mutation_started = True
                rates = port_rates.get(case) if port_rates else None
                readback = shaper.apply_ports(rates) if rates else shaper.apply(cap)
                receipt = {"requested_aggregate_Gbps": cap, "readback": readback}
                if port_rates:
                    receipt["requested_port_Mbps"] = rates or [cap * 125] * 8
                write_json(folder / f"{case}-shaping.json", receipt)
            if case.startswith("profile-"):
                profiles += 1
                profile_case(args, config, state, shaper, folder, case, profiles)
            else:
                run_case(args, config, state, shaper, folder, case)
            status["completed_cases"].append({"name": case, "aggregate_D_egress_cap_Gbps": cap, "requests": count})
            write_json(folder / "progress.json", status)
            time.sleep(2)
        status.update(stage="complete", current=None)
    except BaseException as error:
        status.update(stage="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if mutation_started:
            # A second interrupt must leave time for all ports to be restored.
            for signum in (signal.SIGINT, signal.SIGTERM):
                signal.signal(signum, signal.SIG_IGN)
            try:
                receipt = shaper.restore(before)
                write_json(folder / "shaping-restore.json", receipt)
                status["restored"] = receipt["restored"]
                if not receipt["restored"]:
                    raise RuntimeError(f"Restore readback failed. See {folder / 'shaping-restore.json'}.")
            finally:
                write_json(folder / "progress.json", status)
        else:
            status["restored"] = True
            write_json(folder / "progress.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--hccn-tool", type=Path, default=Path("/usr/local/Ascend/driver/tools/hccn_tool"))
    parser.add_argument("--sudo", action="store_true", help="Use sudo -n for hccn_tool.")
    parser.add_argument("--profiles", action="store_true", help="Append separate P0/D0 profiler windows.")
    parser.add_argument("--port-rates-mbps", type=int, nargs=8,
                        help="Use the supplied port rates in the middle arm of an equal/weighted/equal comparison.")
    parser.add_argument("--apply", action="store_true", help="Run real traffic and change local D egress caps.")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    config = read_deployment(args.config)
    port_rates = None
    if args.port_rates_mbps:
        if args.profiles:
            parser.error("Run method profiling separately from the fixed-total port comparison.")
        try:
            cases, port_rates = port_comparison(args.port_rates_mbps)
        except ValueError as error:
            parser.error(str(error))
    else:
        cases = [row for row in CASES if args.profiles or not row[0].startswith("profile-")]
    if not args.apply:
        plan = []
        for case, cap, count in cases:
            row = {"case": case, "cap_Gbps": cap, "requests": count, "offered_s_inv": offered_rate(case)}
            if port_rates and cap is not None:
                row["port_Mbps"] = port_rates.get(case, [cap * 125] * 8)
            plan.append(row)
        print(json.dumps(plan, indent=2))
        return
    run(args, cases, config, port_rates)


if __name__ == "__main__":
    main()
