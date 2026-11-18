"""Read-only Linux inventory and timestamped host/RNIC/Prometheus collection."""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request

from pd_kv_transfer_opt.common import clock_anchor, new_directory, write_json


def command(argv: list[str], timeout: float = 8) -> dict:
    begin = time.monotonic_ns()
    try:
        result = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
        # Commands here are bounded inventory/status reads, not service log dumps.
        data = result.stdout[:524288].decode(errors="replace")
        value = {"argv": argv, "exit_code": result.returncode, "stdout": data,
                 "stderr": result.stderr[:65536].decode(errors="replace"),
                 "truncated": len(result.stdout) > 524288 or len(result.stderr) > 65536}
    except (OSError, subprocess.TimeoutExpired) as exc:
        value = {"argv": argv, "error": repr(exc)}
    return {**value, "start_monotonic_ns": begin, "end_monotonic_ns": time.monotonic_ns()}


def read_text(path: Path) -> dict:
    try:
        return {"value": path.read_text().strip()}
    except OSError as exc:
        return {"error": repr(exc)}


def npu_snapshot() -> dict:
    # Some hosts wrap the text command and omit idle devices. Keep structured
    # output when available, and retain the failed attempt on vendor-only hosts.
    structured = command(["npu-smi", "info", "--json"], timeout=5)
    if structured.get("exit_code") == 0:
        try:
            return {"format": "json", "data": json.loads(structured["stdout"]), "command": structured}
        except (KeyError, ValueError):
            pass
    return {"format": "text", "command": command(["npu-smi", "info"], timeout=5),
            "structured_attempt": structured}


def hccn_stats(devices: list[int], tool: str, use_sudo: bool) -> dict:
    """Retain NPU fabric counters separately from host RNIC counters."""
    result = {}
    for device in devices:
        argv = (["sudo", "-n"] if use_sudo else []) + [tool, "-i", str(device), "-stat", "-g"]
        record = command(argv, timeout=3)
        parsed = {}
        if record.get("exit_code") == 0 and not record.get("truncated"):
            for line in record["stdout"].splitlines():
                match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*):\s*(\d+)", line.strip())
                if match:
                    parsed[match.group(1)] = int(match.group(2))
        result[str(device)] = {"command": record, "raw_counters": parsed,
                               "available": bool(parsed)}
    return result


def counters() -> dict:
    result = {}
    patterns = ["/sys/class/infiniband/*/ports/*/counters/*",
                "/sys/class/infiniband/*/ports/*/hw_counters/*",
                "/sys/class/net/*/statistics/*"]
    for pattern in patterns:
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            if path.is_file():
                result[str(path)] = read_text(path)
    return result


def inventory(output: Path, nics: list[str], hccn_devices: list[int] | None = None,
              hccn_tool: str = "/usr/local/Ascend/driver/tools/hccn_tool", hccn_sudo: bool = False) -> None:
    commands = [["uname", "-a"], ["lscpu"], ["numactl", "--hardware"],
                ["lspci", "-D", "-nn"], ["npu-smi", "info"],
                ["rdma", "link", "show"], ["ibv_devices"], ["ibv_devinfo"],
                ["ip", "-j", "address"], ["ip", "-j", "link"],
                ["chronyc", "tracking"], ["chronyc", "sources", "-v"]]
    for nic in nics:
        commands.extend([["ethtool", "-i", nic], ["ethtool", "-S", nic],
                         ["ethtool", "-a", nic], ["ethtool", nic]])
    packages = {}
    for package in ("torch", "torch-npu", "vllm", "vllm-ascend", "mooncake-transfer-engine", "mooncake-transfer-engine-npu", "pegaflow-llm"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    topology = {}
    for prefix in ("net", "infiniband"):
        for path in sorted(Path(f"/sys/class/{prefix}").glob("*/device")):
            topology[str(path)] = {"pci_path": str(path.resolve()),
                                   "numa_node": read_text(path / "numa_node")}
    write_json(output / "inventory.json", {"schema": "rdma-lab-inventory/v1",
               "host": socket.gethostname(), "platform": platform.platform(),
               "anchor": clock_anchor(), "packages": packages, "topology": topology,
               "npu_snapshot": npu_snapshot(),
               "hccn_stats": hccn_stats(hccn_devices or [], hccn_tool, hccn_sudo),
               "commands": [command(argv) for argv in commands], "counters": counters(),
               "driver_version": read_text(Path("/usr/local/Ascend/driver/version.info"))})


def fetch_metrics(url: str) -> dict:
    start = time.monotonic_ns()
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = response.read(2097153)
            value = {"status": response.status, "text": payload[:2097152].decode(errors="replace"),
                     "truncated": len(payload) > 2097152}
    except (OSError, urllib.error.URLError) as exc:
        value = {"error": repr(exc)}
    return {**value, "start_monotonic_ns": start, "end_monotonic_ns": time.monotonic_ns()}


def sample(output: Path, seconds: float, interval: float, metrics: dict[str, str],
           metrics_every: float, npu_every: float, pids: list[int],
           hccn_devices: list[int] | None = None, hccn_every: float = 5,
           hccn_tool: str = "/usr/local/Ascend/driver/tools/hccn_tool", hccn_sudo: bool = False) -> None:
    origin = time.monotonic()
    due_metrics = due_npu = due_hccn = origin
    sample_no = 0
    with gzip.open(output / "host-samples.jsonl.gz", "xt", encoding="utf-8") as stream:
        while time.monotonic() - origin < seconds:
            start = time.monotonic()
            row = {"type": "host_sample", "sample_no": sample_no, "anchor": clock_anchor(),
                   "elapsed_s": start - origin, "counters": counters(),
                   "proc": {name: read_text(Path("/proc") / name)
                            for name in ("stat", "meminfo", "vmstat", "loadavg", "pressure/cpu", "pressure/memory", "pressure/io")},
                   "pids": {str(pid): {name: read_text(Path(f"/proc/{pid}/{name}"))
                                        for name in ("stat", "status", "schedstat", "io")}
                            for pid in pids}}
            if start >= due_metrics:
                row["metrics"] = {name: fetch_metrics(url) for name, url in metrics.items()}
                due_metrics = start + metrics_every
            if npu_every > 0 and start >= due_npu:
                row["npu"] = npu_snapshot()
                due_npu = start + npu_every
            if hccn_devices and start >= due_hccn:
                row["hccn_stats"] = hccn_stats(hccn_devices, hccn_tool, hccn_sudo)
                # Anchor the cadence to the run origin; per-tick jitter must not
                # postpone a sample by a full extra interval.
                due_hccn += max(1, int((start - due_hccn) // hccn_every) + 1) * hccn_every
            row["collection_elapsed_s"] = time.monotonic() - start
            row["overran_interval"] = row["collection_elapsed_s"] > interval
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            sample_no += 1
            time.sleep(max(0, origin + sample_no * interval - time.monotonic()))
    write_json(output / "collection-end.json", {"anchor": clock_anchor(), "samples": sample_no,
               "elapsed_s": time.monotonic() - origin})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=0, help="0: inventory only")
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--nic", action="append", default=[])
    parser.add_argument("--metrics", action="append", default=[], metavar="NAME=URL")
    parser.add_argument("--metrics-every", type=float, default=5)
    parser.add_argument("--npu-every", type=float, default=10, help="0 disables npu-smi sampling")
    parser.add_argument("--pid", type=int, action="append", default=[])
    parser.add_argument("--hccn-device", type=int, action="append", default=[],
                        help="NPU physical device ID; repeat for each participating card")
    parser.add_argument("--hccn-every", type=float, default=5)
    parser.add_argument("--hccn-tool", default="/usr/local/Ascend/driver/tools/hccn_tool")
    parser.add_argument("--hccn-sudo", action="store_true", help="Use noninteractive sudo for read-only hccn stats")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run this collector on the Linux experiment host")
    import math
    if (not all(math.isfinite(value) for value in (args.seconds, args.interval, args.metrics_every, args.npu_every, args.hccn_every))
            or args.seconds < 0 or min(args.interval, args.metrics_every) <= 0 or args.npu_every < 0
            or args.hccn_every <= 0 or any(device < 0 or device > 63 for device in args.hccn_device)
            or len(set(args.hccn_device)) != len(args.hccn_device)
            or any(not re.fullmatch(r"[A-Za-z0-9_.:-]+", nic) for nic in args.nic)):
        parser.error("Invalid sampling parameters")
    metrics = dict(item.split("=", 1) for item in args.metrics)
    output = new_directory(args.output)
    inventory(output, args.nic, args.hccn_device, args.hccn_tool, args.hccn_sudo)
    if args.seconds:
        sample(output, args.seconds, args.interval, metrics, args.metrics_every, args.npu_every, args.pid,
               args.hccn_device, args.hccn_every, args.hccn_tool, args.hccn_sudo)


if __name__ == "__main__":
    main()
