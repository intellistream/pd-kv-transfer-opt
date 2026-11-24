"""Run the prepared native 2P2D matrix from a controller with SSH access to both hosts."""

import argparse
from collections import Counter
from contextlib import ExitStack, suppress
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path, PurePosixPath
import select
import shlex
import signal
import subprocess
import sys
import tarfile
import time

from pd_kv_transfer_opt.common import new_directory, read_json, sha256, write_json
from pd_kv_transfer_opt.experiments.basin_burst_validation import verified_shaping
from pd_kv_transfer_opt.experiments.hardware import PORTS
from pd_kv_transfer_opt.experiments.serve import endpoints, read_deployment, store_pool_layout
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import parse_metrics
from pd_kv_transfer_opt.workloads.native_replay_inputs import SPECS


MODULE = "pd_kv_transfer_opt.experiments.native_npu_suite"
INPUT_DIR = Path("data/native-npu-inputs")
MATRIX = Path("configs/basin-windows-v1/matrix.json")
BASELINE = Path("configs/ascend-2p2d-decode-apc.example.json")
SOFTWARE_REFERENCE = Path("results/2026-11-19-r28-natural-bandwidth/manifest.json")
START_TIMEOUT_S = 960
ARM_TIMEOUT_S = 3600


@dataclass(frozen=True)
class Arm:
    name: str
    cap: int
    workload: str
    workload_sha256: str
    plan: str | None = None
    arm: str | None = None
    profile: bool = False
    requests: int = 0


def arms(group, include_termination):
    if group == "cancel-check":
        workload = "data/basin-smoke-inputs.jsonl.gz"
        digest = "173d5c65f78b52c744e32057a3156c07ed9c6a76e60386924fd7302745a0ae83"
        if sha256(workload) != digest:
            raise ValueError("Unexpected input hash for the Basin cancellation diagnostic")
        return [Arm("basin-cancel-check-200", 200, workload, digest,
                    "configs/basin-smoke-short.json", "burst-terminate", requests=27)]
    result = []
    if group in ("formal", "r43", "r69", "r43-profile"):
        receipt = read_json(INPUT_DIR / "manifest.json")
        workloads = ("r43", "r69") if group == "formal" else (group.removesuffix("-profile"),)
        for name in workloads:
            info = receipt["workloads"][name]
            if sha256(SPECS[name]) != info["spec_sha256"]:
                raise ValueError(f"Prepared inputs do not match the {name} plan")
            path = (INPUT_DIR / info["file"]).as_posix()
            if group == "r43-profile":
                result.append(Arm("r43-profile-200", 200, path, info["sha256"], profile=True,
                                  requests=info["requests"]))
            else:
                for order, caps in enumerate(((100, 200), (200, 100)), start=1):
                    result.extend(Arm(f"{name}-b{order}-cap{cap}", cap, path, info["sha256"],
                                      requests=info["requests"]) for cap in caps)
    if group in ("formal", "basin", "basin-termination"):
        matrix = read_json(MATRIX)
        for condition in matrix["conditions"]:
            optional = bool(condition.get("qualification"))
            if (optional and not (include_termination or group == "basin-termination")
                    or not optional and group == "basin-termination"):
                continue
            plan = matrix["plans"][condition["plan"]]
            if sha256(plan["path"]) != plan["sha256"]:
                raise ValueError(f"Frozen Basin plan has changed: {plan['path']}")
            result.extend(Arm(f"basin-{condition['plan']}-{condition['arm']}-{cap}", cap,
                              matrix["workload"]["path"], matrix["workload"]["sha256"],
                              plan["path"], condition["arm"], requests=condition["formal_calls"])
                          for cap in condition["cap_order_Gbps"])
    verified = {}
    for arm in result:
        if arm.workload not in verified:
            verified[arm.workload] = sha256(arm.workload)
        if verified[arm.workload] != arm.workload_sha256:
            raise ValueError(f"Unexpected prepared input hash for {arm.name}")
    return result


def shared_stage_files(deployment):
    return {path.as_posix() for path in {*Path("pd_kv_transfer_opt").rglob("*.py"), deployment, BASELINE}}


def prepare_bundles(destination, deployment, cases):
    common = shared_stage_files(deployment)
    decode = set()
    for arm in cases:
        decode.add(arm.workload)
        if arm.plan is not None:
            decode.add(arm.plan)
        else:
            decode.add(SPECS[arm.name.split("-", 1)[0]])
    destination = new_directory(destination)
    receipt = {}
    for role, paths in (("P", common), ("D", common | decode)):
        path = destination / f"{role}.tar.gz"
        with tarfile.open(path, "w:gz", compresslevel=4) as bundle:
            for entry in sorted(paths):
                bundle.add(entry, arcname=entry, recursive=False)
        receipt[role] = {"file": path.name, "files": len(paths), "bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(destination / "manifest.json", receipt)
    return receipt


class Host:
    def __init__(self, settings, direct_hosts):
        self.ssh = settings["ssh"]
        self.host_repo = PurePosixPath(settings["host_repo"])
        self.runtime_repo = PurePosixPath(settings.get("runtime_repo", self.host_repo))
        self.runtime_prefix = settings.get("runtime_prefix", [])
        self.hccn_tool = settings.get("hccn_tool", "/usr/local/Ascend/driver/tools/hccn_tool")
        self.sudo = settings.get("hccn_sudo", True)
        self.host_python = settings.get("host_python", "python3")
        self.runtime_python = settings.get("runtime_python", "python3")
        self.no_proxy = ",".join(("localhost", "127.0.0.1", *direct_hosts))
        if self.ssh.startswith("-") or ".." in self.host_repo.parts or not self.runtime_repo.is_absolute():
            raise ValueError("Use a valid SSH destination and an absolute or home-relative host path")

    def ssh_command(self, command):
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ConnectionAttempts=2",
                "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2", self.ssh, command]

    def command(self, argv, runtime=False):
        repo = self.runtime_repo if runtime else self.host_repo
        env_argv = ["env", f"NO_PROXY={self.no_proxy}", f"no_proxy={self.no_proxy}", *map(str, argv)]
        command = f"cd {shlex.quote(str(repo))} && exec {shlex.join(env_argv)}"
        if runtime and self.runtime_prefix:
            command = shlex.join([*self.runtime_prefix, "sh", "-c", command])
        return self.ssh_command(command)

    def run(self, argv, *, runtime=False, payload=None, check=True, timeout=20):
        return subprocess.run(self.command(argv, runtime), input=payload, stdout=subprocess.PIPE,
                              check=check, timeout=timeout)

    def python(self, module, *args, runtime=False):
        interpreter = self.runtime_python if runtime else self.host_python
        return [interpreter, "-B", "-u", "-m", module, *map(str, args)]

    def put_json(self, path, value):
        self.run([self.host_python, "-c",
                  "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())", path],
                 payload=json.dumps(value).encode())

    def stage(self, bundle):
        target = shlex.quote(str(self.host_repo))
        command = self.ssh_command(f"mkdir -p -- {target} && exec tar -xzf - -C {target}")
        with bundle.open("rb") as stream:
            subprocess.run(command, stdin=stream, check=True, timeout=300)


def idle(host):
    result = subprocess.run(host.ssh_command("npu-smi info"), stdout=subprocess.PIPE, check=True, timeout=25)
    status = result.stdout.decode(errors="replace")
    if "No running processes." not in status and not all(
            f"No running processes found in NPU {port}" in status for port in PORTS):
        raise RuntimeError(f"All eight cards must be idle on {host.ssh}")
    return status


def verify_stage(host, role, required, deployment, config, site, reference):
    assets = shared_stage_files(deployment)
    for runtime, paths in ((False, sorted(assets)), (True, sorted(assets | required))):
        response = host.run(["sha256sum", *paths], runtime=runtime, timeout=60)
        received = [line.split()[0] for line in response.stdout.decode().splitlines()]
        if len(received) != len(paths):
            raise ValueError(f"Incomplete staged asset receipt on {role}")
        mismatches = [path for path, digest in zip(paths, received, strict=True) if sha256(path) != digest]
        if mismatches:
            raise ValueError(f"Staged assets differ on {role}: {mismatches}")
    workspace = PurePosixPath(site.get("source_workspace", "/vllm-workspace"))
    revisions = {}
    for name, tree in (("vllm", "vllm"), ("ascend", "vllm-ascend")):
        git = ["git", "-C", str(workspace / tree)]
        revisions[name] = host.run([*git, "rev-parse", "HEAD"], runtime=True).stdout.decode().strip()
        if host.run([*git, "status", "--porcelain", "--untracked-files=no"], runtime=True).stdout.strip():
            raise ValueError(f"Tracked changes in the pinned {tree} checkout on {role}")
    script = "from importlib.metadata import version; print(version('mooncake-transfer-engine-npu'))"
    revisions["mooncake"] = host.run([host.runtime_python, "-c", script], runtime=True).stdout.decode().strip()
    if revisions != reference[role]:
        raise ValueError(f"Pinned software does not match on {role}: {revisions}")
    model = PurePosixPath(config[role]["model_path"])
    names = list(reference["model_hashes"])
    hashes = host.run(["sha256sum", *(str(model / name) for name in names)], runtime=True, timeout=60)
    if [line.split()[0] for line in hashes.stdout.decode().splitlines()] != list(reference["model_hashes"].values()):
        raise ValueError(f"Pinned model metadata or tokenizer differs on {role}")
    if role == "D":
        host.run(["test", "-f", site["upstream_proxy"]], runtime=True)
        host.run([host.runtime_python, "-c", "import httpx"], runtime=True)
    return {"software": revisions, "model_hashes": reference["model_hashes"], "idle_npu_smi": idle(host)}


def hold_child(argv):
    """Stop only this session's child group when its controlling SSH stdin closes."""
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        while child.poll() is None and not stopping:
            if select.select([sys.stdin], [], [], .2)[0]:
                stopping = True
    finally:
        if child.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    return 0 if stopping else child.returncode


class Job:
    def __init__(self, host, argv, output, *, runtime=False, guarded=True):
        self.output = output
        if guarded:
            argv = host.python(MODULE, "_hold", *argv, runtime=runtime)
        with output.open("xb") as log:
            self.process = subprocess.Popen(host.command(argv, runtime), stdin=subprocess.PIPE,
                                            stdout=log, stderr=subprocess.STDOUT)

    def __enter__(self):
        return self

    def ensure_running(self):
        if self.process.poll() is not None:
            raise RuntimeError(f"Process exited during the experiment. See {self.output}")

    def close(self):
        self.process.stdin.close()
        try:
            return self.process.wait(timeout=150)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError(f"SSH session did not close. See {self.output}")

    def __exit__(self, *_exc):
        self.close()

    def event(self, kind):
        for line in self.output.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("event") == kind:
                return row
        return None


def wait_for(probe, live, seconds, description):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for job in live:
            job.ensure_running()
        ready = probe()
        if ready:
            return ready
        time.sleep(.5)
    raise TimeoutError(f"Timed out waiting for {description}")


def state_ready(host, folder, label, role, config):
    script = "from pathlib import Path; import sys; p=Path(sys.argv[1]); print(p.read_text()) if p.is_file() else sys.exit(1)"
    response = host.run([host.host_python, "-c", script, f"{folder}/state.json"], check=False)
    if response.returncode:
        return None
    state = json.loads(response.stdout)
    if state["stage"] == "ready":
        if (state["run_id"], state["role"], state["store_pool_layout"]) != (label, role, store_pool_layout(config)):
            raise ValueError("Serving state does not match the expected arm and layout")
        return state
    return None


def http_get(host, url):
    script = """import sys, urllib.request
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(sys.argv[1], timeout=2) as response:
        sys.stdout.buffer.write(response.read())
except OSError:
    sys.exit(1)
"""
    probe = host.run([host.runtime_python, "-c", script, url], runtime=True, check=False)
    return probe.stdout.decode() if probe.returncode == 0 else None


def proxy_state(host, endpoint, *, require_idle=False):
    body = http_get(host, f"{endpoint}/healthcheck")
    if body is not None:
        state = json.loads(body)
        if (state.get("prefill_instances") == state.get("decode_instances") == 2
                and (not require_idle or state.get("request_num") == 0)):
            return state
    return None


def store_snapshot(host, config, *, registered):
    body = http_get(host, f"http://{config['D']['host']}:{config['mooncake_metrics_port']}/metrics")
    if body is None:
        return None
    metrics = parse_metrics(body.splitlines())
    key_count = metrics.get(("master_key_count", ""))
    if key_count is None:
        return None
    if key_count != 0:
        raise ValueError("The new Store did not report zero keys")
    capacity = {label: size for (name, label), size in metrics.items() if name == "segment_total_capacity_bytes"}
    expected = Counter({config[role]["host"]: 8 for role in ("P", "D")})
    actual = Counter(label.removeprefix('segment="').removesuffix('"').rsplit(":", 1)[0] for label in capacity)
    if registered and (actual != expected or any(size != config["store_pool"]["global_segment_size"]
                                                  for size in capacity.values())):
        return None
    return {"key_count": 0, "segment_capacity_bytes": capacity}


def telemetry_ready(host, folder):
    inventory = f"{folder}/host-telemetry/inventory.json"
    sample = f"{folder}/host-telemetry/host-samples.jsonl.gz"
    if host.run(["test", "-s", sample], check=False).returncode:
        return False
    devices = json.loads(host.run(["cat", inventory]).stdout)["hccn_stats"]
    if not all({"mac_tx_total_oct_num", "mac_rx_total_oct_num"} <= devices[str(port)]["raw_counters"].keys()
               for port in PORTS):
        raise RuntimeError(f"Missing NPU TX/RX counters on {host.ssh}")
    return True


def host_archive(host, folder, output):
    with output.open("xb") as stream:
        result = subprocess.run(host.command(["tar", "-czf", "-", "-C", folder, "."]), stdout=stream)
    if result.returncode:
        raise RuntimeError(f"Archive incomplete: {output}")


def run_arm(arm, config, deployment, site, hosts, suite_id, output):
    label = f"{suite_id}-{arm.name}"
    folders = {role: f"runs/{suite_id}/{arm.name}/{role}" for role in hosts}
    endpoint = f"http://127.0.0.1:{config['proxy_port']}"
    output.mkdir()
    write_json(output / "case.json", asdict(arm))
    live = []
    decode, d_folder = hosts["D"], folders["D"]
    with ExitStack() as archives:
        for role, host in hosts.items():
            idle(host)
            host.run(["mkdir", "-p", str(PurePosixPath(folders[role]).parent)])
            host.run(["mkdir", folders[role]])
            archives.callback(host_archive, host, folders[role], output / f"{role}.tar.gz")
        with ExitStack() as stack:
            for role in ("D", "P"):
                host, folder = hosts[role], folders[role]
                args = ["--config", deployment, "--role", role, "--run-dir", folder, "--run-id", label,
                        "--observe-kv-io", "--profile-kv-lifetime", "--observe-abort"]
                if arm.profile and role == "D":
                    args.extend(["--decode-native-profiler", "--profile-decode-steps", "--decode-step-stride", "4"])
                argv = host.python("pd_kv_transfer_opt.experiments.serve", *args, runtime=True)
                live.append(stack.enter_context(Job(host, argv, output / f"{role}-serve.log", runtime=True)))
                if role == "D":
                    wait_for(lambda: store_snapshot(decode, config, registered=False), live, 45, "empty Store Master")
            for role, host in hosts.items():
                state = wait_for(lambda: state_ready(host, folders[role], label, role, config), live,
                                 START_TIMEOUT_S, f"{role} engines")
                write_json(output / f"{role}-ready.json", state)
            store = wait_for(lambda: store_snapshot(decode, config, registered=True), live, 45, "all 16 Store segments")
            write_json(output / "cache-initialization.json", store)
            proxy = decode.python("pd_kv_transfer_opt.experiments.proxy", "--config", deployment, "--run-dir",
                                  f"{d_folder}/proxy", "--routing", "native", "--upstream", site["upstream_proxy"],
                                  runtime=True)
            argv = ["env", "-u", "PDKV_PREFILL_ORDER_PLAN", *proxy]
            live.append(stack.enter_context(Job(decode, argv, output / "proxy.log", runtime=True)))
            state = wait_for(lambda: proxy_state(decode, endpoint, require_idle=True), live, 60, "idle native proxy")
            write_json(output / "proxy-before.json", state)

            holders = {}
            for role, host in hosts.items():
                folder = folders[role]
                shaper = ["--hccn-tool", host.hccn_tool]
                metrics = [value for name, base in endpoints(config).items() if name.startswith(role)
                           for value in ("--metrics", f"{name}={base}/metrics")]
                if role == "D":
                    metrics += ["--metrics", f"Store=http://{config['D']['host']}:{config['mooncake_metrics_port']}/metrics"]
                argv = host.python("pd_kv_transfer_opt.experiments.collect", "--output", f"{folder}/host-telemetry",
                                   "--seconds", ARM_TIMEOUT_S + 240, "--hccn-every", "1.0", "--npu-every", "10",
                                   *metrics, *shaper, *(("--hccn-sudo",) if host.sudo else ()),
                                   *(option for port in PORTS for option in ("--hccn-device", port)))
                live.append(stack.enter_context(Job(host, argv, output / f"{role}-telemetry.log")))
                argv = host.python("pd_kv_transfer_opt.experiments.bandwidth_hold", "--aggregate-gbps", arm.cap,
                                   "--receipt", f"{folder}/shaping.jsonl", "--hold-seconds", ARM_TIMEOUT_S + 120,
                                   *shaper, *(("--sudo",) if host.sudo else ()))
                holder = Job(host, argv, output / f"{role}-shaping.log", guarded=False)
                holders[role] = holder
                live.append(holder)

                def assert_restored(job=holder, name=role):
                    job.close()
                    receipt = job.event("restored")
                    if receipt is None or not receipt["restored"]:
                        raise RuntimeError(f"Restoration on {name} is unverified. See {job.output}")

                stack.callback(assert_restored)
            for role, job in holders.items():
                report = wait_for(lambda: job.event("applied"), live, 120, f"{role} cap readback")
                document = {"role": role, "aggregate_Gbps": report["aggregate_Gbps"], "readback": report["ports"]}
                local = output / f"{role}-shaping-readback.json"
                write_json(local, document)
                verified_shaping(local, arm.cap, role=role)
                decode.put_json(f"{d_folder}/{local.name}", document)
                wait_for(lambda: telemetry_ready(hosts[role], folders[role]), live, 120, f"{role} telemetry")

            args = ["--workload", arm.workload, "--endpoint", endpoint, "--model", config["model_name"],
                    "--output", f"{d_folder}/client", "--run-id", label, "--timeout", "600"]
            if arm.plan:
                module = "pd_kv_transfer_opt.experiments.basin_burst_validation"
                args.extend(["--plan", arm.plan, "--arm", arm.arm, "--cap", arm.cap, "--p-shaping-readback",
                             f"{d_folder}/P-shaping-readback.json", "--d-shaping-readback",
                             f"{d_folder}/D-shaping-readback.json"])
            else:
                module = "pd_kv_transfer_opt.experiments.loadgen"
                spec = SPECS[arm.name.split("-", 1)[0]]
                args.extend(["--window-start", "0", "--seconds", "600", "--return-token-ids", "--compact-json",
                             "--tokenizer-fingerprint", read_json(spec)["tokenizer_fingerprint"]])
            capture = None
            if arm.profile:
                argv = decode.python("pd_kv_transfer_opt.profiling.r43_decode_capture", "--config", deployment,
                                     "--workload-manifest", SPECS["r43"], "--shaping", f"{d_folder}/D-shaping-readback.json",
                                     "--run-dir", d_folder, "--output", f"{d_folder}/decode-windows.json",
                                     "--done-file", f"{d_folder}/client/client-end.json", runtime=True)
                capture = stack.enter_context(Job(decode, argv, output / "r43-capture.log", runtime=True))
            argv = decode.python(module, *args, runtime=True)
            client = stack.enter_context(Job(decode, argv, output / "client.log", runtime=True))
            wait_for(lambda: client.process.poll() is not None, live, ARM_TIMEOUT_S, "client drain")
            if client.process.returncode:
                raise RuntimeError(f"Client run failed. See {client.output}")
            if capture is not None:
                wait_for(lambda: capture.process.poll() is not None, live, 240, "CANN capture completion")
                if capture.process.returncode:
                    raise RuntimeError(f"CANN capture failed. See {capture.output}")
                receipt = json.loads(decode.run(["cat", f"{d_folder}/decode-windows.json"]).stdout)
                write_json(output / "decode-windows.json", receipt)
                if receipt["stage"] != "captured":
                    print(f"Missing decode windows: {receipt['missing_windows']}", file=sys.stderr, flush=True)
            if not arm.plan:
                decode.run(decode.python("pd_kv_transfer_opt.experiments.analyze", f"{d_folder}/client",
                                        runtime=True), runtime=True, timeout=120)
            summary = json.loads(decode.run(["cat", f"{d_folder}/client/summary.json"]).stdout)
            write_json(output / "summary.json", summary)
            state = proxy_state(decode, endpoint)
            write_json(output / "proxy-after.json", state)
            if arm.arm != "burst-terminate":
                if (not summary["quality"]["client_accounting_complete"]
                        or summary["counts"].get("ok") != arm.requests):
                    raise RuntimeError("Incomplete request accounting in this arm")
        return summary["finite_cohort"].get("output_tokens_per_s")


def main():
    if sys.argv[1:2] == ["_hold"]:
        raise SystemExit(hold_child(sys.argv[2:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--group", choices=("formal", "r43", "r69", "basin", "basin-termination",
                                            "cancel-check", "r43-profile"),
                        default="formal")
    parser.add_argument("--include-qualified-termination", action="store_true")
    parser.add_argument("--termination-qualification", type=Path,
                        help="Reviewed cancellation receipt from the live short diagnostic")
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-bundles", type=Path, help="Build two local staging packs without SSH")
    parser.add_argument("--stage-bundle", type=Path, help="Directory with the local P.tar.gz and D.tar.gz packs")
    parser.add_argument("--execute", action="store_true", help="Start the allocated experiment hosts")
    args = parser.parse_args()
    site = read_json(args.site)
    deployment = Path(site["deployment"])
    if deployment.is_absolute() or ".." in deployment.parts:
        raise ValueError("Use the same checkout-relative deployment config on both hosts and the controller")
    config = read_deployment(deployment)
    template = read_deployment(BASELINE)
    for key in ("api_server_count", "proxy_workers", "proxy_port", "mooncake_rpc_port", "mooncake_metrics_port"):
        if config[key] != template[key]:
            raise ValueError(f"The site setting {key} differs from the experiment baseline")
    for role in ("P", "D"):
        for key in ("enable_prefix_caching", "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
                    "compilation_config", "recompute_scheduler_enable", "devices", "api_ports", "kv_ports"):
            if config[role].get(key) != template[role].get(key):
                raise ValueError(f"The site setting {role}.{key} differs from the experiment baseline")
    direct_hosts = [config[role]["host"] for role in ("P", "D")]
    hosts = {role: Host(site["hosts"][role], direct_hosts) for role in ("P", "D")}
    if hosts["P"].ssh == hosts["D"].ssh:
        raise ValueError("The two roles need distinct SSH destinations")
    if args.prepare_bundles and (args.execute or args.stage_bundle):
        parser.error("Bundle preparation is offline")
    cases = arms(args.group, args.include_qualified_termination)
    if args.prepare_bundles:
        print(json.dumps(prepare_bundles(args.prepare_bundles, deployment, cases), indent=2))
        return
    if not args.execute:
        print(json.dumps({"store": store_pool_layout(config), "arms": [asdict(arm) for arm in cases]}, indent=2))
        return
    if not args.run_id or not args.output or args.run_id in (".", "..") or not all(
            char.isascii() and (char.isalnum() or char in "-_.") for char in args.run_id):
        parser.error("Execution requires a fresh --output and a simple --run-id")
    qualification = None
    if args.group != "cancel-check" and any(arm.arm == "burst-terminate" for arm in cases):
        if args.termination_qualification is None:
            parser.error("Run the short live cancellation diagnostic before running qualified termination arms")
        qualification = read_json(args.termination_qualification)
        if (not qualification.get("run_id") or not qualification.get("server_abort_and_resource_release_verified")
                or set(qualification.get("covered_stages", [])) != {"prefill", "decode_kv_wait", "decode_generation"}):
            parser.error("Cancellation qualification must cover P in flight, D KV wait and D generation")
    reference = read_json(SOFTWARE_REFERENCE)["software"]
    folder = new_directory(args.output)
    if qualification:
        write_json(folder / "termination-qualification.json", qualification)
    if args.stage_bundle:
        receipts = read_json(args.stage_bundle / "manifest.json")
        for role in hosts:
            if sha256(args.stage_bundle / f"{role}.tar.gz") != receipts[role]["sha256"]:
                raise ValueError(f"The {role} staging archive differs from its receipt")
        for host in hosts.values():
            idle(host)
        for role, host in hosts.items():
            host.stage(args.stage_bundle / f"{role}.tar.gz")
    for role, host in hosts.items():
        required = set()
        if role == "D":
            required.update(arm.workload for arm in cases)
            required.update(arm.plan for arm in cases if arm.plan is not None)
            required.update(SPECS[arm.name.split("-", 1)[0]] for arm in cases if arm.plan is None)
        readiness = verify_stage(host, role, required, deployment, config, site, reference)
        write_json(folder / f"{role}-preflight.json", readiness)
    for arm in cases:
        print(f"Starting {arm.name} with {arm.cap} Gbps per host", flush=True)
        tps = run_arm(arm, config, deployment.as_posix(), site, hosts, args.run_id, folder / arm.name)
        print(f"Finished {arm.name}, throughput={tps}", flush=True)


if __name__ == "__main__":
    main()
