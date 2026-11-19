"""Launch the two local vLLM-Ascend instances for one role of the 2P2D deployment."""

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from pd_kv_transfer_opt.common import read_json, write_json


ROLES = ("P", "D")
PROFILE_CONFIG = {"profiler": "torch", "torch_profiler_with_stack": False, "torch_profiler_with_memory": False}


def read_deployment(path):
    config = read_json(path)
    if config["P"]["host"] == config["D"]["host"]:
        raise ValueError("The two roles require different hosts.")
    for role in ROLES:
        item = config[role]
        if "enable_prefix_caching" in item and type(item["enable_prefix_caching"]) is not bool:
            raise ValueError(f"{role}.enable_prefix_caching must be a boolean when specified.")
        if sorted(device for group in item["devices"] for device in group) != list(range(8)):
            raise ValueError(f"{role} must use each device 0–7 exactly once.")
        if [len(group) for group in item["devices"]] != [4, 4]:
            raise ValueError("TP4/EP4 requires four devices per instance.")
        ports = item["api_ports"] + item["kv_ports"]
        if role == "D":
            ports += [config["mooncake_rpc_port"], config["mooncake_metrics_port"], config["proxy_port"]]
        if len(item["api_ports"]) != 2 or len(item["kv_ports"]) != 2 or len(set(ports)) != len(ports):
            raise ValueError(f"Invalid or conflicting {role} ports.")
    return config


def endpoints(config):
    return {f"{role}{idx}": f"http://{config[role]['host']}:{port}"
            for role in ROLES for idx, port in enumerate(config[role]["api_ports"])}


def model_commands(config, role, run_dir, *, observe=False):
    local = config[role]
    producer = role == "P"
    kv_role = "kv_producer" if producer else "kv_consumer"
    store = {"metadata_server": "P2PHANDSHAKE", "protocol": "ascend", "device_name": "",
             "master_server_address": f"{config['D']['host']}:{config['mooncake_rpc_port']}",
             "global_segment_size": 0 if producer else 16 * 1024**3,
             "local_buffer_size": 32 * 1024**2, "preferred_segment": not producer,
             "prefer_alloc_in_same_node": not producer}
    graph = local.get("compilation_config", {
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 1024, 2048, 3072, 4096],
    } if producer else {
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16], "cudagraph_mode": "FULL_DECODE_ONLY",
    })
    commands = []
    for idx, (api_port, kv_port, devices) in enumerate(zip(local["api_ports"], local["kv_ports"], local["devices"], strict=True)):
        label = f"{role}{idx}"
        pd = {"kv_connector": "MooncakeConnectorV1", "kv_role": kv_role, "kv_buffer_device": "npu",
              "kv_port": kv_port, "kv_parallel_size": 1, "kv_rank": 0 if producer else 1,
              "kv_connector_extra_config": {"prefill": {"dp_size": 1, "tp_size": 4},
                                            "decode": {"dp_size": 1, "tp_size": 4}}}
        # The lookup socket is local IPC, but independently launched instances on a host need different IDs.
        extra = {"backend": "mooncake", "lookup_rpc_port": str(api_port), "load_async": False}
        if not producer:
            extra["consumer_is_to_put"] = True
        pool = {"kv_connector": "AscendStoreConnector", "kv_role": kv_role, "kv_connector_extra_config": extra}
        transfer = {"kv_connector": "MultiConnector", "kv_role": kv_role, "kv_load_failure_policy": "fail",
                    "kv_connector_extra_config": {"connectors": [pd, pool]}}
        argv = ["vllm", "serve", local["model_path"], "--host", local["host"], "--port", str(api_port),
                "--api-server-count", str(config["api_server_count"]), "--served-model-name", config["model_name"],
                "--dtype", "bfloat16", "--tensor-parallel-size", "4", "--data-parallel-size", "1",
                "--enable-expert-parallel", "--distributed-executor-backend", "mp",
                "--max-model-len", "262144", "--max-num-seqs", str(local.get("max_num_seqs", 16)),
                "--max-num-batched-tokens", str(local.get("max_num_batched_tokens", 4096)),
                "--gpu-memory-utilization", str(local.get("gpu_memory_utilization", 0.8)),
                "--seed", "20261117", "--generation-config", "vllm",
                "--block-size", "128", "--mamba-cache-mode", "align",
                "--additional-config", json.dumps({"enable_cpu_binding": False}),
                "--kv-transfer-config", json.dumps(transfer), "--compilation-config", json.dumps(graph)]
        if "enable_prefix_caching" in local:
            argv.append("--enable-prefix-caching" if local["enable_prefix_caching"] else "--no-enable-prefix-caching")
        if observe:
            profiler = dict(PROFILE_CONFIG, torch_profiler_dir=str(run_dir / "native-process-profile" / label))
            argv.extend(["--profiler-config", json.dumps(profiler)])
        cpu_sets = local.get("cpu_sets")
        if cpu_sets:
            argv = ["taskset", "-c", cpu_sets[idx], *argv]
        commands.append({"id": label, "devices": devices, "port": api_port, "argv": argv})
    return store, commands


def launch(config, role, run_dir, run_id, state, store):
    from vllm.logger import DEFAULT_LOGGING_CONFIG

    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logging = copy.deepcopy(DEFAULT_LOGGING_CONFIG)
    mode_file = state.get("same_head_mode_file")
    if state["observations_enabled"]:
        name = ("pd_kv_transfer_opt.experiments.same_head_kv.InstallFilter" if mode_file is not None
                else "pd_kv_transfer_opt.profiling.rdma_full_observability.InstallFilter")
        for handler in logging["handlers"].values():
            handler.setdefault("filters", []).append("rdma_observations")
        logging.setdefault("filters", {})["rdma_observations"] = {"()": name}
    write_json(run_dir / "logging.json", logging)
    write_json(run_dir / "store.json", store)
    local = config[role]
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONHASHSEED="0", VLLM_HOST_IP=local["host"],
               HCCL_INTRA_ROCE_ENABLE="1", ASCEND_CONNECT_TIMEOUT="10000", ASCEND_TRANSFER_TIMEOUT="10000",
               ACL_OP_INIT_MODE="1", MC_TRANSFER_TIMEOUT="30", ISSUE23_ROLE=role,
               ISSUE23_RUN_ID=run_id, ISSUE23_P_GRAPH_REPLAY_OBSERVABILITY="1" if state["observations_enabled"] else "0",
               ISSUE23_BATCH_GET_CONCURRENCY="1",
               VLLM_LOGGING_CONFIG_PATH=str(run_dir / "logging.json"), MOONCAKE_CONFIG_PATH=str(run_dir / "store.json"))
    if mode_file is not None:
        env["ISSUE23_SAME_HEAD_MODE"] = mode_file
    repo = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(item for item in (repo, env.get("PYTHONPATH")) if item)
    children = []
    stop = False

    def on_signal(_signum, _frame):
        nonlocal stop
        stop = True

    def save(stage):
        state["stage"] = stage
        tmp = run_dir / "state.tmp"
        write_json(tmp, state)
        tmp.replace(run_dir / "state.json")

    def spawn(argv, process_env, name):
        with (logs / f"{name}.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(argv, env=process_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, on_signal)
    if (run_dir / "state.json").exists():
        raise ValueError("Use a fresh run directory for this host.")
    save("starting")
    start = time.monotonic()
    try:
        if role == "D":
            master = importlib.metadata.distribution("mooncake-transfer-engine-npu").locate_file("mooncake/mooncake_master")
            spawn([str(master), f"--rpc_address={local['host']}", f"--rpc_port={config['mooncake_rpc_port']}",
                   f"--metrics_port={config['mooncake_metrics_port']}", "--enable_metric_reporting=true",
                   "--default_kv_lease_ttl=11000", "--logtostderr=true"], env, "master")
        for item in state["instances"]:
            process_env = dict(env, ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, item["devices"])))
            item["pid"] = spawn(item["argv"], process_env, item["id"]).pid
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        ready = False
        while not stop and not (run_dir / "STOP").exists():
            if any(child.poll() is not None for child in children):
                raise RuntimeError("A serving process exited. See the run logs.")
            if not ready:
                for item in state["instances"]:
                    try:
                        with opener.open(f"{endpoints(config)[item['id']]}/health", timeout=2) as response:
                            if response.status != 200:
                                break
                    except OSError:
                        break
                else:
                    ready = True
                    save("ready")
                if not ready and time.monotonic() - start > 900:
                    raise TimeoutError("Local engines did not become healthy.")
            time.sleep(1)
    except BaseException as error:
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        save("finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--observe", action="store_true", help="Enable method and NPU event observations for profiling runs.")
    parser.add_argument("--same-head-kv", action="store_true", help="Enable the P-side same-head A/B mode file.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = read_deployment(args.config)
    run_dir = args.run_dir.resolve()
    observe = args.observe or args.same_head_kv
    store, commands = model_commands(config, args.role, run_dir, observe=observe)
    state = {"run_id": args.run_id, "role": args.role, "instances": commands, "store_config": store,
             "observations_enabled": observe}
    if args.same_head_kv:
        if args.role != "P" or config["P"].get("enable_prefix_caching") is not False:
            parser.error("--same-head-kv requires P with local prefix caching disabled")
        state["same_head_mode_file"] = str(run_dir / "same-head-mode")
    if args.dry_run:
        print(json.dumps(state, indent=2))
        return
    launch(config, args.role, run_dir, args.run_id, state, store)


if __name__ == "__main__":
    main()
