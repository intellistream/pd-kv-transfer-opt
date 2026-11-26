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
STORE_RANK_GIB_BY_TP = {2: 16, 4: 8}
STORE_LOCAL_OPTIONS = {
    "local_buffer_size": 32 * 1024**2,
    "preferred_segment": True,
    "prefer_alloc_in_same_node": True,
}


def store_pool_options(config):
    tp_size = len(config["P"]["devices"][0])
    rank_gib = STORE_RANK_GIB_BY_TP[tp_size]
    expected = {"global_segment_size": rank_gib * 1024**3, **STORE_LOCAL_OPTIONS}
    options = config.get("store_pool", expected)
    if (not isinstance(options, dict) or options.keys() != expected.keys()
            or any(type(options[key]) is not type(value) or options[key] != value
                   for key, value in expected.items())):
        raise ValueError(f"store_pool must match the TP{tp_size} layout: {expected}")
    return options.copy()


def store_pool_layout(config):
    options = store_pool_options(config)
    workers = {role: sum(map(len, config[role]["devices"])) for role in ROLES}
    segment_size = options["global_segment_size"]
    capacity = {role: count * segment_size for role, count in workers.items()}
    return {"id": f"two-sided-{segment_size // 1024**3}gib-local-first/v1", "model_workers_by_role": workers,
            "capacity_bytes_by_role": capacity, "total_capacity_bytes": sum(capacity.values())}


def read_deployment(path):
    config = read_json(path)
    if config["P"]["host"] == config["D"]["host"]:
        raise ValueError("The two roles require different hosts.")
    for role in ROLES:
        item = config[role]
        if {"store_pool", "global_segment_size", *STORE_LOCAL_OPTIONS} & item.keys():
            raise ValueError("Use the common top-level store_pool on both roles.")
        for key in ("enable_prefix_caching", "enable_matmul_allreduce"):
            if key in item and type(item[key]) is not bool:
                raise ValueError(f"{role}.{key} must be a boolean when specified.")
        recompute = item.get("recompute_scheduler_enable", False)
        if type(recompute) is not bool or (recompute and role != "D"):
            raise ValueError("recompute_scheduler_enable must be a boolean and may only be enabled on D.")
        groups = item["devices"]
        sizes = [len(group) for group in groups]
        if sizes not in ([2, 2], [4, 4]):
            raise ValueError(f"{role} requires two equally sized TP2 or TP4 device groups.")
        devices = [device for group in groups for device in group]
        if any(type(device) is not int or not 0 <= device < 8 for device in devices) or len(set(devices)) != len(devices):
            raise ValueError(f"{role} requires distinct device IDs in the range 0–7.")
        ports = item["api_ports"] + item["kv_ports"]
        if role == "D":
            ports += [config["mooncake_rpc_port"], config["mooncake_metrics_port"], config["proxy_port"]]
        if len(item["api_ports"]) != 2 or len(item["kv_ports"]) != 2 or len(set(ports)) != len(ports):
            raise ValueError(f"Invalid or conflicting {role} ports.")
    if len(config["P"]["devices"][0]) != len(config["D"]["devices"][0]):
        raise ValueError("P and D must use the same TP size.")
    config["store_pool"] = store_pool_options(config)
    return config


def endpoints(config):
    return {f"{role}{idx}": f"http://{config[role]['host']}:{port}"
            for role in ROLES for idx, port in enumerate(config[role]["api_ports"])}


def mooncake_store_config(config):
    return {"metadata_server": "P2PHANDSHAKE", "protocol": "ascend", "device_name": "",
            "master_server_address": f"{config['D']['host']}:{config['mooncake_rpc_port']}",
            **store_pool_options(config)}


def store_environment(store, run_dir):
    # The pinned backend reads the two overrides before the generated JSON.
    return {"MOONCAKE_CONFIG_PATH": str(run_dir / "store.json"),
            "MOONCAKE_MASTER": store["master_server_address"],
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": str(store["global_segment_size"]),
            "ASCEND_ENABLE_USE_FABRIC_MEM": "0"}


def model_commands(config, role, run_dir, *, observe=False, profile_decode_steps=False,
                   decode_language_model_only=False, decode_native_profiler=False):
    local = config[role]
    tp_size = len(local["devices"][0])
    producer = role == "P"
    kv_role = "kv_producer" if producer else "kv_consumer"
    store = mooncake_store_config(config)
    graph = local.get("compilation_config", {
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 1024, 2048, 3072, 4096],
    } if producer else {
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16], "cudagraph_mode": "FULL_DECODE_ONLY",
    })
    additional_config = {"enable_cpu_binding": False}
    for key in ("recompute_scheduler_enable", "enable_matmul_allreduce"):
        if key in local:
            additional_config[key] = local[key]
    commands = []
    for idx, (api_port, kv_port, devices) in enumerate(zip(local["api_ports"], local["kv_ports"], local["devices"], strict=True)):
        label = f"{role}{idx}"
        pd = {"kv_connector": "MooncakeConnectorV1", "kv_role": kv_role, "kv_buffer_device": "npu",
              "kv_port": kv_port, "kv_parallel_size": 1, "kv_rank": 0 if producer else 1,
              "kv_connector_extra_config": {"prefill": {"dp_size": 1, "tp_size": tp_size},
                                            "decode": {"dp_size": 1, "tp_size": tp_size}}}
        # The lookup socket is local IPC, but independently launched instances on a host need different IDs.
        extra = {"backend": "mooncake", "lookup_rpc_port": str(api_port), "load_async": False}
        if not producer:
            extra["consumer_is_to_put"] = True
        pool = {"kv_connector": "AscendStoreConnector", "kv_role": kv_role, "kv_connector_extra_config": extra}
        transfer = {"kv_connector": "MultiConnector", "kv_role": kv_role, "kv_load_failure_policy": "fail",
                    "kv_connector_extra_config": {"connectors": [pd, pool]}}
        argv = ["vllm", "serve", local["model_path"], "--host", local["host"], "--port", str(api_port),
                "--api-server-count", str(config["api_server_count"]), "--served-model-name", config["model_name"],
                "--dtype", "bfloat16", "--tensor-parallel-size", str(tp_size), "--data-parallel-size", "1",
                "--enable-expert-parallel", "--distributed-executor-backend", "mp",
                "--max-model-len", "262144", "--max-num-seqs", str(local.get("max_num_seqs", 16)),
                "--max-num-batched-tokens", str(local.get("max_num_batched_tokens", 4096)),
                "--gpu-memory-utilization", str(local.get("gpu_memory_utilization", 0.8)),
                "--seed", "20261117", "--generation-config", "vllm",
                "--block-size", "128", "--mamba-cache-mode", "align",
                "--additional-config", json.dumps(additional_config),
                "--kv-transfer-config", json.dumps(transfer), "--compilation-config", json.dumps(graph)]
        if "enable_prefix_caching" in local:
            argv.append("--enable-prefix-caching" if local["enable_prefix_caching"] else "--no-enable-prefix-caching")
        if decode_language_model_only:
            argv.append("--language-model-only")
        if observe or decode_native_profiler:
            profiler = dict(PROFILE_CONFIG, torch_profiler_dir=str(run_dir / "native-process-profile" / label))
            if profile_decode_steps or decode_native_profiler:
                profiler["ignore_frontend"] = True
            argv.extend(["--profiler-config", json.dumps(profiler)])
        cpu_sets = local.get("cpu_sets")
        if cpu_sets:
            argv = ["taskset", "-c", cpu_sets[idx], *argv]
        commands.append({"id": label, "devices": devices, "port": api_port, "argv": argv})
    return store, commands


def launch(config, role, run_dir, run_id, state, store):
    if store != mooncake_store_config(config):
        raise ValueError("The worker Store configuration differs from the validated deployment.")
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
    elif state.get("kv_io_observations_enabled", False):
        for handler in logging["handlers"].values():
            handler.setdefault("filters", []).append("kv_io_observations")
        logging.setdefault("filters", {})["kv_io_observations"] = {
            "()": "pd_kv_transfer_opt.profiling.scoped_native_observability.InstallFilter",
        }
    if state.get("page_observations_enabled", False) or state.get("abort_observations_enabled", False):
        for handler in logging["handlers"].values():
            handler.setdefault("filters", []).append("kv_page_observations")
        logging.setdefault("filters", {})["kv_page_observations"] = {
            "()": "pd_kv_transfer_opt.profiling.kv_lifetime_observability.InstallFilter",
            "lifetimes": state.get("page_observations_enabled", False),
            "aborts": state.get("abort_observations_enabled", False),
        }
    if state.get("prefill_admission_observations", False):
        for handler in logging["handlers"].values():
            handler.setdefault("filters", []).append("prefill_admission_observations")
        logging.setdefault("filters", {})["prefill_admission_observations"] = {
            "()": "pd_kv_transfer_opt.profiling.prefill_admission_observability.InstallFilter",
        }
    if state.get("decode_step_observations", False):
        for handler in logging["handlers"].values():
            handler.setdefault("filters", []).append("decode_step_observations")
        logging.setdefault("filters", {})["decode_step_observations"] = {
            "()": "pd_kv_transfer_opt.profiling.decode_step_observability.InstallFilter",
        }
    if state.get("prefill_order_enabled", False):
        for handler in logging["handlers"].values():
            handler.setdefault("filters", []).append("prefill_order")
        logging.setdefault("filters", {})["prefill_order"] = {
            "()": "pd_kv_transfer_opt.experiments.prefill_order.InstallFilter",
        }
    write_json(run_dir / "logging.json", logging)
    write_json(run_dir / "store.json", store)
    local = config[role]
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONHASHSEED="0", VLLM_HOST_IP=local["host"],
               HCCL_INTRA_ROCE_ENABLE="1", ASCEND_CONNECT_TIMEOUT="10000", ASCEND_TRANSFER_TIMEOUT="10000",
               ACL_OP_INIT_MODE="1", MC_TRANSFER_TIMEOUT="30", ISSUE23_ROLE=role,
               ISSUE23_RUN_ID=run_id, ISSUE23_P_GRAPH_REPLAY_OBSERVABILITY="1" if state["observations_enabled"] else "0",
               ISSUE23_BATCH_GET_CONCURRENCY="1",
               VLLM_LOGGING_CONFIG_PATH=str(run_dir / "logging.json"), **store_environment(store, run_dir))
    state["inherited_HCCL_RDMA_TIMEOUT"] = env.get("HCCL_RDMA_TIMEOUT")
    state["communication_environment"] = {name: env.get(name) for name in (
        "HCCL_INTRA_ROCE_ENABLE", "HCCL_INTRA_PCIE_ENABLE", "HCCL_INTER_HCCS_DISABLE", "HCCL_RDMA_TIMEOUT")}
    if mode_file is not None:
        env["ISSUE23_SAME_HEAD_MODE"] = mode_file
    if state.get("prefill_admission_observations", False):
        env["ISSUE23_P_ADMISSION_OBSERVABILITY"] = "1"
    if state.get("decode_step_observations", False):
        env["PDKV_DECODE_STEP_STRIDE"] = str(state.get("decode_step_stride", 16))
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
            if state.get("prefill_order_enabled", False):
                process_env.update(PDKV_PREFILL_ORDER_INSTANCE=item["id"],
                                   PDKV_PREFILL_ORDER_DIR=str(run_dir / "prefill-order"))
            if (observer := state.get("adxl_observer")) is not None:
                preloads = [observer, process_env.get("LD_PRELOAD", "")]
                process_env["LD_PRELOAD"] = ":".join(filter(None, preloads))
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
    parser.add_argument("--observe-kv-io", action="store_true", help="Record Store I/O and KV handoffs without device profiling.")
    parser.add_argument("--profile-kv-lifetime", action="store_true",
                        help="Observe scheduler page demand and connector-owned page lifetimes.")
    parser.add_argument("--observe-abort", action="store_true",
                        help="Record scheduler abort and block-release events without device profiling.")
    parser.add_argument("--profile-prefill-admission", action="store_true",
                        help="Join P HTTP processing and EngineCore input events with --observe.")
    parser.add_argument("--profile-decode-steps", action="store_true",
                        help="Sample D steps with --decode-native-profiler or --observe.")
    parser.add_argument("--decode-step-stride", type=int, default=16,
                        help="Decode offset interval for sampled step markers.")
    parser.add_argument("--decode-language-model-only", action="store_true",
                        help="Run the native D language-only input path for text token prompts.")
    parser.add_argument("--decode-native-profiler", action="store_true",
                        help="Expose the native D worker profiler for separate diagnostic windows.")
    parser.add_argument("--adxl-observer", type=Path, help="Optional native ADXL timing library for P workers.")
    parser.add_argument("--same-head-kv", action="store_true", help="Enable the P-side same-head A/B mode file.")
    parser.add_argument("--prefill-order", action="store_true",
                        help="Enable the experimental P waiting-queue order and per-arm audit.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = read_deployment(args.config)
    run_dir = args.run_dir.resolve()
    observe = args.observe or args.same_head_kv
    if args.profile_decode_steps and (args.role != "D" or not (args.observe or args.decode_native_profiler)):
        parser.error("--profile-decode-steps requires --decode-native-profiler or --observe on D")
    if args.decode_step_stride < 1 or (args.decode_step_stride != 16 and not args.profile_decode_steps):
        parser.error("--decode-step-stride must be positive and requires --profile-decode-steps")
    if args.role != "D" and (args.decode_language_model_only or args.decode_native_profiler):
        parser.error("Decode input-path and native profiler options require --role D")
    store, commands = model_commands(config, args.role, run_dir, observe=observe,
                                    profile_decode_steps=args.profile_decode_steps,
                                    decode_language_model_only=args.decode_language_model_only,
                                    decode_native_profiler=args.decode_native_profiler)
    state = {"run_id": args.run_id, "role": args.role, "instances": commands, "store_config": store,
             "store_pool_layout": store_pool_layout(config),
             "store_environment": store_environment(store, run_dir),
             "observations_enabled": observe, "decode_language_model_only": args.decode_language_model_only,
             "decode_native_profiler": args.decode_native_profiler}
    if args.profile_kv_lifetime:
        state["page_observations_enabled"] = True
    if args.observe_kv_io:
        state["kv_io_observations_enabled"] = True
    if args.observe_abort:
        state["abort_observations_enabled"] = True
    if args.profile_prefill_admission:
        if args.role != "P" or not args.observe:
            parser.error("--profile-prefill-admission requires --observe on P")
        state["prefill_admission_observations"] = True
    if args.profile_decode_steps:
        state["decode_step_observations"] = True
        state["decode_step_stride"] = args.decode_step_stride
    if args.adxl_observer is not None:
        if args.role != "P" or not args.observe or not args.adxl_observer.is_file():
            parser.error("--adxl-observer requires an existing library and --observe on P")
        state["adxl_observer"] = str(args.adxl_observer.resolve())
    if args.same_head_kv:
        if args.role != "P" or config["P"].get("enable_prefix_caching") is not False:
            parser.error("--same-head-kv requires P with local prefix caching disabled")
        state["same_head_mode_file"] = str(run_dir / "same-head-mode")
    if args.prefill_order:
        if args.role != "P":
            parser.error("--prefill-order requires --role P")
        state["prefill_order_enabled"] = True
    if args.dry_run:
        print(json.dumps(state, indent=2))
        return
    launch(config, args.role, run_dir, args.run_id, state, store)


if __name__ == "__main__":
    main()
