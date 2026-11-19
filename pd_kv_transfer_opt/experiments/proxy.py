"""Start the pinned upstream P/D proxy with request timing and optional fixed routes."""

import argparse
import json
import os
from pathlib import Path
import sys

from pd_kv_transfer_opt.common import read_json, write_json
from pd_kv_transfer_opt.experiments.rdma_full_protocol import CASES, OUTPUT_TOKENS, request_id
from pd_kv_transfer_opt.experiments.serve import read_deployment


def routes(config):
    result = {}
    for case, _, count in CASES:
        for index in range(count):
            slot = index % 2
            result[request_id(case, index)] = {
                "max_tokens": OUTPUT_TOKENS,
                "prefill": f"{config['P']['host']}:{config['P']['api_ports'][slot]}",
                "decode": f"{config['D']['host']}:{config['D']['api_ports'][slot]}",
            }
    return result


def resolve_fixture(fixture, config):
    addresses = {f"{role}{i}": f"{config[role]['host']}:{port}"
                 for role in ("P", "D") for i, port in enumerate(config[role]["api_ports"])}
    resolved = {}
    for rid, request in fixture["requests"].items():
        prefill, decode = request["prefill"], request["decode"]
        if prefill not in ("P0", "P1") or decode not in ("D0", "D1"):
            raise ValueError(f"Invalid role labels in route fixture for {rid}.")
        resolved[rid] = dict(request, prefill=addresses[prefill], decode=addresses[decode])
    return {"requests": resolved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True, help="Pinned vLLM-Ascend example proxy file.")
    parser.add_argument("--routing", choices=("native", "fixture", "r14"), default="native",
                        help="Use the upstream scheduler unless explicitly replaying routes (default: native).")
    parser.add_argument("--route-fixture", type=Path)
    parser.add_argument("--no-upstream-keepalive", action="store_true",
                        help="Disable proxy-to-engine connection reuse for diagnostic reproduction.")
    args = parser.parse_args()
    if (args.routing == "fixture") != (args.route_fixture is not None):
        parser.error("--route-fixture is required with --routing fixture and only used in that mode")
    config = read_deployment(args.config)
    run_dir = args.run_dir.resolve()
    if not args.upstream.is_file():
        raise FileNotFoundError(args.upstream)
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = run_dir / "fixed-routes.json"
    if args.routing == "r14":
        write_json(fixture_path, {"requests": routes(config)})
    elif args.routing == "fixture":
        write_json(fixture_path, resolve_fixture(read_json(args.route_fixture), config))
    elif fixture_path.exists():
        raise ValueError("Native routing requires a run directory without a route fixture.")
    os.environ.update(PD_KV_RUN_DIR=str(run_dir), PD_KV_PROXY_SOURCE=str(args.upstream.resolve()),
                      ISSUE23_HTTP_NO_KEEPALIVE="1" if args.no_upstream_keepalive else "0",
                      ISSUE23_LIGHT_PROXY_OBSERVATIONS="1")
    sys.argv = [sys.argv[0], "--host", "127.0.0.1", "--port", str(config["proxy_port"]),
                "--prefiller-hosts", *([config["P"]["host"]] * 2),
                "--prefiller-ports", *map(str, config["P"]["api_ports"]),
                "--decoder-hosts", *([config["D"]["host"]] * 2),
                "--decoder-ports", *map(str, config["D"]["api_ports"]), "--max-retries", "1",
                "--workers", str(config["proxy_workers"]), "--log-level", "WARNING"]
    import uvicorn
    from pd_kv_transfer_opt.experiments.proxy_app import events, module

    os.environ["ISSUE23_PROXY_ARGS_JSON"] = json.dumps(vars(module.global_args))
    module.bootstrap_parent_process(module.global_args)
    try:
        uvicorn.run("pd_kv_transfer_opt.experiments.proxy_app:app", host="127.0.0.1",
                    port=config["proxy_port"], workers=config["proxy_workers"], log_level="warning")
    finally:
        module.cleanup_manager_config(config["proxy_port"])
        events.close()


if __name__ == "__main__":
    main()
