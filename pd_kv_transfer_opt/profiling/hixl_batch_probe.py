"""Measure fixed-byte batches on task-owned NPU buffers with Mooncake TransferEngine."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from statistics import median
import time

from pd_kv_transfer_opt.common import read_json, write_json


KIB = 1024
SHAPES = (
    ("state", 64, 12 * KIB, (1, 4, 16, 64)),
    ("attention", 840, 512 * KIB, (32, 128, 256, 840)),
)
GAP = 4 * KIB


def layouts():
    offset = 0
    for name, count, size, batches in SHAPES:
        span = count * (size + GAP)
        yield name, count, size, batches, offset, span
        offset += span


def signature(torch, count):
    values = torch.arange(count, dtype=torch.int32)
    return (values % 127 + 1).to(torch.uint8)


def source_tensor(torch, total, device):
    host = torch.full((total,), 191, dtype=torch.uint8)
    for _, count, size, _, start, span in layouts():
        rows = host.narrow(0, start, span).view(count, size + GAP)
        rows[:, 0] = signature(torch, count)
        rows[:, size - 1] = 233 - rows[:, 0]
    return host.to(device)


def verify(torch, tensor, count, size, start, span):
    rows = tensor.narrow(0, start, span).view(count, size + GAP)
    first, middle, last, guard = (rows[:, col].cpu() for col in (0, size // 2, size - 1, size))
    expected = signature(torch, count)
    if not (torch.equal(first, expected) and torch.equal(last, 233 - expected)
            and bool((middle == 191).all()) and bool((guard == 0).all())):
        raise ValueError("RDMA payload or guard bytes do not match")


def transfer(torch, engine, tensor, peer, repetitions):
    reports = {}
    for name, count, size, batches, start, span in layouts():
        source = [peer["pointer"] + start + index * (size + GAP) for index in range(count)]
        dest = [tensor.data_ptr() + start + index * (size + GAP) for index in range(count)]
        size_list = [size] * count
        reports[name] = {}
        commands = {}
        for batch in batches:
            commands[batch] = [(dest[left:left + batch], source[left:left + batch], size_list[left:left + batch])
                               for left in range(0, count, batch)]
        session = peer["session"]
        tensor.narrow(0, start, span).zero_()
        torch.npu.synchronize()

        def run(batch):
            begin = time.monotonic_ns()
            for local, remote, sizes in commands[batch]:
                status = engine.batch_transfer_sync_read(session, local, remote, sizes)
                if status != 0:
                    raise RuntimeError(f"Mooncake READ returned {status} at {name}/{batch}")
            return time.monotonic_ns() - begin

        for batch in batches:
            run(batch)
        samples = {batch: [] for batch in batches}
        for repeat in range(repetitions):
            # Alternate repetitions so connection warmup and ordinary drift do not favor large batches.
            order = batches if repeat % 2 == 0 else batches[::-1]
            for batch in order:
                samples[batch].append(run(batch))
        for batch in batches:
            # Check identical bytes and untouched 4 KiB guard after a fresh transfer for every vector.
            tensor.narrow(0, start, span).zero_()
            torch.npu.synchronize()
            run(batch)
            verify(torch, tensor, count, size, start, span)
            observed = samples[batch]
            reports[name][str(batch)] = {
                "payload_bytes": count * size, "descriptor_count": count, "descriptor_bytes": size,
                "sync_calls_per_transfer": len(commands[batch]), "samples_ns": observed,
                "transfer_ms_p50": median(observed) / 1e6,
                "effective_Gbps_p50": 8 * count * size / median(observed), "payload_and_guards_checked": True,
            }
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("source", "reader"))
    parser.add_argument("--host", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peer", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--port", type=int, default=26749)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    if args.role == "reader" and args.peer is None:
        parser.error("Reader requires metadata for the owned source buffer")
    if args.repetitions < 1:
        parser.error("Repetitions must be positive")

    import torch
    import torch_npu  # noqa: F401
    from mooncake.engine import TransferEngine

    os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] = json.dumps({"comm_resource_config.listen_port": args.port})
    device = f"npu:{args.device}"
    torch.npu.set_device(args.device)
    total = sum(span for _, _, _, _, _, span in layouts())
    if args.role == "source":
        tensor = source_tensor(torch, total, device)
    else:
        tensor = torch.zeros((total,), device=device, dtype=torch.uint8)
    torch.npu.synchronize()
    engine = TransferEngine()
    if engine.initialize(args.host, "P2PHANDSHAKE", "ascend", "") or engine.register_memory(tensor.data_ptr(), total):
        raise RuntimeError("Mooncake engine initialization or memory registration failed")

    try:
        if args.role == "source":
            stop = args.output.with_suffix(".stop")
            finished = args.output.with_suffix(".closed")
            if any(path.exists() for path in (args.output, stop, finished)):
                raise ValueError("Use an unused output path for the private source registration")
            temporary = args.output.with_suffix(".tmp")
            write_json(temporary, {"session": f"{args.host}:{engine.get_rpc_port()}",
                                   "pointer": tensor.data_ptr(), "bytes": total})
            temporary.replace(args.output)
            deadline = time.monotonic() + 360
            while not stop.exists() and time.monotonic() < deadline:
                time.sleep(0.2)
        else:
            peer = read_json(args.peer)
            if peer["bytes"] != total or peer["session"].rsplit(":", 1)[0] == args.host:
                raise ValueError("The peer must own an equally sized registered buffer on the other lab host")
            results = transfer(torch, engine, tensor, peer, args.repetitions)
    finally:
        if engine.unregister_memory(tensor.data_ptr()):
            raise RuntimeError("Mooncake buffer unregistration failed")
    if args.role == "source":
        finished.touch()
    else:
        temporary = args.output.with_suffix(".tmp")
        write_json(temporary, {"wheel": importlib.metadata.version("mooncake-transfer-engine-npu"),
                               "async_setting_present": "ASCEND_USE_ASYNC_TRANSFER" in os.environ,
                               "device": args.device, "shapes": results, "registered_bytes": total,
                               "unregistered": True})
        temporary.replace(args.output)
    # This disposable actor has finished and unregistered its buffer. Let the process
    # release its runtime instead of running the wheel's native shutdown at interpreter exit.
    os._exit(0)


if __name__ == "__main__":
    main()
