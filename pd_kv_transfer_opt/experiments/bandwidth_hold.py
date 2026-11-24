"""Hold a verified per-host RDMA cap while a controller keeps stdin open."""

import argparse
import json
from pathlib import Path
import signal
import sys
import time

from pd_kv_transfer_opt.experiments.hardware import Shaper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-gbps", type=int, required=True, help="Total egress cap across this host's eight ports.")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--hold-seconds", type=int, default=900)
    parser.add_argument("--hccn-tool", default="/usr/local/Ascend/driver/tools/hccn_tool")
    parser.add_argument("--sudo", action="store_true")
    args = parser.parse_args()
    if not 80 <= args.aggregate_gbps <= 1600 or args.hold_seconds < 1:
        parser.error("The cap must be 80–1600 Gbps and hold time must be positive.")

    def interrupt(signum, _frame):
        raise InterruptedError(signal.Signals(signum).name)

    with args.receipt.open("x", encoding="utf-8", buffering=1) as receipts:
        def record(event, **fields):
            line = json.dumps({"event": event, "unix_ns": time.time_ns(),
                               "monotonic_ns": time.monotonic_ns(), **fields})
            receipts.write(line + "\n")
            print(line, flush=True)

        signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM)
        for signum in signals:
            signal.signal(signum, interrupt)
        shaper = Shaper(args.hccn_tool, args.sudo)
        original = shaper.read()
        record("initial", ports=original)
        try:
            readback = shaper.apply(args.aggregate_gbps)
            record("applied", aggregate_Gbps=args.aggregate_gbps, ports=readback)
            signal.alarm(args.hold_seconds)
            sys.stdin.buffer.read(1)
        finally:
            for signum in signals:
                signal.signal(signum, signal.SIG_IGN)
            signal.alarm(0)
            receipt = shaper.restore(original)
            record("restored", **receipt)
            if not receipt["restored"]:
                raise RuntimeError("Port restoration did not match the initial configuration.")


if __name__ == "__main__":
    main()
