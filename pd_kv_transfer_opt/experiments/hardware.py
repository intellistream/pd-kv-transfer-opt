"""Read and set the installed 910B D-side hardware shaper."""

import re
import subprocess
import time


PORTS = range(8)
LIMITS = re.compile(r"bw_limit\[(\d+) Mbps\],\s*bw_max\[(\d+) Mbps\],\s*bw_cur\[(\d+) Mbps\]")
COUNTERS = ("mac_tx_total_oct_num", "mac_rx_total_oct_num", "mac_rx_pfc_pkt_num",
            "mac_tx_pfc_pkt_num", "roce_tx_cnp_pkt_num", "roce_new_pkt_rty_num")


class Shaper:
    def __init__(self, path, sudo=False):
        self.command = (["sudo", "-n"] if sudo else []) + [str(path)]

    def call(self, port, *flags):
        answer = subprocess.run([*self.command, "-i", str(port), *flags], text=True, capture_output=True, timeout=8)
        if answer.returncode:
            raise RuntimeError(f"Port {port} {flags} returned {answer.returncode}: {answer.stderr.strip()}")
        return answer.stdout.strip()

    def read(self):
        values = {}
        for port in PORTS:
            raw = self.call(port, "-shaping", "-g")
            match = LIMITS.fullmatch(raw)
            if not match:
                raise ValueError(f"Unrecognized shaping reply on port {port}: {raw}")
            values[port] = dict(zip(("configured_Mbps", "maximum_Mbps", "current_Mbps"), map(int, match.groups())))
        return values

    def set(self, port, mbps):
        if not 10000 <= mbps <= 200000:
            raise ValueError(f"Invalid port cap: {mbps}")
        self.call(port, "-shaping", "-s", "bw_limit", str(mbps))

    def apply(self, gbps):
        per_port, remainder = divmod(gbps * 1000, len(PORTS))
        if remainder:
            raise ValueError("Aggregate cap must divide equally across eight ports.")
        for port in PORTS:
            self.set(port, per_port)
        actual = self.read()
        if any(item["configured_Mbps"] != per_port or item["current_Mbps"] != per_port for item in actual.values()):
            raise RuntimeError(f"Shaper readback differs from requested cap: {actual}")
        return actual

    def restore(self, before):
        errors = {}
        for port, value in before.items():
            for _ in range(3):
                try:
                    self.set(port, value["configured_Mbps"])
                    errors.pop(port, None)
                    break
                except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                    errors[port] = str(exc)
                    time.sleep(.5)
        after = self.read()
        return {"requested": before, "verified": after, "errors": errors, "restored": before == after}

    def sample(self):
        records = []
        for port in PORTS:
            start = time.monotonic_ns()
            text = self.call(port, "-stat", "-g")
            end = time.monotonic_ns()
            counters = {k: int(v) for k, v in re.findall(r"^\s*([a-z][a-z0-9_]*):\s*(\d+)\s*$", text, re.M) if k in COUNTERS}
            if "mac_tx_total_oct_num" not in counters:
                raise RuntimeError(f"Missing TX counter on port {port}")
            records.append({"port": port, "monotonic_ns": (start + end) // 2, "counters": counters})
        return {"unix_ns": time.time_ns(), "ports": records}
