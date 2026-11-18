"""Event-driven max-min fluid transport model; not a packet/RoCE simulator.

Flows use explicit directed resources. Resources shared by opposite directions
must be named explicitly (e.g. a common memory controller). Static arrival replay
is conditional on those arrivals and is not a serving-policy counterfactual.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import math

from pd_kv_transfer_opt.common import read_json, write_json, output_path


@dataclass(frozen=True)
class Flow:
    id: str
    arrival_s: float
    bytes: int
    path: tuple[str, ...]
    startup_s: float = 0.0
    admission_group: str = "default"

    @property
    def ready_s(self) -> float:
        return self.arrival_s + self.startup_s


def max_min_rates(flows: list[Flow], capacities: dict[str, float]) -> dict[str, float]:
    """Progressive filling across every resource on each active flow's path."""
    rates = {flow.id: 0.0 for flow in flows}
    residual = dict(capacities)
    pending = list(flows)
    while pending:
        counts = Counter(resource for flow in pending for resource in flow.path)
        increments = {key: residual[key] / count for key, count in counts.items()}
        increment = min(increments.values())
        saturated = {key for key, value in increments.items()
                     if math.isclose(value, increment, rel_tol=1e-12, abs_tol=1e-15)}
        for flow in pending:
            rates[flow.id] += increment
        for resource, count in counts.items():
            residual[resource] = max(0.0, residual[resource] - increment * count)
        pending = [flow for flow in pending if not saturated.intersection(flow.path)]
    return rates


def simulate(flows: list[Flow], capacities: dict[str, float],
             admission_limits: dict[str, int] | None = None) -> dict:
    if not capacities or any(not math.isfinite(value) or value <= 0 for value in capacities.values()):
        raise ValueError("Every resource needs a finite positive byte/s capacity")
    if len({flow.id for flow in flows}) != len(flows):
        raise ValueError("Flow IDs must be unique")
    for flow in flows:
        if (not flow.id or not isinstance(flow.bytes, int) or isinstance(flow.bytes, bool)
                or flow.bytes < 0 or not flow.path or len(set(flow.path)) != len(flow.path)
                or not set(flow.path).issubset(capacities)
                or not all(math.isfinite(v) and v >= 0 for v in (flow.arrival_s, flow.startup_s))):
            raise ValueError(f"Invalid flow: {flow.id}")
    limits = admission_limits or {}
    if any(not isinstance(value, int) or value < 1 for value in limits.values()):
        raise ValueError("Admission limits must be positive integers")
    arrivals = sorted(enumerate(flows), key=lambda pair: (pair[1].ready_s, pair[0]))
    waiting: list[Flow] = []
    active: dict[str, Flow] = {}
    remaining = {flow.id: float(flow.bytes) for flow in flows}
    records: dict[str, dict] = {}
    epochs = []
    cursor = 0
    now = arrivals[0][1].ready_s if arrivals else 0.0
    while cursor < len(arrivals) or waiting or active:
        while cursor < len(arrivals) and arrivals[cursor][1].ready_s <= now:
            flow = arrivals[cursor][1]
            waiting.append(flow)
            cursor += 1
        counts = Counter(flow.admission_group for flow in active.values())
        for flow in waiting[:]:
            group = flow.admission_group
            if counts[group] >= limits.get(group, math.inf):
                continue
            waiting.remove(flow)
            records[flow.id] = {"id": flow.id, "bytes": flow.bytes,
                                "arrival_s": flow.arrival_s, "ready_s": flow.ready_s,
                                "service_start_s": now, "queue_wait_s": now - flow.ready_s}
            if flow.bytes == 0:
                records[flow.id].update(complete_s=now, service_s=0.0)
            else:
                active[flow.id] = flow
                counts[group] += 1
        if not active:
            if cursor < len(arrivals):
                now = arrivals[cursor][1].ready_s
                continue
            if waiting:
                raise AssertionError("Admission deadlock")
            break
        rates = max_min_rates(list(active.values()), capacities)
        completion_delta = min(remaining[key] / rate for key, rate in rates.items())
        next_arrival = arrivals[cursor][1].ready_s if cursor < len(arrivals) else math.inf
        end = min(now + completion_delta, next_arrival)
        dt = end - now
        if dt < 0 or not math.isfinite(end):
            raise AssertionError("Invalid simulated time")
        epochs.append({"start_s": now, "end_s": end, "rates_bytes_s": rates})
        for key, rate in rates.items():
            remaining[key] = max(0.0, remaining[key] - rate * dt)
        now = end
        for key in list(active):
            if remaining[key] <= max(1e-8, active[key].bytes * 1e-12):
                record = records[key]
                record.update(complete_s=now, service_s=now - record["service_start_s"])
                del active[key]
    # Conservation and capacity audits are part of the model, not post-hoc fitting.
    integrated = Counter()
    by_id = {flow.id: flow for flow in flows}
    for epoch in epochs:
        usage = Counter()
        for key, rate in epoch["rates_bytes_s"].items():
            integrated[key] += rate * (epoch["end_s"] - epoch["start_s"])
            for resource in by_id[key].path:
                usage[resource] += rate
        if any(value > capacities[key] * (1 + 1e-10) for key, value in usage.items()):
            raise AssertionError("Resource capacity exceeded")
    if any(not math.isclose(integrated[flow.id], flow.bytes, rel_tol=1e-9, abs_tol=1e-6)
           for flow in flows):
        raise AssertionError("Bytes were lost or created")
    return {"model": "directed_resource_max_min_fluid/v1", "scope": "conditional_transport_replay",
            "flows": [records[flow.id] for flow in flows], "rate_epochs": epochs,
            "audit": {"byte_conservation": True, "resource_capacity": True}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = read_json(args.input)
    flows = [Flow(**{**row, "path": tuple(row["path"])}) for row in data["flows"]]
    result = simulate(flows, data["capacities_bytes_s"], data.get("admission_limits"))
    result["calibration"] = data.get("calibration", {"status": "unknown"})
    path = output_path(args.output)
    if path.exists():
        raise FileExistsError(path)
    write_json(path, result)


if __name__ == "__main__":
    main()
