"""Build a reproducible 2x2 scheduler/system experiment with drift controls."""

from __future__ import annotations

import argparse
import math
import random

from pd_kv_transfer_opt.common import output_path, write_json


def build(r0: float, workloads: list[str], repeats: int, seed: int) -> dict:
    if not math.isfinite(r0) or r0 <= 0 or repeats < 1 or not workloads:
        raise ValueError("Positive pilot sustainable rate, repeats, and workloads required")
    if len(set(workloads)) != len(workloads):
        raise ValueError("Workloads must be distinct")
    rng = random.Random(seed)
    cells = [(workload, factor, repeat) for repeat in range(repeats)
             for workload in workloads for factor in (.6, .85, 1.05)]
    rng.shuffle(cells)
    rows = []
    for block, (workload, factor, repeat) in enumerate(cells):
        arms = [("scheduler", True, False), ("system", False, True), ("joint", True, True)]
        rng.shuffle(arms)
        arms = [("base-before", False, False), *arms, ("base-after", False, False)]
        for position, (name, scheduler, system) in enumerate(arms):
            rows.append({"run_id": f"block-{block:03d}-{position}-{name}", "block_id": block,
                         "arm": name, "scheduler": scheduler, "system": system,
                         "workload": workload, "arrival_rate_req_s": r0 * factor,
                         "rate_factor": factor, "repeat": repeat,
                         "workload_seed": seed + repeat,
                         "warmup_s": 60, "measurement_s": 600,
                         "cache_initial_state": "reset_then_identical_preconditioning",
                         "intervention_manifest": "must be filled with exact patch/config hashes"})
    return {"status": "planned_not_executed", "pilot_r0_req_s": r0, "seed": seed,
            "design": "scheduler x system, bracketed native controls, randomized treatment order",
            "notes": ["R0 comes from a pilot, not from previous hardware or these demo tools.",
                      "Hold cache, physical memory, routing inputs, output contract and telemetry equal.",
                      "Baseline drift rejects a block; do not choose its more favorable baseline."],
            "runs": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0", type=float, required=True)
    parser.add_argument("--workload", action="append", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(args.r0, args.workload, args.repeats, args.seed)
    path = output_path(args.output)
    if path.exists():
        raise FileExistsError(path)
    write_json(path, result)


if __name__ == "__main__":
    main()
