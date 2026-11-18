"""Paired run/block bootstrap. Input CSV: block_id,control,treatment."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics

from pd_kv_transfer_opt.common import output_path, percentile, write_json


def compare(rows: list[dict], seed: int = 20261116, samples: int = 10000) -> dict:
    if len(rows) < 3 or len({row["block_id"] for row in rows}) != len(rows):
        raise ValueError("At least three distinct independent paired blocks required")
    pairs = [(float(row["control"]), float(row["treatment"])) for row in rows]
    if any(not math.isfinite(a) or not math.isfinite(b) or a <= 0 or b < 0 for a, b in pairs):
        raise ValueError("Controls must be positive; treatments nonnegative; all values finite")
    if samples < 100:
        raise ValueError("At least 100 bootstrap samples required")
    rng = random.Random(seed)
    def relative(data):
        return (statistics.fmean(b for _, b in data) / statistics.fmean(a for a, _ in data) - 1) * 100
    effects = [relative([rng.choice(pairs) for _ in pairs]) for _ in range(samples)]
    return {"paired_blocks": len(pairs), "treatment_relative_change_percent": relative(pairs),
            "paired_bootstrap_95_percent_interval": [percentile(effects, .025), percentile(effects, .975)],
            "method": "ratio of paired block means; resample entire independent blocks",
            "seed": seed, "samples": samples,
            "qualification": "CI is descriptive with few blocks; drift, SLO, correctness and bottleneck attribution need separate checks"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.csv, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    path = output_path(args.output)
    if path.exists():
        raise FileExistsError(path)
    write_json(path, compare(rows))


if __name__ == "__main__":
    main()
