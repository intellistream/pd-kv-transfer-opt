"""Small, dependency-free helpers shared by the experiment tools."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time


def output_path(value: str | Path) -> Path:
    return Path(value).resolve()


def new_directory(value: str | Path) -> Path:
    path = output_path(value)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value: object) -> None:
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clock_anchor() -> dict:
    before = time.monotonic_ns()
    wall = time.time_ns()
    after = time.monotonic_ns()
    return {"monotonic_ns": (before + after) // 2, "unix_ns": wall,
            "sampling_uncertainty_ns": (after - before + 1) // 2}


def percentile(values: list[float], probability: float) -> float | None:
    if not 0 < probability <= 1:
        raise ValueError("Probability must lie in (0, 1]")
    return sorted(values)[math.ceil(len(values) * probability) - 1] if values else None


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Expected a finite positive number")
    return number
