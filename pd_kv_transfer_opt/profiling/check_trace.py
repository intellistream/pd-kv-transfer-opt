"""Audit host span pairing without subtracting clocks from different hosts."""

from __future__ import annotations

import argparse
from collections import Counter

from pd_kv_transfer_opt.common import read_jsonl, write_json, output_path


def audit(rows: list[dict]) -> dict:
    errors = []
    headers = [row for row in rows if row.get("type") == "trace_header"]
    footers = [row for row in rows if row.get("type") == "trace_footer"]
    if len(headers) != 1 or len(footers) != 1:
        errors.append("Require one header and one footer per writer file")
    if any(row.get("dropped_events") != 0 or row.get("writer_failure") for row in footers):
        errors.append("Dropped events or writer failures")
    begins, ends = {}, {}
    names = Counter()
    for row in rows:
        if row.get("type") != "event":
            continue
        names[row["name"]] += 1
        if row.get("phase") not in ("begin", "end"):
            continue
        key = (row["host"], row["pid"], row["span_id"])
        target = begins if row["phase"] == "begin" else ends
        if key in target:
            errors.append(f"Duplicate span phase: {key}")
        target[key] = row
    if begins.keys() != ends.keys():
        errors.append("Unpaired span begins/ends")
    durations = {}
    for key in begins.keys() & ends.keys():
        begin, end = begins[key], ends[key]
        if begin["name"] != end["name"] or begin["monotonic_ns"] > end["monotonic_ns"]:
            errors.append(f"Invalid span: {key}")
        durations.setdefault(begin["name"], []).append((end["monotonic_ns"] - begin["monotonic_ns"]) / 1e6)
    return {"valid_host_span_accounting": not errors, "errors": errors,
            "event_counts": dict(names), "span_durations_ms": durations,
            "scope": "host spans only; no device execution or cross-host clock validation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(read_jsonl(args.trace))
    path = output_path(args.output)
    if path.exists():
        raise FileExistsError(path)
    write_json(path, result)
    if result["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
