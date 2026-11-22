"""Normalize the pinned FreeInference release into a whole-session replay ledger."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256 as digest
import json
from pathlib import Path

from pd_kv_transfer_opt.common import read_json, sha256
from pd_kv_transfer_opt.simulator.agent_trace_gap import (
    ATTENTION_BLOCK_TOKENS,
    TRACE_BLOCK_TOKENS,
    fits,
    read_bytes,
    same_source,
    shared_native_blocks,
)


@dataclass(frozen=True, slots=True)
class Turn:
    tool_gap_ms: int
    prompt: int
    output: int
    pages: tuple[int, ...]
    adjacent_pages: int
    finish: str


@dataclass(frozen=True, slots=True)
class Session:
    name: str
    source_start_ms: int
    source_duration_ms: int
    source_read_bytes: int
    turns: tuple[Turn, ...]

    @property
    def output(self):
        return sum(turn.output for turn in self.turns)


def normalize_session(row, keys):
    requests = row["requests"]
    if (not requests or not all(map(fits, requests))
            or any(left["end_ms"] > right["start_ms"] for left, right in zip(requests, requests[1:]))):
        return None

    stride = ATTENTION_BLOCK_TOKENS // TRACE_BLOCK_TOKENS
    normalized = []
    before = None
    for request in requests:
        tokens = request["provider_tokens"]
        block_ids = request["block_ids"] or []
        # The last 16-token key of each complete native page includes all
        # preceding tokens in its source namespace.
        namespace = (row["week"], request["provider"], request["model"])
        limit = min(len(block_ids), tokens["prompt"] // TRACE_BLOCK_TOKENS)
        pages = [keys.setdefault((*namespace, block), len(keys) + 1)
                 for block in block_ids[stride - 1:limit:stride]]
        shared = 0
        gap = 0
        if before is not None:
            gap = request["start_ms"] - before["end_ms"]
            if same_source(before, request):
                shared = shared_native_blocks(before["block_ids"] or [], block_ids,
                                              min(before["provider_tokens"]["prompt"], tokens["prompt"]))
                actual = next((i for i, (a, b) in enumerate(zip(normalized[-1][3], pages)) if a != b),
                              min(len(normalized[-1][3]), len(pages)))
                if shared != actual:
                    raise ValueError("A native page key disagrees with the adjacent source prefix")
        normalized.append([gap, max(tokens["prompt"], len(block_ids) * TRACE_BLOCK_TOKENS),
                           tokens["completion"], pages, shared, request["finish_reason"]])
        before = request

    source_id = json.dumps([row["week"], row["session_id"]], separators=(",", ":"))
    name = digest(source_id.encode()).hexdigest()[:20]
    start = requests[0]["start_ms"]
    return [name, start, requests[-1]["end_ms"] - start, sum(read_bytes(r[4]) for r in normalized), normalized]


def build(directory, screen, output):
    from compression import zstd

    source = read_json(screen)["FreeInference"]
    gate = source["steady_state_gate"]
    cutoff = gate["minimum_remote_MB_per_output_at_decode_peak"] * 1e6
    counts = Counter()
    keys, seen = {}, set()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    with zstd.open(temporary, "wt", encoding="utf-8") as dest:
        header = {"revision": "bb26cf72c9c20f2b05f765d88548156a74d3c253",
                  "sources": {name: data["sha256"] for name, data in source["sources"].items()},
                  "high_read_bytes_per_output": cutoff}
        print(json.dumps(header, sort_keys=True), file=dest)
        for name, expected in source["sources"].items():
            path = directory / name
            if sha256(path) != expected["sha256"]:
                raise ValueError(f"Unexpected source digest for {name}")
            with zstd.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    key = (row["week"], row["session_id"])
                    if key in seen:
                        raise ValueError("Repeated FreeInference session")
                    seen.add(key)
                    session = normalize_session(row, keys)
                    if session:
                        counts["complete"] += 1
                        counts["high"] += session[3] >= cutoff * sum(turn[2] for turn in session[4])
                        print(json.dumps(session, separators=(",", ":")), file=dest)
            print(f"{name} {len(seen)} source sessions {dict(counts)}", flush=True)
    if (counts["complete"] != source["complete_compatible_sessions"]
            or counts["high"] != gate["high_read_density_complete_sessions"]):
        raise ValueError(f"Normalized workload did not reproduce the original screen: {counts}")
    temporary.replace(output)


def load(path):
    from compression import zstd

    with zstd.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        sessions = []
        for line in stream:
            name, start, duration, reads, turns = json.loads(line)
            sessions.append(Session(name, start, duration, reads, tuple(
                Turn(gap, prompt, output, tuple(pages), shared, finish)
                for gap, prompt, output, pages, shared, finish in turns)))
    return header, sessions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "screen", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    build(args.data, args.screen, args.output)


if __name__ == "__main__":
    main()
