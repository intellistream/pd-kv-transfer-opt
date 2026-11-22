"""Materialize a replay selection from the pinned Terminal-Bench request file."""

from array import array
from hashlib import sha256
import json
import sys

from pd_kv_transfer_opt.experiments.loadgen import validate_workload


def prompt_hash(ids):
    tokens = array("I", ids)
    if tokens.itemsize != 4:
        raise ValueError("Expected 32-bit token IDs")
    if sys.byteorder != "little":
        tokens.byteswap()
    return sha256(tokens.tobytes()).hexdigest()


def load_cohort(source, plan, salt, decode=json.loads):
    """Keep original request bodies, predecessor links and output budgets."""
    if plan["ignore_eos"] or plan["temperature"] != 0 or plan["route_policy"] != "upstream native without route override":
        raise ValueError("The selection must keep the native replay policy")
    specs = {row["id"]: row for row in plan["request_sources"]}
    digest = sha256()
    selected = {}
    with source.open("rb") as stream:
        for line in stream:
            digest.update(line)
            request = decode(line)
            key = request["id"]
            if key not in specs:
                continue
            spec = specs[key]
            fixed = ("session_id", "predecessor", "tool_gap_s", "max_tokens", "prompt_tokens")
            if (key in selected or any(request[k] != spec[k] for k in fixed)
                    or request["offered_s"] != spec["original_offered_s"]
                    or len(request["prompt"]) != spec["prompt_tokens"]
                    or prompt_hash(request["prompt"]) != spec["prompt_sha256"]):
                raise ValueError(f"The parent workload no longer matches {key}")
            selected[key] = {**request, "offered_s": spec["offered_s"], "cache_salt": salt}
    parent_sha256 = plan.get("parent_requests_sha256") or plan["parent_r42_sha256"]
    if digest.hexdigest() != parent_sha256 or len(selected) != plan["requests"] or specs.keys() != selected.keys():
        raise ValueError("Incomplete selection or mismatched parent workload")
    result = [selected[key] for key in specs]
    validate_workload(result)
    roots = [request for request in result if request["predecessor"] is None]
    if len(roots) != plan["sessions"] or max(row["offered_s"] for row in roots) != plan["last_session_start_s"]:
        raise ValueError("Incorrect number of roots or last root arrival")
    return result
