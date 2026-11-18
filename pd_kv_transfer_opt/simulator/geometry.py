"""Compute logical hybrid-state sizes; actual connector wire bytes need measurement."""

from __future__ import annotations

import argparse
import json

from pd_kv_transfer_opt.common import read_json


def geometry(config: dict, tp: int, length: int) -> dict:
    model = config.get("text_config", config)
    if tp < 1 or length < 1:
        raise ValueError("Positive TP and token length required")
    types = model["layer_types"]
    if len(types) != model["num_hidden_layers"]:
        raise ValueError("Layer list must match the model")
    full = types.count("full_attention")
    linear = types.count("linear_attention")
    heads = model["num_key_value_heads"]
    if max(heads, tp) % min(heads, tp):
        raise ValueError("Cannot assume ordinary equal KV sharding/replication for this TP")
    # BF16 K/V; replication estimate assumes each rank stores at least one KV head.
    logical = 2 * full * heads * model["head_dim"] * 2
    stored = 2 * full * max(heads, tp) * model["head_dim"] * 2
    recurrent = (linear * model["linear_num_value_heads"] * model["linear_key_head_dim"]
                 * model["linear_value_head_dim"] * 4)
    return {"assumptions": "BF16 full-attention K/V, FP32 recurrent state, ordinary equal TP KV layout",
            "full_attention_layers": full, "linear_attention_layers": linear,
            "logical_attention_bytes_per_token": logical,
            "estimated_aggregate_stored_attention_bytes_per_token": stored,
            "logical_attention_bytes_at_length": logical * length,
            "estimated_aggregate_stored_attention_bytes_at_length": stored * length,
            "logical_recurrent_bytes_per_snapshot": recurrent,
            "wire_bytes": None,
            "unmodeled": ["convolution state", "allocator/page/group padding", "multiple hybrid state checkpoints",
                          "connector selection of replicated ranks", "runtime dtype/layout changes"],
            "note": "Stored bytes are not wire bytes. Measure per-group/rank allocations and actual RDMA byte counters."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--length", type=int, default=32768)
    args = parser.parse_args()
    print(json.dumps(geometry(read_json(args.config), args.tp, args.length), indent=2))
