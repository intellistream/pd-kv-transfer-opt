"""Extract portable measurement rows from local event and profiler exports."""

import argparse
import gzip
import io
import json
from pathlib import Path
import re


KEYS = {
    "scope": "run_id role",
    "runtime_state": "run_id role stage execution_mode ready_seconds elapsed_seconds pool_load_async pool_residency",
    "worker_rank": "instance rank pid",
    "npu": "unix_ns device aicore_percent hbm_used_mb hbm_total_mb",
    "fabric": "unix_ns device counters",
    "p_port_snapshot": "case edge unix_ns ports",
    "proxy": "type monotonic_ns unix_ns id proxy_request_id status input_tokens max_tokens chunks route",
    "client_manifest": "case run_id workload_sha256 aggregate_D_egress_cap_Gbps selected_requests window_start "
                       "prefix_ids prompt_lengths offered_rate output_tokens background_traffic profiler_enabled seconds "
                       "fixed_replay_sha256 ignore_eos arm_mode phase",
    "client": "case type monotonic_s anchor id status http_status prompt_tokens completion_tokens offered_s ready_s "
              "dispatch_s first_s end_s all_content_chunks_have_token_ids streamed_token_ids finish_reason done_signal",
    "client_summary": "case schema run_id counts quality window finite_cohort fixed_output_deviations",
    "cap_port_sample": "case unix_ns ports",
    "capture": "capture host_role pid kernel_csv_sha256 kernel_csv_bytes trace_json_bytes",
    "kernel": "capture start_ns duration_ns name core",
    "CPU_scope": "capture name category tid start_ns duration_ns",
    "API_stat": "capture Device_id Level Time(us) Count Avg(us) Min(us) Max(us) Variance",
}
EVENT_KEYS = {
    "worker_load_begin": "request_ids load_async batch worker_enter_ns tp_rank",
    "worker_load_end": "request_ids load_async elapsed_ns batch worker_enter_ns tp_rank",
    "store_begin": "call_id operation keys payload_bytes batch tp_rank request_id call_ordinal",
    "store_end": "call_id operation keys payload_bytes negative_returns elapsed_ns batch tp_rank request_id call_ordinal",
    "mooncake_backend_get": "keys elapsed_ns",
    "pool_match_tokens": "request_id local_tokens external_tokens load_async elapsed_ns",
    "native_transfer_enqueue": "request_id remote_request_id shard_idx",
    "native_transfer_submit": "remote_request_id payload_bytes descriptor_count descriptor_size_histogram",
    "native_handle_begin": "remote_request_id",
    "native_handle_end": "remote_request_id",
    "p_chunk_arrival": "request_id arrival_ns prompt_tokens",
    "p_chunk_dispatch": "batch rows begin_ns dispatch_ns running waiting budget total preempted",
    "p_chunk_ready": "batch rows ready_ns",
    "p_chunk_graph": "batch runtime_mode padded_tokens aclgraph_replays",
    "p_chunk_device": "batch worker_enter_ns forward_enter_ns forward_cpu_return_ns forward_device_ms",
    "p_chunk_rank_forward": "batch tp_rank worker_enter_ns forward_enter_ns forward_cpu_return_ns",
    "same_head_pair_ready": "tp_rank peer_rank",
    "same_head_local_copy": "request_ids tp_rank payload_bytes elapsed_ns device_elapsed_ms pair_wait_ns operations verify method "
                            "batch request_id call_ordinal copy_begin_ns copy_end_ns",
    "same_head_verify": "request_ids tp_rank sample_bytes samples method batch request_id call_ordinal extra_get_bytes run_mode",
    "same_head_copy_layout": "request_ids tp_rank raw_segments layout_boundaries local_only_breaks peer_only_breaks both_breaks contiguous_both",
    "r18_restore_decision": "batch tp_rank request_id call_ordinal mode native_bytes keys attention_segments",
    "r18_packed_preflight": "batch tp_rank requests nonempty_gets same_request_order plan_ns wait_ns",
    "r20_pipeline_batch": "batch tp_rank jobs eligible",
    "r20_copy_wait": "batch worker_enter_ns tp_rank request_id call_ordinal next_call_ordinal probes device_elapsed_ms record_start_ns "
                     "enqueue_end_ns wait_start_ns wait_end_ns",
}
HEADER = "type pid thread_id unix_ns monotonic_ns instance"
FRAME = re.compile(r"^\s*([\w.<>]+) \(([\w./-]+\.py):(\d+)\)\s*$")


def select(row, fields):
    return {key: row[key] for key in fields.split() if key in row}


def frames(text):
    lines = []
    for line in text.splitlines():
        match = FRAME.fullmatch(line)
        if match:
            method, path, number = match.groups()
            relative = path if path.startswith(("vllm/", "vllm_ascend/")) else Path(path).name
            lines.append(f"{method} ({relative}:{number})")
    return "\n".join(lines)


def control(row):
    name = row["name"]
    raw = row["data"]
    if name == "progress.json":
        if raw["stage"] != "complete" or not raw["restored"]:
            raise ValueError("Publish completed runs after the shaper restore readback.")
        value = select(raw, "stage current completed_cases restored recovered_profile_stop")
    elif name.endswith("-trace.json"):
        value = select(raw, "case stage D_worker_start D_worker_stop")
        for action in ("start", "stop"):
            value[action] = [select(item, "instance status start_unix_ns end_unix_ns") for item in raw.get(action, [])]
            for src, dst in zip(raw.get(action, []), value[action]):
                if "status" not in dst and "HTTPError 500" in src.get("error", ""):
                    dst["status"] = 500
    elif name == "shaping-initial.json":
        value = raw
    elif name == "shaping-restore.json":
        if not raw["restored"]:
            raise ValueError("Missing shaper restoration.")
        value = select(raw, "requested verified restored")
    elif name.endswith("-shaping.json"):
        value = select(raw, "requested_aggregate_Gbps readback")
    elif re.fullmatch(r"[ab]-b\d+-cap(?:100|200)(?:-prep)?-(?:pool-reset|progress|restore|master-(?:before|after))\.json", name):
        value = select(raw, "case phase stage initial_keys remaining_keys attempts success restored requested verified "
                            "master_key_count master_evicted_key_count master_put_end_failures_total "
                            "master_batch_put_end_failures_total master_put_start_alloc_failures_total")
    else:
        return None
    return {"name": name, "data": value}


def sanitize(kind, data):
    if kind == "native":
        extras = EVENT_KEYS.get(data["type"])
        return select(data, HEADER + " " + extras) if extras is not None else None
    if kind == "cpu_stack":
        snapshots = []
        for item in data["snapshots"]:
            if item["exit_code"] != 0:
                continue
            snapshot = select(item, "process pid begin_unix_ns end_unix_ns exit_code")
            snapshot["stack"] = frames(item["stack"])
            snapshots.append(snapshot)
        return {"case": data["case"], "trigger": sanitize("native", data["trigger"]), "snapshots": snapshots}
    if kind == "clock":
        lines = [line for line in data.get("text", "").splitlines()
                 if line.startswith(("System time", "Root delay", "Root dispersion", "Last offset"))]
        return {"returncode": data.get("returncode"), "text": "\n".join(lines)}
    if kind == "cap_control":
        return control(data)
    if kind in ("P_store_errors", "store_error", "master_event"):
        raise ValueError("Review error-bearing traces before publishing measurement rows.")
    if kind not in KEYS:
        return None
    result = select(data, KEYS[kind])
    if kind == "scope" and data["run_id"].endswith(("-r19-natural-ab", "-r20-packed-overlap")):
        result["active"] = select(data["active"], "p_local_prefix_caching d_local_prefix_caching p_batched_tokens "
                                  "p_graph_capture_sizes max_model_len p_graph_strategy d_graph_strategy "
                                  "pool_capacity_GiB experimental_batch_get_concurrency rdma_pressure")
    if kind == "client_manifest":
        result["request_sources"] = [select(r, "id source_id prompt_tokens offered_s max_tokens")
                                     for r in data["request_sources"]]
    if kind == "client" and data.get("type") == "request_end" and data["status"] != "ok":
        raise ValueError("Request failed. Keep the run outside the published baseline.")
    if kind == "API_stat":
        result["API Name"] = data["API Name"]
    return result


def publish(source, output):
    with gzip.open(source, "rt", encoding="utf-8") as inp, output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as out:
                for line in inp:
                    event = json.loads(line)
                    clean = sanitize(event["kind"], event["data"])
                    if clean is not None:
                        out.write(json.dumps({"kind": event["kind"], "data": clean}, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    publish(args.source, args.output)


if __name__ == "__main__":
    main()
