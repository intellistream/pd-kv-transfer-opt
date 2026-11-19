"""Screen copy and restore choices on a fixed replay, preserving worker order."""

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from pd_kv_transfer_opt.common import write_json
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union, normalized_proxy_id, read_bundle
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import get_batches, stats


GET = "batch_get_into_multi_buffers"
FORWARD = "p_chunk_rank_forward"


@dataclass
class Read:
    begin: int
    end: int
    size: int
    request_id: str | None = None
    native_size: int | None = None
    copy: dict | None = None
    call_ordinal: int | None = None

    @property
    def returned(self):
        if self.copy is None:
            return self.end
        return self.copy.get("copy_end_ns", self.copy["monotonic_ns"])

    @property
    def copy_begin(self):
        return self.copy.get("copy_begin_ns", self.returned - self.copy["elapsed_ns"])


@dataclass
class Load:
    begin: dict
    end: dict | None = None
    reads: list[Read] = field(default_factory=list)
    native_bytes: dict = field(default_factory=dict)
    verified: bool = False


def load_spans(events):
    """Join completed native calls and their following copy on the worker thread."""
    loads = []
    active = {}
    submitted = {}
    for row in sorted(events, key=lambda event: event["monotonic_ns"]):
        kind = row["type"]
        worker = row["pid"], row["thread_id"]
        if kind == "worker_load_begin":
            if worker in active:
                raise ValueError("Nested synchronous load")
            active[worker] = Load(row)
        elif worker in active and kind == "worker_load_end":
            span = active.pop(worker)
            span.end = row
            loads.append(span)
        elif worker in active and kind == "r18_restore_decision":
            active[worker].native_bytes[row["call_ordinal"]] = row["native_bytes"]
        elif worker in active and row.get("operation") == GET:
            if kind == "store_begin":
                submitted[row["call_id"]] = row["monotonic_ns"]
            elif kind == "store_end":
                if row["negative_returns"]:
                    raise ValueError("Failed native GET in observation")
                span = active[worker]
                ordinal = row.get("call_ordinal")
                span.reads.append(Read(submitted.pop(row["call_id"]), row["monotonic_ns"], row["payload_bytes"],
                                       row.get("request_id"), span.native_bytes.get(ordinal), call_ordinal=ordinal))
        elif worker in active and kind == "same_head_local_copy" and not row["verify"]:
            reads = active[worker].reads
            if row.get("call_ordinal") is None:
                previous = reads[-1]
            else:
                previous, = [read for read in reads if read.call_ordinal == row["call_ordinal"]
                             and read.request_id == row["request_id"]]
            if previous.copy is not None or row["method"] != "packed":
                raise ValueError("Expected one packed copy after a native call")
            previous.copy = row
        elif worker in active and kind == "same_head_verify":
            span = active[worker]
            span.verified = True
            if row.get("extra_get_bytes") is not None:
                checked = span.reads.pop()
                if checked.copy is not None or checked.size != row["extra_get_bytes"]:
                    raise ValueError("Cannot separate validation GET from the observed path")
    if any(span.reads for span in active.values()) or submitted:
        raise ValueError("Incomplete synchronous load observations")
    return loads


def client_ids(decode):
    return {row["proxy_request_id"]: row["id"] for row in decode["proxy"] if row["type"] == "prefill_begin"}


def request_names(load, names):
    if all(read.request_id is not None for read in load.reads):
        ids = [read.request_id for read in load.reads]
    elif len(load.reads) == len(load.begin["request_ids"]):
        ids = load.begin["request_ids"]
    else:
        return ()
    sequence = []
    for read, rid in zip(load.reads, ids, strict=True):
        if read.copy is not None and read.copy.get("request_id"):
            if normalized_proxy_id(read.copy["request_id"]) != normalized_proxy_id(rid):
                return ()
        sequence.append(names.get(normalized_proxy_id(rid)))
    return tuple(sequence) if all(sequence) else ()


def calibrated_direct(prefill, decode):
    """Index an earlier native trace by stable session request and physical destination."""
    names = client_ids(decode)
    ranks = {row["pid"]: row["rank"] for row in prefill["worker_rank"]}
    samples = {}
    repeated = set()
    for span in load_spans(prefill["native"]):
        sequence = request_names(span, names)
        if not sequence or span.begin["pid"] not in ranks:
            continue
        for name, read in zip(sequence, span.reads, strict=True):
            key = name, span.begin["instance"], ranks[span.begin["pid"]], read.size
            if key in samples:
                repeated.add(key)
            samples[key] = read.end - read.begin
    return {key: value for key, value in samples.items() if key not in repeated}


def join_forwards(prefill):
    """Pair scheduler dispatches with local rank forwards and exact worker windows."""
    events = prefill["native"]
    _, batches, errors = get_batches(events)
    forward_queues = defaultdict(deque)
    for row in sorted(events, key=lambda event: event["monotonic_ns"]):
        if row["type"] == FORWARD:
            forward_queues[row["instance"], row["batch"], row["tp_rank"]].append(row)
    joined = []
    by_worker = defaultdict(list)
    for batch in batches:
        if "ready_ns" not in batch:
            continue
        group = {}
        for rank in range(4):
            queue = forward_queues[batch["instance"], batch["batch"], rank]
            while queue and queue[0]["worker_enter_ns"] < batch["dispatch_ns"]:
                queue.popleft()
            if queue and queue[0]["forward_enter_ns"] <= batch["ready_ns"]:
                group[rank] = queue.popleft()
        if len(group) != 4:
            continue
        node = {"batch": batch, "forwards": group, "loads": {}}
        joined.append(node)
        for rank, forward in group.items():
            by_worker[forward["pid"], forward["thread_id"]].append((node, rank, forward))

    unmatched = []
    for span in load_spans(events):
        worker = span.begin["pid"], span.begin["thread_id"]
        options = [(node, rank) for node, rank, fwd in by_worker[worker]
                   if span.begin.get("batch", fwd["batch"]) == fwd["batch"]
                   and fwd["worker_enter_ns"] <= span.begin["monotonic_ns"]
                   <= span.end["monotonic_ns"] <= fwd["forward_enter_ns"]]
        if len(options) != 1 or options[0][1] in options[0][0]["loads"]:
            unmatched.append(span)
        else:
            node, rank = options[0]
            node["loads"][rank] = span
    return batches, joined, errors, unmatched


def replay(node, direct=None, remove_copy=False, staged=None):
    """Replay this batch with fixed releases, observed gaps and GET service."""
    states = {}
    for pair in ((0, 1), (2, 3)):
        a = pair[0]
        loads = node["loads"]
        cursor = [loads[rank].begin["monotonic_ns"] for rank in pair]
        before = cursor.copy()
        for index, reads in enumerate(zip(*(loads[rank].reads for rank in pair), strict=True)):
            gap = [read.begin - previous for read, previous in zip(reads, before, strict=True)]
            if min(gap) < 0:
                raise ValueError("GET begins before the preceding phase completed")
            begin = [t + g for t, g in zip(cursor, gap, strict=True)]
            if staged is not None and staged[0] == index:
                _, completed, scatter_ns = staged
                stop = [max(t, completed[rank]) + scatter_ns for rank, t in zip(pair, begin, strict=True)]
            else:
                stop = [t + read.end - read.begin for t, read in zip(begin, reads, strict=True)]
            if direct is not None and direct[0] == (a, index):
                cursor = [t + duration for t, duration in zip(begin, direct[1], strict=True)]
            elif reads[0].copy is not None and reads[1].copy is not None:
                observed_join = max(read.end for read in reads)
                new_join = max(stop)
                cursor = []
                for read in reads:
                    elapsed = read.returned - read.copy_begin
                    # Host event emission can lag return. Hold the measured residual fixed.
                    release = new_join + read.copy_begin - observed_join
                    if release < new_join:
                        raise ValueError("Packed copy inferred before both GETs completed")
                    cursor.append(release if remove_copy else release + elapsed)
            elif reads[0].copy is None and reads[1].copy is None:
                cursor = stop
            else:
                raise ValueError("Only one rank recorded packed completion")
            before = [read.returned for read in reads]
        for rank, now, previous in zip(pair, cursor, before, strict=True):
            finish = loads[rank].end["monotonic_ns"]
            forward = node["forwards"][rank]["forward_enter_ns"]
            if previous > finish or finish > forward:
                raise ValueError("Load/forward ordering is inconsistent")
            states[rank] = now + finish - previous, now + forward - previous
    return max(state[0] for state in states.values()), max(state[1] for state in states.values())


def compare_batches(nodes, names, samples):
    report = []
    excluded = Counter()
    for node in nodes:
        loads = node["loads"]
        if not any(read.copy is not None for span in loads.values() for read in span.reads):
            continue
        if len(loads) != 4:
            excluded["missing_load_rank"] += 1
            continue
        if any(span.verified for span in loads.values()):
            excluded["includes_validation_get"] += 1
            continue
        orders = {rank: request_names(span, names) for rank, span in loads.items()}
        if not orders[0] or len(set(orders.values())) != 1:
            excluded["unmatched_request_order"] += 1
            continue
        try:
            fixed = replay(node)
            expected = (max(span.end["monotonic_ns"] for span in loads.values()),
                        max(row["forward_enter_ns"] for row in node["forwards"].values()))
            if fixed != expected:
                raise ValueError("Observed baseline failed replay")
            zero_copy = replay(node, remove_copy=True)
        except ValueError as error:
            excluded[str(error)] += 1
            continue
        batch = node["batch"]
        candidates = []
        possible = 0
        for a in (0, 2):
            pair = a, a + 1
            for index, name in enumerate(orders[a]):
                reads = [loads[rank].reads[index] for rank in pair]
                if all(read.copy is not None for read in reads):
                    possible += 1
                    sizes = [read.size + read.copy["payload_bytes"] for read in reads]
                    if any(read.native_size is not None and size != read.native_size
                           for read, size in zip(reads, sizes, strict=True)):
                        raise ValueError("Packed GET and copy bytes disagree with native descriptors")
                    durations = [samples.get((name, batch["instance"], rank, size))
                                 for rank, size in zip(pair, sizes, strict=True)]
                    if None not in durations:
                        added = sum(size - read.size for size, read in zip(sizes, reads, strict=True))
                        candidates.append(((a, index), name, added, durations))
        record = {"instance": batch["instance"], "batch": batch["batch"],
                  "dispatch_ns": batch["dispatch_ns"], "requests_in_batch": len(batch["rows"]),
                  "restore_requests": list(orders[0]), "scheduler_waiting": batch["waiting"],
                  "dispatch_to_result_ms": (batch["ready_ns"] - batch["dispatch_ns"]) / 1e6,
                  "copy_ms_before_later_get": stats((read.returned - read.copy_begin) / 1e6
                                                      for span in loads.values() for read in span.reads[:-1]
                                                      if read.copy is not None),
                  "zero_copy_launch_gain_ms": (fixed[1] - zero_copy[1]) / 1e6,
                  "possible_choices": possible, "matched_choices": len(candidates), "sensitivity": {}}
        for scale in (0.75, 1.0, 1.25):
            choices = [(fixed, 0, None)]
            for (pair, index), name, added, durations in candidates:
                prediction = replay(node, ((pair, index), [round(scale * duration) for duration in durations]))
                choices.append((prediction, added, [name, pair]))
            picked = {}
            for policy, metric in (("communication_finish", 0), ("batch_ready", 1)):
                state, added, action = min(choices, key=lambda choice: (choice[0][metric], choice[1]))
                picked[policy] = {"action": action, "extra_D_bytes": added,
                                  "communication_gain_ms": (fixed[0] - state[0]) / 1e6,
                                  "launch_gain_ms": (fixed[1] - state[1]) / 1e6}
            record["sensitivity"][str(scale)] = picked
        report.append(record)
    return report, dict(excluded)


def summarize(prefill, decode, baseline_p, baseline_d):
    observed, = decode["client_manifest"]
    calibration, = baseline_d["client_manifest"]
    for key in ("workload_sha256", "fixed_replay_sha256"):
        if not observed.get(key) or observed[key] != calibration.get(key):
            raise ValueError(f"Native calibration and packed trace must share {key}")
    if any(row.get("background_traffic") for row in (observed, calibration)):
        raise ValueError("Expected natural replays without injected traffic")
    batches, joined, errors, unmatched = join_forwards(prefill)
    names = client_ids(decode)
    samples = calibrated_direct(baseline_p, baseline_d)
    report, excluded = compare_batches(joined, names, samples)
    completed = {row["id"] for row in decode["client"] if row["type"] == "request_end" and row["status"] == "ok"}
    proxy_ended = {row["id"] for row in decode["proxy"] if row["type"] == "decode_end"}
    names_with_load = set()
    records = []
    get_bytes = 0
    for span in load_spans(prefill["native"]):
        names_with_load.update(request_names(span, names))
        records.extend(read.copy for read in span.reads if read.copy is not None)
        get_bytes += sum(read.size for read in span.reads)
    windows = defaultdict(list)
    for row in records:
        if all(key in row for key in ("batch", "call_ordinal", "copy_begin_ns", "copy_end_ns")):
            windows[row["instance"], row["batch"], row["tp_rank"] // 2, row["call_ordinal"]].append(row)
    complete = [(key, min(row["copy_begin_ns"] for row in rows), max(row["copy_end_ns"] for row in rows))
                for key, rows in windows.items() if len({row["tp_rank"] for row in rows}) == len(rows) == 2]
    overlap = Counter()
    for (left, start, end), (right, other_start, other_end) in combinations(complete, 2):
        if min(end, other_end) > max(start, other_start):
            overlap["same_instance" if left[0] == right[0] else "cross_instance"] += 1
    preflights = [row for row in prefill["native"] if row["type"] == "r18_packed_preflight"]
    decisions = [row for row in prefill["native"] if row["type"] == "r18_restore_decision"]
    policies = {}
    for scale in ("0.75", "1.0", "1.25"):
        items = [row["sensitivity"][scale] for row in report]
        policies[scale] = {policy: {
            "changed_batches": sum(entry[policy]["action"] is not None for entry in items),
            "extra_D_bytes": sum(entry[policy]["extra_D_bytes"] for entry in items),
            "launch_gain_sum_ms": sum(entry[policy]["launch_gain_ms"] for entry in items),
            "communication_gain_sum_ms": sum(entry[policy]["communication_gain_ms"] for entry in items),
        } for policy in ("communication_finish", "batch_ready")}
        policies[scale]["different_actions"] = sum(row["communication_finish"]["action"] != row["batch_ready"]["action"]
                                                     for row in items)
        policies[scale]["additional_launch_gain_ms"] = (policies[scale]["batch_ready"]["launch_gain_sum_ms"]
                                                         - policies[scale]["communication_finish"]["launch_gain_sum_ms"])
    nominal = "1.0"
    only_completed = [row for row in report if all(name in completed for name in row["restore_requests"])]
    eligible_batches = {(row["instance"], row["batch"]) for row in report}
    return {"workload_sha256": observed["workload_sha256"], "fixed_replay_sha256": observed["fixed_replay_sha256"],
            "selected_requests": observed["selected_requests"], "complete_client_rows": len(completed),
            "proxy_decode_ends": len(proxy_ended),
            "scheduler_batches": len(batches), "four_rank_forward_batches": len(joined),
            "scheduler_pairing_errors": errors, "unmatched_worker_loads": len(unmatched),
            "requests_with_restore": len(names_with_load), "native_calibration_samples": len(samples),
            "observed_get_payload_bytes_excluding_verification": get_bytes,
            "native_descriptor_bytes_for_same_restores": sum(row["native_bytes"] for row in decisions),
            "copy_rank_events": len(records), "packed_pair_operations": len(complete),
            "synchronous_copy_overlaps": dict(overlap),
            "copy_instance_union_ms": {instance: interval_union((start, end) for key, start, end in complete
                                                                 if key[0] == instance) / 1e6
                                       for instance in sorted({key[0] for key, _, _ in complete})},
            "copy_wall_ms": stats(copy["elapsed_ns"] / 1e6 for copy in records),
            "eligible_copy_wall_ms": stats(copy["elapsed_ns"] / 1e6 for copy in records
                                           if (copy["instance"], copy.get("batch")) in eligible_batches),
            "pair_wait_ms": stats(copy["pair_wait_ns"] / 1e6 for copy in records),
            "restore_rank_modes": dict(Counter(row["mode"] for row in decisions)),
            "preflight_plan_ms": stats(row["plan_ns"] / 1e6 for row in preflights),
            "preflight_pair_wait_ms": stats(row["wait_ns"] / 1e6 for row in preflights),
            "eligible_packed_batches": len(report), "excluded_packed_batches": excluded,
            "multi_restore_packed_batches": sum(len(row["restore_requests"]) > 1 for row in report),
            "positive_zero_copy_launch_gain_batches": sum(row["zero_copy_launch_gain_ms"] > 0 for row in report),
            "zero_copy_launch_gain_ms": stats(row["zero_copy_launch_gain_ms"] for row in report),
            "possible_choices": sum(row["possible_choices"] for row in report),
            "matched_choices": sum(row["matched_choices"] for row in report),
            "all_choices_matched_batches": sum(row["possible_choices"] == row["matched_choices"] for row in report),
            "only_completed_client_rows_nominal": {"batches": len(only_completed), "additional_launch_gain_ms": sum(
                row["sensitivity"][nominal]["batch_ready"]["launch_gain_ms"]
                - row["sensitivity"][nominal]["communication_finish"]["launch_gain_ms"] for row in only_completed)},
            "policy_comparison": policies, "batches": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prefill", "decode", "native_prefill", "native_decode", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    result = summarize(*(read_bundle(path) for path in (args.prefill, args.decode, args.native_prefill, args.native_decode)))
    write_json(args.output, result)


if __name__ == "__main__":
    main()
