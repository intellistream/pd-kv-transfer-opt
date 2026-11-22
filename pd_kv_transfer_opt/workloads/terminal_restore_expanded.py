"""Find more original, observed short-output windows without adding another restore wave."""

from collections import defaultdict, deque
from math import ceil
from statistics import median

from pd_kv_transfer_opt.common import read_jsonl, sha256
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS, ATTENTION_BYTES, STATE_BYTES
from pd_kv_transfer_opt.workloads.terminal_restore_checkpoints import plan, root_order
from pd_kv_transfer_opt.workloads.terminal_restore_dense import cold_prefix_bound
from pd_kv_transfer_opt.workloads.terminal_restore_order import prefill_weights, routes
from pd_kv_transfer_opt.workloads.terminal_restore_stable import guard_cold_arrivals, read_reference


FAMILIES = ("caffe-cifar-10", "financial-document-processor", "make-doom-for-mips", "make-mips-interpreter")


def select(ledger_path, template, observed, source_output, fit, *, max_observed_tokens=750,
           task_ranges=None, allow_unseen=False, minimum_common_prefix=65_536):
    if observed["ledger_sha256"] != sha256(ledger_path) or not (
        observed["source_revision"] == source_output["source_revision"] == template["source_revision"]
    ):
        raise ValueError("The source, observed output receipts and length ledger must agree")
    ledger = {row["id"]: row for row in read_jsonl(ledger_path)}
    counts = source_output["source_completion_tokens"]
    outcomes = observed["outcomes"]
    if task_ranges is None:
        task_ranges = dict.fromkeys(FAMILIES, (0, float("inf")))

    def eligible(row):
        key = row["id"]
        history = outcomes.get(key, ())
        first, last = task_ranges[row["task"]]
        adequate_observations = len(history) >= 4 or (allow_unseen and not history)
        return (first <= row["step_id"] <= last and not row["truncated"]
                and type(counts[key]) is int and 0 < counts[key] <= 8192 and adequate_observations
                and row["prompt_tokens"] + 8192 <= template["max_context_tokens"]
                and all(reason == "stop" and 0 < n <= max_observed_tokens for reason, n in history))

    pairs, inputs = [], []
    for family in task_ranges:
        overlap, occupied = set(), set()
        rows = sorted((row for row in ledger.values() if row["task"] == family), key=lambda row: -row["step_id"])
        for child in rows:
            root = ledger.get(child["predecessor"])
            if root is None or root["task"] != family or not eligible(root) or not eligible(child):
                continue
            lcp = child["source_prompt_lcp_to_previous"]
            blocks = lcp // ATTENTION_BLOCK_TOKENS
            root_key, key = root["id"], child["id"]
            if lcp < max(minimum_common_prefix, .9 * root["prompt_tokens"]) or blocks in overlap or {root_key, key} & occupied:
                continue
            overlap.add(blocks)
            occupied.update((root_key, key))
            pairs.append({"task": family, "root": root_key, "successor": key, "checkpoint_blocks": blocks,
                          "potential_GET_bytes": blocks * ATTENTION_BYTES + STATE_BYTES,
                          "historical_completions": {r: len(outcomes.get(r, ())) for r in (root_key, key)},
                          "estimated_output_tokens": {r: median(n for _, n in outcomes[r]) if outcomes.get(r) else counts[r]
                                                      for r in (root_key, key)}})
            for row, predecessor in ((root, None), (child, root_key)):
                inputs.append({**{k: row[k] for k in ("id", "task", "step_id", "prompt_tokens", "token_ids_uint32le_sha256")},
                               "predecessor": predecessor, "session_id": root_key, "offered_s": 0.0,
                               "tool_gap_s": 0, "max_tokens": max(8192, counts[row["id"]])})
        if not overlap:
            raise ValueError(f"No observed original pairs were selected in {family}")
    return {**{k: template[k] for k in ("source_repo", "source_revision", "tokenizer_sha256", "max_context_tokens")},
            "selection": {"pairs": pairs, "historical_outcomes": observed["evidence"], "min_observed_outputs": 4,
                          "unseen_requests_are_for_natural_EOS_calibration": allow_unseen,
                          "minimum_common_prefix_tokens": minimum_common_prefix,
                          "max_natural_observed_output": max_observed_tokens,
                          "source_ledger_sha256": observed["ledger_sha256"], "admission_fit": fit,
                          "shared_root_tokens_upper_by_task": cold_prefix_bound(pairs, ledger_path)},
            "requests": inputs}


def arrange_successors(grouped, root_roles, weights):
    remaining = {family: deque(reversed(rows)) for family, rows in grouped.items()}
    loads, order = [0.0, 0.0], []
    moved_bytes = 0
    while remaining:
        target = int(loads[1] < loads[0])
        heads = [rows[0] for rows in remaining.values()]
        next_rows = [row for row in heads if root_roles[row["root"]] != target]
        row = max(next_rows or heads, key=lambda item: (item["potential_GET_bytes"], item["successor"]))
        key, family = row["successor"], row["task"]
        order.append(key)
        loads[target] += weights[key]
        if root_roles[row["root"]] != target:
            moved_bytes += row["potential_GET_bytes"]
        remaining[family].popleft()
        if not remaining[family]:
            del remaining[family]
    return order, moved_bytes


def frozen_plan(spec, body_receipt, ledger_path, successor_offer_s):
    grouped = defaultdict(list)
    pairs = spec["selection"]["pairs"]
    for pair in pairs:
        grouped[pair["task"]].append(pair)
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["checkpoint_blocks"], row["root"]))
    weights = prefill_weights(body_receipt["wire_body_bytes"])
    roots = routes(root_order(grouped, weights), weights)
    order, moved = arrange_successors(grouped, roots, weights)
    workload = plan(spec, body_receipt, successor_offer_s, admission_fit=spec["selection"]["admission_fit"],
                    ordered_successors=order)
    guard_cold_arrivals(workload)
    cold = {f"P{i}": cold_prefix_bound([p for p in pairs if roots[p["root"]] == i], ledger_path) for i in (0, 1)}
    workload["arrival_policy"] = ("All original roots count in the timed run. Offer the successors in a single window, "
                                  "released by their own original predecessor. Account for body-dependent ingress. "
                                  "The measured native route determines actual recoveries.")
    workload["expanded_checkpoint_calibration"] = {"estimated_GET_under_empty_native_proxy_bytes": moved,
                                                   "projected_cold_root_prefix_by_P": cold,
                                                   "actual_running_P_reservations_may_change_assignment": True}
    return workload


def after_observed_readiness(archive, guard_s=.5):
    workload, by_case, trace_sha256 = read_reference(archive)
    successors = sorted((row for row in workload["request_sources"] if row["predecessor"] is not None),
                        key=lambda row: row["offered_s"])
    first = successors[0]["offered_s"]
    constraints = {}
    for case, rows in by_case.items():
        last_parent_return = max(rows[row["predecessor"]]["P_credit_release_s"] for row in successors)
        last_dependency_delay = max(rows[row["predecessor"]]["end_s"] + row["tool_gap_s"] - row["offered_s"]
                                    for row in successors)
        observed_order = sorted((row["id"] for row in successors), key=lambda key: rows[key]["P_start_s"])
        first_divergence = next(({"planned": spec["id"], "observed": observed,
                                  "planned_predecessor": spec["predecessor"],
                                  "planned_predecessor_done_s": rows[spec["predecessor"]]["end_s"],
                                  "planned_offered_s": spec["offered_s"], "observed_P_begin_s": rows[observed]["P_start_s"]}
                                 for spec, observed in zip(successors, observed_order, strict=True) if spec["id"] != observed), None)
        constraints[case] = {"last_parent_credit_release_s": last_parent_return,
                             "largest_parent_end_minus_successor_offer_s": last_dependency_delay,
                             "first_actual_ingress_order_difference": first_divergence}
    shift = max(max(row["last_parent_credit_release_s"] - first, row["largest_parent_end_minus_successor_offer_s"])
                for row in constraints.values())
    shift = ceil(10 * max(0.0, shift + guard_s)) / 10
    workload["request_sources"] = [{**row, "offered_s": row["offered_s"] + shift}
                                   if row["predecessor"] is not None else row for row in workload["request_sources"]]
    workload["pressure_window_s"] = [t + shift for t in workload["pressure_window_s"]]
    workload["successor_ready_calibration"] = {"reference_trace_sha256": trace_sha256, "initial_successor_offer_s": first,
                                               "common_shift_s": shift, "guard_s": guard_s, "reference_constraints": constraints}
    workload["arrival_policy"] = ("Shift the same ordered successor wave by a common interval derived from all reference "
                                  "arms. Preserve all original dependencies, input bodies, generation budgets and within-wave cadence.")
    return workload
