"""Search native Terminal-Bench replay windows for concentrated RDMA and short EOS tails."""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
from heapq import heappop, heappush
from itertools import pairwise, product
from math import ceil
from pathlib import Path
from statistics import mean, median

from pd_kv_transfer_opt.common import percentile, read_json, read_jsonl, sha256, write_json
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS, ATTENTION_BYTES, STATE_BYTES
from pd_kv_transfer_opt.simulator.decode_supply import simulate_decode
from pd_kv_transfer_opt.simulator.rdma_replay import PORTS, egress_epochs, load_native_archive
from pd_kv_transfer_opt.simulator.terminal_magnitude import ingress_span, select_groups, source_options
from pd_kv_transfer_opt.simulator.terminal_wave import decode_deliveries, ongoing_supply_gap, prefill_wave, profile_supply, sample_ports
from pd_kv_transfer_opt.workloads.terminal_restore_checkpoints import root_order
from pd_kv_transfer_opt.workloads.terminal_restore_expanded import arrange_successors
from pd_kv_transfer_opt.workloads.terminal_restore_order import prefill_weights, routes
from pd_kv_transfer_opt.workloads.terminal_restore_pairs import historical_outputs
from pd_kv_transfer_opt.workloads.terminal_restore_stable import guard_cold_arrivals


TRAINING = (
    ("2026-11-22-r73-terminal-source-restore-expanded", "b2-cap200"),
    ("2026-11-22-r76-terminal-source-restore-density-measured", "b2-cap200"),
)
HOLDOUT = "2026-11-22-r78-terminal-source-restore-stable-tail"
METHODS = "2026-11-22-r69-terminal-source-restore-short-guard-methods"
ORDERS = ("b1", "b2")
CAP_ORDERS = ((100, 200), (200, 100))


@lru_cache(maxsize=None)
def native(path):
    return load_native_archive(path)


def load_source(ledger_path, eos_path, completion_path, revision, results):
    receipt = read_json(eos_path)
    source = read_json(completion_path)
    ledger = {row["id"]: row for row in read_jsonl(ledger_path)}
    counts = source["source_completion_tokens"]
    if (not receipt["source_revision"] == source["source_revision"] == revision
            or receipt["ledger_sha256"] != sha256(ledger_path) or ledger.keys() != counts.keys()):
        raise ValueError("Expected the same original source in every archive")
    paths = [results / key for key in receipt["evidence"]]
    outcomes, evidence = historical_outputs(paths, ledger, revision)
    if evidence != receipt["evidence"] or {key: [list(row) for row in rows] for key, rows in outcomes.items()} != receipt["outcomes"]:
        raise ValueError("The EOS evidence does not reproduce the original receipts")
    sizes = {}
    for path in paths:
        w = read_json(path / "manifest.json")["workload"]
        for key, n in w.get("arrival_order_screen", {}).get("wire_body_bytes", {}).items():
            if key in ledger:
                if n != sizes.setdefault(key, n):
                    raise ValueError(f"Changed source request body size {key}")
    return ledger, counts, outcomes, sizes


class ColdPrefixes:
    """Charge each root for new aligned tokens after local reuse of earlier requests."""

    def __init__(self, ledger):
        self.rows = defaultdict(list)
        self.ledger = ledger
        self.index = {}
        for row in ledger.values():
            self.rows[row["task"]].append(row)
        for task, rows in self.rows.items():
            rows.sort(key=lambda row: row["step_id"])
            self.index[task] = {row["id"]: i for i, row in enumerate(rows)}

    @lru_cache(maxsize=None)
    def shared(self, task, first, second):
        a, b = sorted((self.index[task][first], self.index[task][second]))
        return min(self.rows[task][i]["source_prompt_lcp_to_previous"] for i in range(a + 1, b + 1))

    def work(self, order, placement):
        seen, tokens = defaultdict(list), {}
        for key in order:
            row = self.ledger[key]
            group = seen[row["task"], placement[key]]
            reuse = max((self.shared(row["task"], old, key) for old in group), default=0)
            tokens[key] = row["prompt_tokens"] - reuse // ATTENTION_BLOCK_TOKENS * ATTENTION_BLOCK_TOKENS
            group.append(key)
        return tokens

    def earlier_root_attention(self, pairs, placement):
        """Bound local attention bytes on the P opposite each request's own parent."""
        roots = defaultdict(list)
        for pair in pairs:
            root = pair["root"]
            roots[pair["task"], placement[root]].append(root)
        covered = {}
        for pair in pairs:
            child, root, task = (pair[key] for key in ("successor", "root", "task"))
            other = 1 - placement[root]
            blocks = max((self.shared(task, earlier, child) // ATTENTION_BLOCK_TOKENS
                          for earlier in roots[task, other]), default=0)
            covered[child] = min(blocks, pair["checkpoint_blocks"]) * ATTENTION_BYTES
        return covered


def suffix_tokens(pair, spec):
    return max(1, spec[pair["successor"]]["prompt_tokens"] - ATTENTION_BLOCK_TOKENS * pair["checkpoint_blocks"])


def observed_wave(path, case):
    manifest, digest, arms = native(path)
    arm = arms[case]
    rows = arm[1]
    workload = manifest["workload"]
    pairs = workload["selection"]["pairs"]
    spec = {row["id"]: row for row in workload["request_sources"]}
    sizes = workload["arrival_order_screen"]["wire_body_bytes"]
    roots = {k: r for k, r in rows.items() if r["predecessor"] is None}
    children = {k: r for k, r in rows.items() if r["predecessor"] is not None}
    start = min(r["offered_s"] for r in children.values())
    if any(r["finish_reason"] != "stop" for r in rows.values()) or max(x["end_s"] for x in roots.values()) > start:
        raise ValueError(f"Expected a ready, natural-EOS observed wave in {path.name}/{case}")
    epochs = egress_epochs(arm)
    by_port = [sum(max(0.0, right - max(left, start)) * rates[port] / 8 for left, right, rates in epochs)
               for port in range(PORTS)]
    total = sum(by_port)
    potential = {p["successor"]: (p["potential_GET_bytes"] / 1e9 if rows[p["root"]]["P"] != rows[p["successor"]]["P"] else 0)
                 for p in pairs}
    fraction = total / sum(potential.values())
    if not .0 < fraction <= 1.0:
        raise ValueError("Training wave needs identifiable cross-P checkpoint reads")
    jobs = {p["successor"]: {"P": children[p["successor"]]["P"], "P_admit_s": children[p["successor"]]["P_start_s"],
                             "GET_GB": potential[p["successor"]] * fraction, "suffix_tokens": suffix_tokens(p, spec),
                             "output_tokens": children[p["successor"]]["output_tokens"],
                             "decode_score": sizes[p["successor"]]} for p in pairs}
    sample_period = median((b["monotonic_ns"] - a["monotonic_ns"]) / 1e9 for a, b in pairwise(arm[0]["telemetry"]))
    return {"archive": path.name, "case": case, "trace_sha256": digest, "rows": rows, "jobs": jobs,
            "wave_start_s": start, "shares": [x / total for x in by_port], "wire_GB": total, "wire_fraction": fraction,
            "P_reply_s": {k: row["P_end_s"] for k, row in children.items()},
            "sample_period_s": sample_period,
            "root_handoff_s": median(r["first_s"] - r["P_end_s"] for r in roots.values()),
            "successor_handoff_s": median(r["first_s"] - r["P_end_s"] for r in children.values()),
            "body_sizes": sizes}


def fit_prefill(training, matched_widths):
    def knots(values):
        return (min(values), *(percentile(values, q) for q in (.25, .5, .75, 1.0)))

    estimates = []
    for width, launch, nonwire, efficiency, warm_rate in product(
            sorted(set(matched_widths)), (.15, .30, .45, .60, .75), (.03, .12, .22, .32),
            (.88, .94, 1.0), (12_400, 24_800)):
        costs = {"GET_batch_width": width, "first_GET_launch_s": launch, "batch_nonwire_s": nonwire,
                 "rail_efficiency": efficiency, "warm_tokens_s": warm_rate}
        errors = []
        for wave in training:
            observed = knots(wave["P_reply_s"].values())
            modeled, _, _, _ = prefill_wave(wave["jobs"], 200, wave["shares"], costs)
            errors.append(mean(abs(a - b) for a, b in zip(observed, knots(modeled.values()), strict=True)))
        estimates.append((mean(errors), costs, errors))
    best, alternative = sorted(estimates, key=lambda row: row[0])[:2]
    return (best[1], best[2]), (alternative[1], alternative[2])


def run_successors(jobs, cap, shares, start, state, prefill, decode, *, measured_placements=None, root_work=None):
    reply, epochs, calls, wire = prefill_wave(jobs, cap, shares, prefill)
    generation_jobs, P_replies = jobs, reply
    if root_work is not None:
        roots, root_replies = root_work
        generation_jobs = {**{key: {**row, "handoff_s": state["root_handoff_s"]} for key, row in roots.items()}, **jobs}
        P_replies = {**root_replies, **reply}
    simulation = decode_deliveries(generation_jobs, P_replies, decode, 16, state["successor_handoff_s"], placements=measured_placements)
    shaped = sample_ports(epochs, start, state["sample_period_s"])
    exact = profile_supply(jobs, reply, simulation, epochs, cap, start)
    sampled = profile_supply(jobs, reply, simulation, shaped, cap, start)
    tail = simulation["per_request"][simulation["tail_id"]]
    finish = simulation["finish_s"] + state["common_D_tail_residual_s"]
    return {"P_reply_s": reply, "sim": simulation, "epochs": epochs, "report": {
        **exact, "tail": {"id": simulation["tail_id"], "natural_output_tokens": generation_jobs[simulation["tail_id"]]["output_tokens"],
                          "first_s": tail["first_s"], "EOS_s": tail["end_s"],
                          "first_token_to_EOS_s": tail["end_s"] - tail["first_s"]},
        "duration_s": finish, "D_TX_GB_by_port": wire,
        "sampled_longest_aggregate_at_least_100_Gbps_s": sampled["longest_aggregate_at_least_100_Gbps_s"],
        "sampled_port_hot_s": sampled["port_hot_s"], "sampled_D_supply": sampled["D"],
        "GET_batches": sum(call["GET_GB"] > 0 for call in calls),
    }}


def root_deliveries(order, placement, admissions, work, outputs, sizes, state):
    """Return root P replies and decode work on the same clock as the planned offers."""
    jobs, reply = {}, {}
    free = [0.0, 0.0]
    for key in order:
        role = placement[key]
        free[role] = max(free[role], admissions[key]) + work[key] / state["cold_tokens_s_per_P"]
        reply[key] = free[role]
        jobs[key] = {"P_admit_s": admissions[key], "decode_score": sizes[key], "output_tokens": outputs[key]}
    return jobs, reply


def root_drain(order, placement, admissions, work, outputs, sizes, state, decode):
    jobs, reply = root_deliveries(order, placement, admissions, work, outputs, sizes, state)
    state_report = decode_deliveries(jobs, reply, decode, 16, state["root_handoff_s"])
    return {"first_D_s": min(row["first_s"] for row in state_report["per_request"].values()),
            "latest_EOS_s": state_report["finish_s"], "last_P_s": max(reply.values()),
            "P_total_novel_tokens_by_role": [sum(work[key] for key in order if placement[key] == role) for role in (0, 1)]}


def reference_state(wave, prefixes, prefill, decode):
    root = {k: row for k, row in wave["rows"].items() if not row["predecessor"]}
    order = sorted(root, key=lambda k: (root[k]["P_start_s"], k))
    observed_placement = {k: int(row["P"][1:]) for k, row in root.items()}
    measured = max(row["P_end_s"] for row in root.values())
    work = prefixes.work(order, observed_placement)
    rate = max(sum(work[k] for k in order if observed_placement[k] == r) for r in (0, 1)) / measured
    state = {key: wave[key] for key in ("archive", "case", "trace_sha256", "shares", "wire_fraction", "sample_period_s",
                                        "root_handoff_s", "successor_handoff_s")}
    state.update(cold_tokens_s_per_P=rate, common_D_tail_residual_s=0.0)
    rooted = root_drain(order, observed_placement, {k: root[k]["P_start_s"] for k in order}, work,
                        {k: row["output_tokens"] for k, row in root.items()}, wave["body_sizes"], state, decode)
    observed_root_eos = max(row["end_s"] for row in root.values())
    state["measured_root_EOS_minus_model_s"] = observed_root_eos - rooted["latest_EOS_s"]
    high = run_successors(wave["jobs"], 200, wave["shares"], wave["wave_start_s"], state, prefill, decode)
    observed_finish = max(row["end_s"] for row in wave["rows"].values())
    # This common correction cannot add to the modeled 100 vs 200 time gap.
    state["common_D_tail_residual_s"] = max(0.0, observed_finish - high["report"]["duration_s"])
    return state


def guarded_arrivals(roots, children, sizes, fit):
    workload = {"request_sources": [{"id": key, "offered_s": 0.0} for key in roots],
                "arrival_order_screen": {"wire_body_bytes": sizes, "admission_fit": fit, "planned_root_order": roots}}
    guard_cold_arrivals(workload)
    root_offers = {r["id"]: r["offered_s"] for r in workload["request_sources"]}
    ingress = {key: fit["offset_s"] + fit["s_per_byte"] * sizes[key] for key in (*roots, *children)}
    root_admits = {key: root_offers[key] + ingress[key] for key in roots}
    child_admits = {key: ingress[children[0]] + ingress_span(i + 1) for i, key in enumerate(children)}
    child_offers = {key: at - ingress[key] for key, at in child_admits.items()}
    return root_offers, root_admits, child_offers, child_admits


def select_eos(pairs, outcomes, quantile):
    return {key: percentile([length for _, length in outcomes[key]], quantile)
            for pair in pairs for key in (pair["root"], pair["successor"])}


def ingress_residuals(arms, sizes, fit):
    """Read proxy ingress time in excess of the request-body fit on the client's clock."""
    def phase(rows, root):
        keys = sorted((key for key, row in rows.items() if (row["predecessor"] is None) == root),
                      key=lambda key: (rows[key]["offered_s"], key))
        return [rows[key]["P_start_s"] - rows[key]["ready_s"] - fit["offset_s"] - fit["s_per_byte"] * sizes[key]
                for key in keys]

    return {case: [phase(arm[1], root) for root in (True, False)] for case, arm in arms.items()}


def offsets_for_order(order, residuals):
    """Match initial position, then sample the archived residuals by relative position."""
    lead = min(len(order), len(residuals), 4)
    values = list(residuals[:lead])
    count = len(order) - lead
    if count:
        last = len(residuals) - 1
        values.extend(residuals[min(last, lead + round(i * (last - lead) / max(1, count - 1)))] for i in range(count))
    return dict(zip(order, values, strict=True))


def protect_order(order, offers, admissions, profiles):
    """Widen only gaps that fail the empirical ingress test, for all bandwidths together."""
    delays = {order[0]: 0.0}
    for previous, key in pairwise(order):
        nominal_gap = admissions[key] - admissions[previous]
        required = max(profile[previous] - profile[key] for profile in profiles) + .002
        delays[key] = delays[previous] + max(0.0, required - nominal_gap)
    return ({key: t + delays[key] for key, t in offers.items()},
            {key: t + delays[key] for key, t in admissions.items()})


def scored_prefill(admissions, weights, releases):
    active, running, roles = [0.0, 0.0], [], {}
    for key, at in sorted(admissions.items(), key=lambda item: (item[1], item[0])):
        while running and running[0][0] <= at:
            _, role, credit = heappop(running)
            active[role] = max(0.0, active[role] - credit)
        role = int(active[1] < active[0])
        roles[key] = role
        active[role] += weights[key]
        if key in releases:
            heappush(running, (max(at, releases[key]), role, weights[key]))
    return roles


def successor_jobs(children, root_p, child_p, admissions, eos, state, by_child, ledger, sizes, *, local_attention=None):
    jobs = {}
    for key in children:
        pair = by_child[key]
        role = child_p[key]
        reusable = local_attention.get(key, 0) if local_attention is not None else 0
        remote = (pair["potential_GET_bytes"] - reusable) / 1e9 if role != root_p[pair["root"]] else 0.0
        jobs[key] = {"P": f"P{role}", "GET_GB": remote * state["wire_fraction"], "P_admit_s": admissions[key],
                     "suffix_tokens": suffix_tokens(pair, ledger), "output_tokens": eos[key], "decode_score": sizes[key]}
    return jobs


def routed_wave(admissions, root_p, sizes, weights, eos, state, prefill, decode, cap, start, by_child, ledger,
                *, lead_credit_s=0.0, root_work=None, local_attention=None):
    roles = scored_prefill(admissions, weights, {})
    seen = set()
    for _ in range(len(admissions) + 1):
        layout = tuple(roles.items())
        if layout in seen:
            break
        seen.add(layout)
        jobs = successor_jobs(admissions, root_p, roles, admissions, eos, state, by_child, ledger, sizes,
                              local_attention=local_attention)
        sim = run_successors(jobs, cap, state["shares"], start, state, prefill, decode, root_work=root_work)
        releases = {key: sim["sim"]["per_request"][key]["first_s"] - lead_credit_s for key in admissions}
        updated = scored_prefill(admissions, weights, releases)
        if updated == roles:
            arm = sim["report"]
            arm.update(output_tokens=sum(eos.values()), throughput_tokens_s=sum(eos.values()) / arm["duration_s"],
                       min_score_release_after_last_ingress_s=min(releases.values()) - max(admissions.values()))
            return sim, roles
        roles = updated
    return None, None


def clock_sensitivity(selected, pairs, sizes, fit, prefixes, ledger, prefill, decode, references,
                      empirical_profiles, root_speed_fraction, credit_lead_s, *, fixed_wave_start_s=None,
                      keep_offers=False, compare_fixed_D=False, earlier_root_retention=0.0):
    if not 0 <= earlier_root_retention <= 1:
        raise ValueError("Earlier-root retention must be a fraction")
    roots = sorted(selected["root_offers"], key=selected["root_offers"].__getitem__)
    children = sorted(selected["successor_offers"], key=lambda key: (selected["successor_offers"][key]
                      + fit["s_per_byte"] * sizes[key], key))
    profiles = {name: [offsets_for_order(order, residues) for order, residues in zip((roots, children), rows, strict=True)]
                for name, rows in empirical_profiles.items()}
    if keep_offers:
        if fixed_wave_start_s is None:
            raise ValueError("Frozen offers need their fixed successor wave start")
        starts = selected["root_offers"]
        offers = {key: t - fixed_wave_start_s for key, t in selected["successor_offers"].items()}
        root_admits = {key: t + fit["offset_s"] + fit["s_per_byte"] * sizes[key] for key, t in starts.items()}
        child_admits = {key: t + fit["offset_s"] + fit["s_per_byte"] * sizes[key] for key, t in offers.items()}
    else:
        starts, original_root_admits, offers, original_child_admits = guarded_arrivals(roots, children, sizes, fit)
        starts, root_admits = protect_order(roots, starts, original_root_admits, [p[0] for p in profiles.values()])
        offers, child_admits = protect_order(children, offers, original_child_admits, [p[1] for p in profiles.values()])
    weights = prefill_weights(sizes)
    by_child = {p["successor"]: p for p in pairs}
    prepared = {}
    for name, (root_jitter, child_jitter) in profiles.items():
        arrivals = {key: t + root_jitter[key] for key, t in root_admits.items()}
        root_roles = scored_prefill(arrivals, weights, {})
        ordered = sorted(arrivals, key=arrivals.__getitem__)
        work = prefixes.work(ordered, root_roles)
        for order, state, eos in zip(ORDERS, references, selected["eos_samples"], strict=True):
            slow_state = {**state, "cold_tokens_s_per_P": state["cold_tokens_s_per_P"] * root_speed_fraction}
            root = root_drain(ordered, root_roles, arrivals, work, eos, sizes, slow_state, decode)
            fast_root = root_drain(ordered, root_roles, arrivals, work, eos, sizes, state, decode)
            if fast_root["first_D_s"] <= max(arrivals.values()) + credit_lead_s:
                return None
            delayed = {key: t + child_jitter[key] for key, t in child_admits.items()}
            prepared[name, order] = (root, root_roles, delayed)
    start = ceil((.60 + max(row[0]["latest_EOS_s"] for row in prepared.values())) * 10) / 10
    if fixed_wave_start_s is not None:
        if fixed_wave_start_s < start:
            return None
        start = fixed_wave_start_s
    predictions = {}
    for (ref, order), (root, layout, times) in prepared.items():
        index = ORDERS.index(order)
        reference, eos = references[index], selected["eos_samples"][index]
        arrivals = {key: start + t for key, t in times.items()}
        cases, roles = {}, {}
        local_attention = {key: bytes_ * earlier_root_retention for key, bytes_ in
                           prefixes.earlier_root_attention(pairs, layout).items()} if earlier_root_retention else None
        for cap in CAP_ORDERS[index]:
            arm, assignment = routed_wave(arrivals, layout, sizes, weights, eos, reference, prefill, decode, cap, start,
                                          by_child, ledger, lead_credit_s=credit_lead_s, local_attention=local_attention)
            if arm is None:
                return None
            cases[str(cap)], roles[cap] = arm, assignment
        ongoing = ongoing_supply_gap(cases["100"], cases["200"])
        reports = {cap: row["report"] for cap, row in cases.items()}
        low, high = (reports[str(cap)] for cap in (100, 200))
        low_GB, high_GB = (sum(arm["D_TX_GB_by_port"]) for arm in (low, high))
        result = {"arms": reports, "root": root, "100_low_loss_fraction": 1 - high["duration_s"] / low["duration_s"],
                                           "ongoing_D_supply_gap": ongoing,
                                           "same_P_routes": sum(roles[100][k] == roles[200][k] for k in children),
                                           "P_routes_by_cap": {str(cap): value for cap, value in roles.items()},
                                           "100_to_200_total_wire_fraction_change": (high_GB - low_GB) / max(low_GB, high_GB)
                                           if max(low_GB, high_GB) else 0.0,
                                           "100_has_more_D_idle_with_P_pending_and_port_hot": all(
                                               low["D"][role]["idle_with_pending_P_and_hot_D_port_s"] > high["D"][role]["idle_with_pending_P_and_hot_D_port_s"]
                                               for role in ("D0", "D1"))}
        if compare_fixed_D:
            placements = {key: row["D"] for key, row in cases["200"]["sim"]["per_request"].items()}
            jobs = successor_jobs(arrivals, layout, roles[100], arrivals, eos, reference, by_child, ledger, sizes,
                                  local_attention=local_attention)
            fixed = decode_deliveries(jobs, cases["100"]["P_reply_s"], decode, 16, reference["successor_handoff_s"],
                                      placements=placements)
            finish = fixed["finish_s"] + reference["common_D_tail_residual_s"]
            result["fixed_to_200_D_placements"] = {"100_finish_s": finish, "200_finish_s": high["duration_s"],
                                                    "100_throughput_tokens_s": low["output_tokens"] / finish,
                                                    "200_throughput_tokens_s": high["throughput_tokens_s"],
                                                    "100_low_loss_fraction": 1 - high["duration_s"] / finish,
                                                    "number_of_changed_D_assignments": sum(
                                                        row["D"] != placements[key] for key, row in cases["100"]["sim"]["per_request"].items())}
        predictions[f"{ref}/{order}"] = result
    return {"wave_start_s": start, "root_speed_fraction": root_speed_fraction, "early_credit_stress_s": credit_lead_s,
            "case_pairs": predictions, "root_offers": starts,
            "successor_offers": selected["successor_offers"] if keep_offers else {key: start + t for key, t in offers.items()}}


def evaluate(layout, outcomes, ledger, sizes, fit, prefill, decode, references, target):
    pairs, roots, children, root_p, work, successor_p = layout
    nominal_pool_bytes = ((sum(work.values()) + sum(suffix_tokens(pair, ledger) for pair in pairs))
                          * ATTENTION_BYTES / ATTENTION_BLOCK_TOKENS + 2 * len(pairs) * STATE_BYTES)
    root_offers, root_admits, child_offers, child_admits = guarded_arrivals(roots, children, sizes, fit)
    if sorted(roots, key=root_admits.__getitem__) != roots or min(child_offers.values()) < -1e-8:
        return None
    eos_samples = [select_eos(pairs, outcomes, q) for q in (.40, .60)]
    root_states = [root_drain(roots, root_p, root_admits, work, eos, sizes, state, decode)
                   for eos, state in zip(eos_samples, references, strict=True)]
    if any(root["first_D_s"] <= max(root_admits.values()) for root in root_states):
        return None
    estimate = max(root["latest_EOS_s"] + max(0.0, state["measured_root_EOS_minus_model_s"])
                   for root, state in zip(root_states, references, strict=True))
    wave_start = ceil((estimate + .60) * 10) / 10
    pressure = {}
    by_key = {p["successor"]: p for p in pairs}
    for index, (order, caps, state, eos) in enumerate(zip(ORDERS, CAP_ORDERS, references, eos_samples, strict=True)):
        jobs = successor_jobs(children, root_p, successor_p, {key: wave_start + t for key, t in child_admits.items()},
                              eos, state, by_key, ledger, sizes)
        cases = {}
        for cap in caps:
            sim = run_successors(jobs, cap, state["shares"], wave_start, state, prefill, decode)
            arm = sim["report"]
            arm.update(output_tokens=sum(eos.values()), throughput_tokens_s=sum(eos.values()) / arm["duration_s"])
            arm["D_local_credit_guard_s"] = arm["first_D_token_s"] - max(job["P_admit_s"] for job in jobs.values())
            cases[str(cap)] = arm
        low, high = (cases[str(cap)] for cap in (100, 200))
        pressure[order] = {"arms": cases, "root": root_states[index],
                           "natural_EOS_work_tokens": sum(eos.values()),
                           "100_low_loss_fraction": 1 - high["duration_s"] / low["duration_s"],
                           "finish_gap_s": low["duration_s"] - high["duration_s"],
                           "D_idle_with_live_P_and_hot_port_reduction_s": {
                               role: low["D"][role]["idle_with_pending_P_and_hot_D_port_s"]
                               - high["D"][role]["idle_with_pending_P_and_hot_D_port_s"] for role in ("D0", "D1")}}
    ref_min = max(target["actual_r69_loss_by_order"].values())
    best_worst = min(row["100_low_loss_fraction"] for row in pressure.values())
    hit_concentration = all(row["arms"]["200"]["peak_20_s_D_TX_Gbps"] > target["r69_max_200_Gbps_in_20_s"]
                            for row in pressure.values())
    supply_relief = all(min(row["D_idle_with_live_P_and_hot_port_reduction_s"].values()) > 0
                        and row["arms"]["100"]["sampled_port_hot_s"] > 0
                        and row["arms"]["200"]["D_first_tokens_in_first_10_s"] > row["arms"]["100"]["D_first_tokens_in_first_10_s"]
                        for row in pressure.values())
    credit_clean = min(arm["D_local_credit_guard_s"] for row in pressure.values() for arm in row["arms"].values()) > 0
    return {"wave_start_s": wave_start, "root_offers": root_offers, "nominal_aligned_unique_checkpoint_GiB": nominal_pool_bytes / 2**30,
            "successor_offers": {key: wave_start + v for key, v in child_offers.items()},
            "pressure": pressure, "worst_100_low_loss_fraction": best_worst, "beats_both_r69_actual_pairs_in_model": best_worst > ref_min,
            "concentration_above_r69": hit_concentration, "D_supply_relief_both_roles_and_states": supply_relief,
            "native_route_arrivals_before_first_score_release": credit_clean, "eos_samples": eos_samples}


def search(outcomes, counts, ledger, sizes, fit, prefixes, prefill, decode, prepared, target, *, wider=False):
    base_grid = ((500, 550, 600, 650), range(0, 15_000_001, 3_000_000),
                 (230_000, 260_000, 290_000, 320_000, 350_000, 380_000, 410_000, 450_000, 500_000))
    deeper_grid = ((575, 600, 625, 650, 675), range(1_500_000, 13_500_001, 3_000_000),
                   tuple(range(275_000, 486_000, 15_000)))
    wider_grid = ((575, 600, 625, 650, 675), (0, 3_000_000, 7_500_000, 12_000_000, 15_000_000),
                  tuple(range(450_000, 600_001, 25_000)))
    grids = (wider_grid,) if wider else (base_grid, deeper_grid)
    window_range = range(65, 89) if wider else range(1, 65)
    weights = prefill_weights(sizes)
    cohorts, tried, accepted = set(), 0, []
    for maximums, penalties, budgets in grids:
        for cutoff in maximums:
            options = source_options(ledger, counts, outcomes, sizes, cutoff)
            for penalty in penalties:
                for groups in select_groups(options, budgets, penalty).values():
                    pairs = [p for group in groups for p in group.pairs]
                    keys = tuple(sorted(p["root"] for p in pairs))
                    if len(groups) < 2 or len(pairs) not in window_range or keys in cohorts:
                        continue
                    cohorts.add(keys)
                    grouped = {g.pairs[0]["task"]: sorted(g.pairs, key=lambda p: (p["checkpoint_blocks"], p["root"]))
                               for g in groups}
                    visited = set()
                    for roles in product((0, 1), repeat=len(groups) - 1):
                        if not any(roles):
                            continue
                        root_ordered = root_order(grouped, weights, (0, *roles))
                        if tuple(root_ordered) in visited:
                            continue
                        visited.add(tuple(root_ordered))
                        root_p = routes(root_ordered, weights)
                        # The pinned native heap uses ordinal, so both idle pools start at P0.
                        children, full = arrange_successors(grouped, root_p, weights)
                        if full / 1e9 * max(r["wire_fraction"] for r in prepared) * 8 / 20 < target["r69_max_200_Gbps_in_20_s"]:
                            continue  # Even putting every byte into 20 seconds would be too light.
                        work = prefixes.work(root_ordered, root_p)
                        tried += 1
                        child_p = routes(children, weights)
                        candidate = evaluate((pairs, root_ordered, children, root_p, work, child_p),
                                             outcomes, ledger, sizes, fit, prefill, decode, prepared, target)
                        if candidate is None or (wider and candidate["nominal_aligned_unique_checkpoint_GiB"] > 128):
                            continue
                        viable = all(candidate[key] for key in ("concentration_above_r69", "D_supply_relief_both_roles_and_states",
                                                                "native_route_arrivals_before_first_score_release"))
                        rank = (viable, candidate["concentration_above_r69"], candidate["worst_100_low_loss_fraction"])
                        accepted.append((rank, candidate, pairs, root_p, child_p, full, cutoff))
    if not accepted:
        raise ValueError("No source cohort met the modeled predecessor and root-ingress constraints")
    return accepted, {"original_source_cohorts": len(cohorts), "simulated_native_layouts": tried,
                    "candidate_layouts_after_root_gate": len(accepted), "jointly_meet_concentration_relief_and_route_gates": sum(row[0][0] for row in accepted),
                    "jointly_meet_all_and_beat_r69": sum(row[0][0] and row[1]["beats_both_r69_actual_pairs_in_model"] for row in accepted),
                    "maximum_observed_EOS_cutoffs_searched": sorted({cutoff for grid in grids for cutoff in grid[0]})}


def loss_after_residual(pair, residual):
    low, high = (pair["arms"][str(cap)]["duration_s"] for cap in (100, 200))
    return 1 - high / max(high, low - residual)


def rank_stress(choices, target, sizes, fit, prefixes, ledger, prefill, decode, references, profiles, cold, lead, residual):
    floor = max(target["actual_r69_loss_by_order"].values())
    best, evaluated, passed, passes_residual = None, 0, 0, 0
    for base in choices:
        if not base[0][0] or not base[1]["beats_both_r69_actual_pairs_in_model"]:
            continue
        selected, pairs = base[1:3]
        stress = clock_sensitivity(selected, pairs, sizes, fit, prefixes, ledger, prefill, decode, references, profiles, cold, lead)
        if stress is None:
            continue
        evaluated += 1
        cases = stress["case_pairs"].values()
        worst = min(p["100_low_loss_fraction"] for p in cases)
        corrected = min(loss_after_residual(p, residual) for p in cases)
        equal = all(p["same_P_routes"] == len(pairs) for p in cases)
        feeds = all(p["100_has_more_D_idle_with_P_pending_and_port_hot"]
                    and p["ongoing_D_supply_gap"]["100_fewer_ready_s"] > 0
                    and p["arms"]["100"]["D_first_tokens_in_first_10_s"]
                    < p["arms"]["200"]["D_first_tokens_in_first_10_s"] for p in cases)
        concentrated = min(p["arms"]["200"]["peak_20_s_D_TX_Gbps"] for p in cases) > target["r69_max_200_Gbps_in_20_s"]
        comparable = max(abs(p["100_to_200_total_wire_fraction_change"]) for p in cases) < .05
        pass_all = feeds and concentrated and equal and comparable and worst > floor
        passed += pass_all
        passes_residual += pass_all and corrected > floor
        score = (pass_all, feeds and concentrated and comparable, corrected, worst)
        if best is None or score > best[0]:
            best = (score, base, stress)
    return best, {"tested_nominally_admissible_layouts": evaluated, "layouts_pass_every_stress_pair": passed,
                  "layouts_still_above_r69_after_heldout_residual": passes_residual}


def run(args):
    screening = read_json(args.screening)
    plan = screening["plan"]
    ledger, counts, outcomes, sizes = load_source(args.ledger, args.eos_receipt, args.source_completions,
                                                  plan["source_revision"], args.results)
    prefixes = ColdPrefixes(ledger)
    training = [observed_wave(args.results / name, case) for name, case in TRAINING]
    methods = read_json(args.results / METHODS / "native-method-summary.json")
    decode = screening["decode_service_fit"]
    (prefill, errors), (alternate_prefill, alternate_errors) = fit_prefill(
        training, [len(row["request_ids"]) for row in methods["matched_four_rank_GET_batches"]])
    references = [reference_state(wave, prefixes, prefill, decode) for wave in training]
    m, heldout_digest, heldout = native(args.results / HOLDOUT)
    fit = m["workload"]["arrival_order_screen"]["admission_fit"]
    holdout_checks = {}
    heldout_state = {"common_D_tail_residual_s": median(r["common_D_tail_residual_s"] for r in references)}
    for order in ORDERS:
        wave = observed_wave(args.results / HOLDOUT, f"{order}-cap200")
        holdout_checks[order] = {}
        for cap in (100, 200):
            state = {**wave, **heldout_state}
            modeled = run_successors(wave["jobs"], cap, wave["shares"], wave["wave_start_s"], state, prefill, decode)["report"]
            actual = heldout[f"{order}-cap{cap}"][1]
            holdout_checks[order][str(cap)] = {
                "observed_D_first_tokens_in_10_s": sum(r["predecessor"] is not None and r["first_s"] <= wave["wave_start_s"] + 10
                                                        for r in actual.values()),
                "observed_P_last_reply_s": max(row["P_end_s"] for row in actual.values()),
                "observed_first_successor_D_token_s": min(row["first_s"] for row in actual.values() if row["predecessor"]),
                "observed_duration_s": max(row["end_s"] for row in actual.values()),
                "observed_full_output_tokens": sum(row["output_tokens"] for row in actual.values()),
                "model_fixed_200_full_output_tokens": sum(row["output_tokens"] for row in wave["rows"].values()),
                "model": modeled}
    # r78 b1 is the equal-P, equal-wire holdout. Freeze both actual outputs and D routes for this check.
    paired_low, paired_high = (heldout[f"b1-cap{cap}"][1] for cap in (100, 200))
    observed_P_only_gap = simulate_decode(paired_high, decode, prefill_http_s={k: row["phases_s"]["P_HTTP"]
                                        for k, row in paired_low.items()})["finish_s"] - simulate_decode(paired_high, decode)["finish_s"]
    anchor = observed_wave(args.results / HOLDOUT, "b1-cap200")
    same_D = {key: paired_high[key]["D"] for key in anchor["jobs"]}
    held_model = [run_successors(anchor["jobs"], cap, anchor["shares"], anchor["wave_start_s"], {**anchor, **heldout_state},
                                 prefill, decode, measured_placements=same_D)["report"]["duration_s"] for cap in (100, 200)]
    observed_gap_optimism_s = max(0.0, held_model[0] - held_model[1] - observed_P_only_gap)
    r69 = screening["selected_bandwidth_pairs"]
    target = {"actual_r69_loss_by_order": {o: p["hardware_low_throughput_loss_fraction"] for o, p in r69.items()},
              "r69_max_200_Gbps_in_20_s": max(p["hardware"]["200"]["pressure"]["peak_window_average_Gbps"] for p in r69.values())}
    choices, search_stats = search(outcomes, counts, ledger, sizes, fit, prefixes, prefill, decode, references, target)
    residue = {}
    for path in (HOLDOUT, *(name for name, _ in TRAINING)):
        raw_manifest, _, raw_arms = native(args.results / path)
        stats = ingress_residuals(raw_arms, raw_manifest["workload"]["arrival_order_screen"]["wire_body_bytes"], fit)
        residue.update({f"{path}/{name}": value for name, value in stats.items()})
    cold_times = [max(r["P_end_s"] for r in arm[1].values() if not r["predecessor"]) for arm in heldout.values()]
    cold = min(cold_times) / max(cold_times)
    lead = max(0.0, *(cap["model"]["first_D_token_s"] - cap["observed_first_successor_D_token_s"]
                      for order in ORDERS for cap in (holdout_checks[order]["200"],)))
    stress_best, stress_counts = rank_stress(choices, target, sizes, fit, prefixes, ledger, prefill, decode, references, residue, cold, lead,
                                             observed_gap_optimism_s)
    if stress_best is None:
        raise ValueError("No nominally promising workload had a stable native clock replay")
    stress_score, selection, stress = stress_best
    rank, selected, pairs, root_p, child_p, full, limit = selection
    nominal_start = selected["wave_start_s"]
    # The saved schedule includes the observed slow-root and ingress guard at both limits.
    primary = clock_sensitivity(selected, pairs, sizes, fit, prefixes, ledger, prefill, decode, references, residue, 1.0, 0.0,
                                fixed_wave_start_s=stress["wave_start_s"])
    main_cases = {order: primary["case_pairs"][f"{HOLDOUT}/{order}-cap200/{order}"] for order in ORDERS}
    alternate = clock_sensitivity(selected, pairs, sizes, fit, prefixes, ledger, alternate_prefill, decode, references, residue,
                                  cold, lead, fixed_wave_start_s=stress["wave_start_s"])
    alternate_comparison = None if alternate is None else {
        "GET_model": alternate_prefill, "P_reply_quantile_MAE_s": alternate_errors,
        "worst_low_loss_fraction": min(v["100_low_loss_fraction"] for v in alternate["case_pairs"].values()),
        "pairs_with_matched_P": sum(v["same_P_routes"] == len(pairs) for v in alternate["case_pairs"].values()),
        "pairs_with_ongoing_supply_gap": sum(v["ongoing_D_supply_gap"]["100_fewer_ready_s"] > 0 for v in alternate["case_pairs"].values())}
    offers = {**stress.pop("root_offers"), **stress.pop("successor_offers")}
    eos = selected.pop("eos_samples")
    selected.pop("root_offers")
    selected.pop("successor_offers")
    pair_by_child = {p["successor"]: p["root"] for p in pairs}
    manifest = {"baseline_commit": m["baseline_commit"], "settings": m["settings"], "shaping": m["shaping"],
                "source_repo": m["workload"]["source_repo"], "source_revision": plan["source_revision"],
                "ledger_sha256": sha256(args.ledger), "source_budget_sha256": sha256(args.source_completions),
                "evidence_receipt_sha256": sha256(args.eos_receipt),
                "native_route_ordinals": ["P0", "P1"], "ignore_eos": False,
                "temperature": m["workload"]["temperature"],
                "route_policy": "upstream native without route override",
                "maximum_observed_natural_EOS_tokens_at_selection": limit, "pairs": pairs,
                "request_sources": [{**{k: ledger[key][k] for k in ("id", "task", "step_id", "token_ids_uint32le_sha256", "prompt_tokens")},
                                     "predecessor": pair_by_child.get(key), "offered_s": at, "tool_gap_s": 0,
                                     "max_tokens": max(8192, counts[key]),
                                     "observed_EOS_reference_tokens": {o: n[key] for o, n in zip(ORDERS, eos, strict=True)}}
                                    for key, at in sorted(offers.items(), key=lambda item: (item[1], item[0]))]}
    manifest["projected_native_P_roles_without_early_credit"] = {key: f"P{role}" for key, role in {**root_p, **child_p}.items()}
    manifest["observed_reference_quantiles"] = [.4, .6]
    manifest["scope"] = "Original prompts and budgets. Projected roles and EOS reference lengths are simulator data."
    candidate_path = args.output.with_name("candidate-workload.json")
    write_json(candidate_path, manifest)
    write_json(args.output, {"candidate_workload_sha256": sha256(candidate_path), "r69_comparison": target,
                             "GET_model": prefill, "training_P_reply_quantile_MAE_s": errors,
                             "decode_fit": decode, "reference_states": references, "holdout_r78": holdout_checks,
                             "search_stats": search_stats, "selected_candidate_nominal_screen": selected, "original_pairs": len(pairs),
                             "nominal_unprotected_successor_start_s": nominal_start,
                             "guarded_candidate_ABBA": main_cases, "guarded_successor_start_s": stress["wave_start_s"],
                             "clock_search": stress_counts, "guarded_clock_sensitivity": stress,
                             "worst_guarded_stress_100_loss_fraction": stress_score[3], "passes_every_stress_pair": stress_score[0],
                             "alternate_GET_fit_on_same_schedule": alternate_comparison,
                             "r78_same_D_heldout_critical_gap": {"measured_P_swap_s": observed_P_only_gap,
                                 "model_s": held_model[0] - held_model[1], "model_positive_residual_s": observed_gap_optimism_s,
                                 "candidate_ABBA_100_loss_subtracting_that_residual": {
                                     k: loss_after_residual(v, observed_gap_optimism_s) for k, v in main_cases.items()},
                                 "candidate_worst_guarded_100_loss_subtracting_that_residual": stress_score[2]},
                             "r78_clock_anchor_sha256": heldout_digest,
                             "nominal_cross_P_checkpoint_GB_under_native_idle_score": full / 1e9,
                             "passes_concentration_supply_relief_and_route_gates": rank[0],
                             "scope": "Read volume is projected from each distinct original checkpoint and observed realized read fractions "
                                      "conditioned on the actual route, with no traffic multiplier. New-cohort APC hits/Store residency "
                                      "are conditional, and throughput is predicted. Each state uses archived per-request EOS lengths "
                                      "fixed across caps. Quantiles and service states differ between AB and BA. CPU arms start from the "
                                      "same empty logical initial state and hold all offerings apart from the cap. The root model assumes "
                                      "retention of aligned local prefixes and release of the full ready successor wave. D scores are "
                                      "updated during the simulation. Validation for r78 freezes high observed ingress/routes and bytes."})
    print(f"Selected {len(pairs)} original windows, worst guarded 100-cap throughput deficit {stress_score[3]:.2%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--eos-receipt", type=Path, required=True)
    parser.add_argument("--source-completions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
