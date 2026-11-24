"""Replay naturally stopped Terminal-Bench windows under continuous Poisson starts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from math import isclose
from pathlib import Path
from random import Random
from statistics import median

from pd_kv_transfer_opt.common import new_directory, percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.simulator.agent_replay import PAGE, Replay, Service
from pd_kv_transfer_opt.simulator.benchmark_pressure_calibration import ledger_pages
from pd_kv_transfer_opt.simulator.r43_arrival import (
    ARCHIVE, CAPS, PREFIX, SEEDS, emit_requests, exponential_arrivals, flow_window, materialize, metrics,
)
from pd_kv_transfer_opt.simulator.terminal_supply import load_source
from pd_kv_transfer_opt.workloads.fetch_trajectories import REPO, REVISION
from pd_kv_transfer_opt.workloads.freeinference import Session, Turn


CALIBRATION = "2026-11-23-r43-conditional-arrivals"
SELECTION_SEED = 20261123
SESSIONS = 30
STRATA = 5
MIN_TURNS = 6
MAX_TURNS = 12
MAX_CONTEXT = 262144
BASE_RATES = (12, 16)


def covered_runs(rows, outcomes, source_outputs):
    current = []
    for row in sorted(rows, key=lambda r: r["step_id"]):
        key = row["id"]
        budget = min(MAX_CONTEXT - row["prompt_tokens"], max(8192, source_outputs[key]))
        outputs = sorted(n for finish, n in outcomes.get(key, ()) if finish == "stop" and 0 < n < budget)
        eligible = outputs and not row["truncated"]
        if current and (not eligible or row["predecessor"] != current[-1]["id"]):
            yield current
            current = []
        if eligible:
            current.append({**row, "output_tokens": outputs[(len(outputs) - 1) // 2],
                            "natural_EOS_observations": len(outputs), "max_tokens": budget})
    if current:
        yield current


def window_for_task(rows, outcomes, source_outputs):
    """Use the earliest longest run, retaining up to twelve consecutive source calls."""
    best = max(covered_runs(rows, outcomes, source_outputs), key=len, default=[])
    return best[:MAX_TURNS], len(best)


def select_windows(ledger, outcomes, source_outputs):
    tasks = defaultdict(list)
    for row in ledger.values():
        tasks[row["task"]].append(row)
    candidates = []
    for name, rows in tasks.items():
        records, audited_depth = window_for_task(rows, outcomes, source_outputs)
        if len(records) >= MIN_TURNS:
            candidates.append((max(row["prompt_tokens"] for row in records), name, records, audited_depth))
    if len(candidates) < SESSIONS:
        raise ValueError(f"Only {len(candidates)} independent tasks have sufficient EOS coverage")
    candidates.sort()
    rng = Random(SELECTION_SEED)
    strata = [candidates[i * len(candidates) // STRATA:(i + 1) * len(candidates) // STRATA]
              for i in range(STRATA)]
    selected = [row for group in strata for row in rng.sample(group, SESSIONS // STRATA)]
    rng.shuffle(selected)
    return selected, len(candidates)


def check_prefix_anchors(pages, exact_path, windows):
    """Anchor first page identities using the exact r43 prefix ledger where available."""
    with gzip.open(exact_path, "rt", encoding="utf-8") as stream:
        archived = json.load(stream)
    anchor, exact = {}, set()
    for key, value in archived.items():
        source_id = key.removeprefix("natural-").removeprefix("sustained-")
        row, modeled = pages[source_id]
        observed = value["blocks"]
        if row["prompt_tokens"] != value["length"] or len(modeled) != len(observed):
            raise ValueError(f"Exact ledger length differs for {source_id}")
        if modeled and anchor.setdefault(modeled[0], observed[0]) != observed[0]:
            raise ValueError(f"Adjacent prefix ledger disagrees with the first page of {source_id}")
        exact.add(source_id)
    hits, first_to_task = 0, {}
    for _, name, rows, _ in windows:
        for row in rows:
            ids = pages[row["id"]][1]
            if not ids:
                continue
            first = anchor.get(ids[0])
            if first is not None:
                if first_to_task.setdefault(first, name) != name:
                    raise ValueError("Shared cross-task prefix needs exact block identities")
                hits += 1
    return {"exact_r43_ledger_sha256": sha256(exact_path), "turns_with_anchored_first_page": hits,
            "turns_also_present_in_exact_r43_ledger": sum(row["id"] in exact for _, _, rows, _ in windows for row in rows)}


def make_workload(ledger_path, eos_path, source_outputs, results):
    ledger, lengths, outcomes, measured_bodies = load_source(ledger_path, eos_path, source_outputs, REVISION, results)
    windows, eligible = select_windows(ledger, outcomes, lengths)
    pages = ledger_pages(ledger_path)
    checks = check_prefix_anchors(pages, results / PREFIX, windows)
    sessions = []
    for peak, task, rows, depth in windows:
        turns = []
        for step, row in enumerate(rows):
            key = row["id"]
            native_pages = pages[key][1]
            previous_pages = pages[rows[step - 1]["id"]][1] if step else ()
            adjacent = next((i for i, (a, b) in enumerate(zip(previous_pages, native_pages)) if a != b),
                            min(len(previous_pages), len(native_pages)))
            turns.append({**{k: row[k] for k in ("id", "step_id", "prompt_tokens", "output_tokens", "max_tokens",
                                                "natural_EOS_observations", "token_ids_uint32le_sha256")},
                          "source_predecessor": row["predecessor"],
                          "predecessor": rows[step - 1]["id"] if step else None,
                          "source_adjacent_prefix_tokens": row["source_prompt_lcp_to_previous"] if step else 0,
                          "tool_gap_ms": 0, "finish_reason": "stop", "adjacent_pages": adjacent,
                          "pages": native_pages, "wire_body_bytes": measured_bodies.get(key)})
        sessions.append({"task": task, "peak_prompt_tokens": peak, "longest_audited_depth": depth, "turns": turns})
    return {"source_repo": REPO, "source_revision": REVISION,
            "source_ledger_sha256": sha256(ledger_path), "EOS_evidence_sha256": sha256(eos_path),
            "source_outputs_sha256": sha256(source_outputs), "validated_evidence_archives": len(read_json(eos_path)["evidence"]),
            "selection": {"eligible_tasks": eligible, "seed": SELECTION_SEED, "peak_length_strata": STRATA,
                          "min_turns": MIN_TURNS, "max_turns": MAX_TURNS, "sampled_tasks": len(sessions)},
            "prefix_validation": checks, "sessions": sessions}


def replay_inputs(workload):
    sessions, bodies, ids = [], {}, {}
    for rank, record in enumerate(workload["sessions"]):
        task, turns = record["task"], []
        for step, row in enumerate(record["turns"]):
            key = f"{task}:{step}"
            turns.append(Turn(row["tool_gap_ms"], row["prompt_tokens"], row["output_tokens"], tuple(row["pages"]),
                              row["adjacent_pages"], row["finish_reason"]))
            ids[key] = row["id"]
            if row["wire_body_bytes"] is not None:
                bodies[key] = row["wire_body_bytes"]
        sessions.append(Session(task, rank, 0, 0, tuple(turns)))
    return sessions, {session.name: i for i, session in enumerate(sessions)}, bodies, ids


def source_setup(results):
    folder = results / CALIBRATION
    summary, arrivals = read_json(folder / "summary.json"), read_json(folder / "arrivals.json")
    manifest = read_json(results / ARCHIVE / "manifest.json")
    prefix_path = results / PREFIX
    if (summary["sources"]["r43_workload_sha256"] != manifest["workload"]["workload_sha256"]
            or summary["sources"]["prefix_ledger_sha256"] != sha256(prefix_path)
            or summary["artifacts"]["arrivals.json"] != sha256(folder / "arrivals.json")):
        raise ValueError("The r43 service and workload reference is inconsistent")
    spec = {row["id"]: row for row in manifest["workload"]["request_sources"]}
    with gzip.open(prefix_path, "rt", encoding="utf-8") as stream:
        prefix = json.load(stream)
    result = {}
    for model, data in summary["calibration"].items():
        service = Service(**data["parameters"])
        sessions, offers, _ = materialize(spec, prefix, arrivals["EOS_by_block"][model])
        result[model] = (service, sessions, offers)
    return summary, result


def per_session_work(sessions, result):
    return {"P_non_GET_engine_s": sum(result["stage"]["P_compute_wall_s_by_instance"]) / len(sessions),
            "P_computed_tokens": result["cache"]["P_computed_tokens"] / len(sessions),
            "D_output_tokens": result["output_tokens"] / len(sessions), "D_to_P_GET_GB": result["GET_GB"] / len(sessions),
            "P_to_D_Pull_GB": result["P_to_D_Pull_GB"] / len(sessions)}


def check_replay(run, offers, ids, output_tokens):
    if run.requests.keys() != ids.keys() or sum(j.turn.output for j in run.requests.values()) != output_tokens:
        raise ValueError("A replay augmented or changed the frozen workload")
    for job in run.requests.values():
        expected = offers[job.session.name]
        if job.step:
            parent = run.requests[f"{job.session.name}:{job.step - 1}"]
            expected = parent.row["EOS_s"] + job.turn.tool_gap_ms / 1000
        if not isclose(job.offered, expected, abs_tol=1e-8):
            raise ValueError(f"Broken predecessor clock for {job.key}")


def run_rate(rate, sessions, order, bodies, ids, models, raw_stream, book):
    output_tokens = sum(s.output for s in sessions)
    plans = {str(seed): exponential_arrivals(order, seed, rate) for seed in SEEDS}
    book[str(rate)] = plans
    paired = []
    for model, (service, _, _) in models.items():
        for seed, offers in plans.items():
            times = list(offers.values())
            a, b = times[len(times) // 4], times[5 * len(times) // 8]
            arms, routes = {}, {}
            for cap in (CAPS if model == "1" else CAPS[::-1]):
                run = Replay(sessions, offers, service, cap, observed_body_bytes=bodies).run()
                check_replay(run, offers, ids, output_tokens)
                result = metrics(run, offers, detailed=True)
                earlier = flow_window(run, offers, a, b)
                later = flow_window(run, offers, b, times[-1])
                seconds = later["end_s"] - later["begin_s"]
                result.update(earlier=earlier, later=later,
                              D_KV_ready_per_s_late=later["D_KV_ready"] / seconds,
                              active_session_drift=later["active_sessions_mean"] - earlier["active_sessions_mean"],
                              P_queue_drift=later["P_mean_waiting"] - earlier["P_mean_waiting"],
                              per_session_work=per_session_work(sessions, result))
                arms[cap] = result
                routes[cap] = {key: (j.row["P"], j.row["D"]) for key, j in run.requests.items()}
                emit_requests(raw_stream, run, ids, f"poisson-{rate:g}-seed-{seed}", cap, int(model))
            paired.append({"seed": int(seed), "model": int(model), "arms": arms,
                           "200_over_100_pct": 100 * (arms[200]["throughput_tokens_s"] / arms[100]["throughput_tokens_s"] - 1),
                           "same_routes": [sum(routes[100][k][i] == routes[200][k][i] for k in ids) for i in range(2)]})
    return paired


def calibrate_rate(rate, pairs, reference, models):
    frozen = {(str(row["block"]), row["seed"]): row for row in reference["continuous_rate_scan"]
              if row["rate_sessions_per_min"] == rate and row["cap_Gbps"] == 200}
    checks = []
    for pair in pairs:
        model, seed = str(pair["model"]), pair["seed"]
        service, sessions, original = models[model]
        offers = exponential_arrivals(original, seed, rate)
        run = Replay(sessions, offers, service, 200).run()
        if not isclose(run.now, frozen[model, seed]["finish_s"], abs_tol=1e-8):
            raise ValueError("Work-rate calibration failed to reproduce the archived r43 CPU run")
        result = metrics(run, offers)
        baseline = per_session_work(sessions, result)
        current = pair["arms"][200]["per_session_work"]
        checks.append({"model": int(model), "seed": seed, "reference_r43": baseline, "new_workload": current,
                       "P_work_ratio": current["P_non_GET_engine_s"] / baseline["P_non_GET_engine_s"],
                       "D_work_ratio": current["D_output_tokens"] / baseline["D_output_tokens"]})
    p_ratio = median(row["P_work_ratio"] for row in checks)
    d_ratio = median(row["D_work_ratio"] for row in checks)
    return {"base_rate": rate, "P_ratio_median": p_ratio, "D_ratio_median": d_ratio,
            "calibrated_rate": round(rate / max(p_ratio, d_ratio), 1), "paired_work": checks}


def stats(samples):
    return {"median": median(samples), "min": min(samples), "max": max(samples)}


def summarize(pairs):
    fields = ("throughput_tokens_s", "finish_s", "final_root_s", "drain_after_last_root_s", "GET_GB", "P_to_D_Pull_GB",
              "P_HBM_hit_tokens", "D_HBM_hit_tokens", "D_KV_ready_per_s_late", "active_session_drift", "P_queue_drift")
    arms = {}
    for cap in CAPS:
        group = [row["arms"][cap] for row in pairs]
        summary = {field: stats([a[field] for a in group]) for field in fields}
        summary.update(
            P_queue_p95_s=stats([a["P_queue"]["all"]["p95_s"] for a in group]),
            P_root_queue_p95_s=stats([a["P_queue"]["roots"]["p95_s"] for a in group]),
            P_computed_tokens=stats([a["cache"]["P_computed_tokens"] for a in group]),
            P_reply_in_offer_window=stats([a["during_root_window"]["P_replies"] for a in group]),
            D_KV_ready_in_offer_window=stats([a["during_root_window"]["D_KV_ready"] for a in group]),
            D_mean_active_late=stats([a["later"]["D_mean_active_sequences"] for a in group]),
            D_generated_tokens_s_late=stats([a["later"]["D_generation_tokens_s"] for a in group]),
            D_GET_mean_Gbps_late=stats([a["later"]["D_to_P_GET_mean_Gbps"] for a in group]),
            D_empty_while_pending_P_and_GET_instance_s=stats([
                a["all_time"]["D_empty_with_P_pending_during_GET_instance_s"] for a in group]),
            GET_peak_10s_Gbps=stats([a["peak_D_to_P_aggregate_Gbps"]["10"] for a in group]),
            last_session_counts={name: sum(a["tail_session"]["session"] == name for a in group)
                                 for name in sorted({a["tail_session"]["session"] for a in group})},
        )
        arms[cap] = summary
    return {"arms": arms, "200_over_100_pct": stats([row["200_over_100_pct"] for row in pairs])}


def execute(args):
    workload = make_workload(args.ledger, args.eos_receipt, args.source_completions, args.results)
    sessions, order, bodies, ids = replay_inputs(workload)
    reference, models = source_setup(args.results)
    output = new_directory(args.output)
    write_json(output / "workload.json", workload)
    counts = [t.output for s in sessions for t in s.turns]
    prompts = [t.prompt for s in sessions for t in s.turns]
    result = {"workload": {"sessions": len(sessions), "requests": len(prompts), "prompt_tokens": sum(prompts),
                           "output_tokens": sum(counts), "prompt_p50": median(prompts), "prompt_p95": percentile(prompts, .95),
                           "EOS_p50": median(counts), "EOS_p95": percentile(counts, .95),
                           "EOS_max": max(counts), "archived_proxy_body_size_turns": len(bodies)},
              "controls": reference["controls"]["settings"], "source_calibration_sha256": sha256(args.results / CALIBRATION / "summary.json"),
              "seeds": SEEDS, "rates": {}, "calibration": [], "summary_by_rate": {}}
    root_plans = {}
    with gzip.open(output / "requests.jsonl.gz", "wt", encoding="utf-8") as trace:
        for rate in BASE_RATES:
            arms = run_rate(rate, sessions, order, bodies, ids, models, trace, root_plans)
            result["rates"][str(rate)] = arms
            result["calibration"].append(calibrate_rate(rate, arms, reference, models))
            print(f"{rate:g} sessions/min: paired runs and r43 work comparison complete", flush=True)
        for item in result["calibration"]:
            rate = item["calibrated_rate"]
            label = f"{rate:g}"
            if label not in result["rates"]:
                result["rates"][label] = run_rate(rate, sessions, order, bodies, ids, models, trace, root_plans)
                print(f"{label} sessions/min: calibrated paired runs complete", flush=True)
    result["summary_by_rate"] = {rate: summarize(pairs) for rate, pairs in result["rates"].items()}
    write_json(output / "arrivals.json", root_plans)
    result["artifacts"] = {path.name: sha256(path) for path in output.iterdir()}
    write_json(output / "summary.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    for name in ("ledger", "eos-receipt", "source-completions", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
