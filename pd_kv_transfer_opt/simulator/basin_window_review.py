"""Replay frozen Basin windows and bracket the D queue and Store assumptions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import count
from math import floor, isclose
from pathlib import Path
from statistics import mean, median

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.simulator.agent_replay import GB, PAGE, Prefill
from pd_kv_transfer_opt.simulator.store_placement_replay import PlacementReplay, basin_sessions, placement_assumptions
from pd_kv_transfer_opt.simulator.terminal_long_sessions import source_setup


class WindowReplay(PlacementReplay):
    """Model accepted aborts while preserving in-flight P work and transfers."""

    def __init__(self, sessions, offers, service, cap, *, stop_names=(), stop_at=None, **settings):
        self.stopped = set()
        self.cancelled = {}
        self.gates = Counter()
        super().__init__(sessions, offers, service, cap, **settings)
        self.actors = {session.name: session for session in sessions}
        self.gate_by_d = [None, None]
        if stop_at is not None:
            self.defer(stop_at, self.stop_sessions, tuple(stop_names))

    def offer(self, session, step, at):
        if session.name not in self.stopped:
            super().offer(session, step, at)

    def p_arrive(self, job):
        if job.key not in self.cancelled:
            super().p_arrive(job)

    def d_arrive(self, role, job):
        if job.key not in self.cancelled:
            job.row["D_arrive_s"] = self.now
            super().d_arrive(role, job)

    def d_route(self, job):
        if job.row.get("abort_pending"):
            self.retire(job, "P_drained")
        else:
            super().d_route(job)

    def d_start_generation(self, role, job):
        if job.key in self.cancelled:
            return
        if job.row.get("abort_pending"):
            self.d[role].preparing -= 1
            self.retire(job, "handoff_drained")
        else:
            super().d_start_generation(role, job)
            job.row["decode_started"] = True

    def retire(self, job, stage):
        if "EOS_s" in job.row or job.key in self.cancelled:
            raise RuntimeError("Attempted to cancel an already retired request")
        partial = 0
        if job.row.get("decode_started"):
            partial = max(0, min(job.turn.output, floor(job.turn.output - job.remaining + 1e-7)))
        if job.p_lease is not None and "D_KV_ready_s" not in job.row:
            self.p[job.row["P"]].cache.release(job.p_lease)
        if job.d_lease is not None:
            self.d[job.row["D"]].cache.release(job.d_lease)
        if "P" in job.row and "D_first_s" not in job.row:
            self.load_p[job.row["P"]] -= job.p_score
        if "D" in job.row:
            self.load_d[job.row["D"]] -= job.body
        job.p_lease = None
        job.d_lease = None
        self.cancelled[job.key] = (stage, self.now, partial)
        self.done += 1

    def stop_sessions(self, names):
        self.stopped.update(names)
        for name in sorted(names):
            for step in range(len(self.actors[name].turns)):
                key = f"{name}:{step}"
                job = self.requests.get(key)
                if job is None:
                    self.cancelled[key] = ("before_offer", self.now, 0)
                    self.done += 1
                elif "EOS_s" not in job.row:
                    self.abort_active(job)

    def abort_active(self, job):
        job.row["abort_pending"] = True
        if "D" in job.row:
            target = self.d[job.row["D"]]
            if job in target.active:
                target.active.remove(job)
                self.retire(job, "D_active")
            elif "D_KV_ready_s" in job.row:
                target.preparing -= 1
                self.retire(job, "D_before_decode")
            elif "D_alloc_s" not in job.row:
                if job in target.waiting:
                    target.waiting.remove(job)
                self.retire(job, "D_queue_or_ingress")
            # A submitted Pull retires after the existing ready callback.
        elif "P" in job.row and job in self.p[job.row["P"]].waiting:
            self.p[job.row["P"]].waiting.remove(job)
            self.retire(job, "P_queue")
        elif job.p_lease is None:
            self.retire(job, "before_P_arrival")
        # In-flight P work retires at its normal reply boundary.

    def d_tick(self, role):
        super().d_tick(role)
        target = self.d[role]
        if not target.waiting:
            self.gate_by_d[role] = None
        elif len(target.active) >= self.service.slots:
            self.gate_by_d[role] = "active_slots"
        elif target.preparing + len(target.active) >= self.service.slots:
            self.gate_by_d[role] = "slots_including_pending_KV"
        else:
            self.gate_by_d[role] = "pages"

    def advance(self, end, periods):
        elapsed = end - self.now
        for role, gate in enumerate(self.gate_by_d):
            if gate is not None:
                self.gates[gate] += elapsed * len(self.d[role].waiting)
        super().advance(end, periods)


def replay_trial(plan, sessions, private_page, arm, service, cap, settings):
    schedule = plan["schedule"]
    initial = schedule["initial_sessions"]
    warm_sessions = [replace(sessions[name], turns=sessions[name].turns[:info["prime_turns"]])
                     for name, info in initial.items()]
    warm_offers = {name: info["prime_start_s"] for name, info in initial.items()}
    warm = WindowReplay(warm_sessions, warm_offers, service, cap, **settings)
    warm.generated_page_ids = count(private_page)
    warm.run()

    extra = schedule["arms"][arm]
    offers = {**schedule["base_offers_s"], **{name: info["initial_remaining_tool_s"] for name, info in initial.items()},
              **{name: schedule["extra_offers_s"][name] for name in extra}}
    actors = [replace(sessions[name], turns=sessions[name].turns[initial[name]["prime_turns"]:])
              if name in initial else sessions[name] for name in offers]
    first = min(offers.values())
    stop = schedule["terminate_burst_at_s"].get(arm)
    live = WindowReplay(actors, {name: at - first for name, at in offers.items()}, service, cap,
                        stop_names=extra, stop_at=None if stop is None else stop - first, **settings)
    origin = warm.now + first
    for old, new in zip(warm.p + warm.d, live.p + live.d, strict=True):
        new.cache = old.cache
        new.put_free_s = max(0., old.put_free_s - origin)
        if isinstance(new, Prefill):
            new.warm = old.warm
    live.store = warm.store
    live.store.host_evicted = {"P": 0, "D": 0}
    p_evictions_before = [p.cache.evictions for p in live.p]
    live.generated_page_ids = warm.generated_page_ids
    for at, _, handler, args in sorted(warm.events):
        if handler != warm.store.publish:
            raise RuntimeError("Unexpected unfinished priming work")
        live.defer(max(0., at - origin), handler, *args)
    live.run()
    result = summarize(live, set(extra), cap, first)
    result.update(warm_s=warm.now, expected_calls=live.expected_count, priming_calls=warm.expected_count,
                  formal_clock_shift_s=first,
                  budget_s=warm.now + max(schedule["observe_until_s"], first + live.now),
                  P_APC_evicted_pages=sum(p.cache.evictions - before for p, before in zip(live.p, p_evictions_before)))
    if result["expected_calls"] != plan["conditions"][arm]["formal_calls_planned"]:
        raise ValueError("Changed frozen call count")
    return result, live


def summarize(run, extra, cap, first):
    jobs = list(run.requests.values())
    done = [job for job in jobs if "EOS_s" in job.row]
    base = [job for job in done if job.session.name not in extra]
    base_expected = sum(len(session.turns) for name, session in run.actors.items() if name not in extra)
    if len(done) + len(run.cancelled) != run.expected_count or len(base) != base_expected:
        raise ValueError("Calls or common base work were lost")
    for job in done:
        if job.step:
            parent = run.requests[f"{job.session.name}:{job.step - 1}"]
            if not isclose(job.offered, parent.row["EOS_s"] + job.turn.tool_gap_ms / 1000, abs_tol=1e-6):
                raise ValueError("Changed causal successor release")

    remote = sum(job.row.get("D_Host_Store_GET_bytes", 0) for job in jobs) / GB
    port = sum((b - a) * sum(rates) / 8 for a, b, rates in run.tx_epochs)
    if not isclose(remote, port, abs_tol=1e-5):
        raise ValueError("Lost cross-host transfer bytes")
    burst_pressure = sum(max(0., min(b + first, 150.) - max(a + first, 60.))
                         for a, b, rates in run.tx_epochs if max(rates) >= .85 * cap / 8)
    completed_outputs = sum(job.turn.output for job in done)
    partial = sum(row[2] for row in run.cancelled.values())
    base_output = sum(job.turn.output for job in base)
    base_finish = max(job.row["EOS_s"] for job in base)
    ttft = [job.row["D_first_s"] - job.offered for job in base]
    post = [job.row["D_first_s"] - job.offered for job in base if 150 <= job.offered + first < 360]
    return {
        "formal_finish_s": run.now, "completed_calls": len(done), "cancelled_calls": len(run.cancelled),
        "model_completed_output_tokens": completed_outputs, "model_partial_output_tokens": partial,
        "model_throughput_tokens_s": (completed_outputs + partial) / run.now,
        "common_base_output_tokens": base_output, "common_base_finish_s": base_finish,
        "common_base_throughput_tokens_s": base_output / base_finish,
        "common_base_TTFT_mean_s": mean(ttft), "common_base_TTFT_p50_s": median(ttft),
        "common_base_post_surge_calls_150_360": len(post),
        "common_base_post_surge_TTFT_p50_s": median(post) if post else None,
        "common_base_D_queue_mean_s": mean(job.row["D_alloc_s"] - job.row["D_arrive_s"] for job in base),
        "D_queue_all_requests_request_seconds_by_model_gate": dict(run.gates),
        "D_to_P_GET_GB": remote, "burst_any_D_port_near_cap_s": burst_pressure,
        "Store_evicted_GB": sum(run.store.host_evicted.values()) / GB,
        "P_recomputed_after_store_eviction_tokens_proxy": sum(
            job.row.get("P_store_capacity_lost_pages", 0) for job in jobs) * PAGE,
        "abort_stages": dict(Counter(stage for stage, _, _ in run.cancelled.values())),
    }


def with_known_lengths(plan, sessions, known):
    grouped = defaultdict(list)
    for row in plan["requests"]:
        grouped[row["session_id"]].append(row["id"])
    return {name: replace(session, turns=tuple(
        replace(turn, output=known.get(key, turn.output))
        for key, turn in zip(grouped[name], session.turns, strict=True)))
        for name, session in sessions.items()}


def review_abort_boundaries(plan, service, settings, submitted_checks):
    sessions, private_page = basin_sessions(plan)
    initial, reference = replay_trial(plan, sessions, private_page, "burst-drain", service, 100, settings)
    extra = set(plan["schedule"]["arms"]["burst-drain"])
    queued = [job for job in reference.requests.values() if job.session.name in extra
              and job.row["P_dispatch_s"] - job.row["P_admit_s"] > 1e-3]
    if not queued:
        raise ValueError("The reference has no genuine P queue window for the cancellation check")
    sample = min(queued, key=lambda job: job.row["P_admit_s"])
    queue_cut = mean((sample.row["P_admit_s"], sample.row["P_dispatch_s"])) + initial["formal_clock_shift_s"]
    checks = []
    for old in [*submitted_checks, {"deadline_s": queue_cut}]:
        modified = {**plan, "schedule": {**plan["schedule"],
                    "terminate_burst_at_s": {"burst-terminate": old["deadline_s"]}}}
        row, _ = replay_trial(modified, sessions, private_page, "burst-terminate", service, 100, settings)
        if "completed" in old:
            if (row["completed_calls"], row["cancelled_calls"]) != (old["completed"], old["cancelled"]):
                raise ValueError("Submitted abort boundary failed to reproduce")
        elif not row["abort_stages"].get("P_queue"):
            raise ValueError("The added deadline missed a confirmed P waiting request")
        checks.append({"deadline_s": old["deadline_s"], "completed": row["completed_calls"],
                       "cancelled": row["cancelled_calls"], "stages": row["abort_stages"]})
    return checks


def execute(args):
    data = read_json(args.known_eos)["outcomes"]
    known = {key: round(median(n for why, n in rows if why == "stop"))
             for key, rows in data.items() if any(why == "stop" for why, _ in rows)}
    source, models = source_setup(args.results)
    original = read_json(args.submitted)
    reference = {(r["group"], r["arm"], r["cap_Gbps"]): r for r in original["cases"]
                 if (r["seed"], r["service_fit"], r["length_mode"]) == (216, "1", "known-or-source")}
    service = models["1"][0]
    rate = 20.97
    placement = placement_assumptions({"P": 64 * 2**30, "D": 64 * 2**30}, rate)
    default = placement["local_first_serial"]
    variants = {
        "submitted": (service, default),
        "prompt_only_D_capacity_bound": (replace(service, reserve_decode_output=False), default),
        "D_steps_20pct_faster": (replace(service, decode_first_s=service.decode_first_s * .8,
                                         decode_add_s=service.decode_add_s * .8), default),
        "D_steps_20pct_slower": (replace(service, decode_first_s=service.decode_first_s * 1.2,
                                         decode_add_s=service.decode_add_s * 1.2), default),
        "D_pages_25pct_smaller": (replace(service, d_pages=service.d_pages * 3 // 4), default),
        "P_pages_25pct_smaller": (replace(service, p_pages=service.p_pages * 3 // 4), default),
        "local_reads_half_rate": (service, placement["local_first_serial_half_rate"]),
        "host_eviction_without_spill": (service, placement["host_eviction_stress"]),
    }
    conditions = [("light", "steady"), ("light", "plus33"), ("light", "plus100"),
                  ("fivefold", "burst-drain"), ("short", "burst-drain"), ("short", "burst-terminate")]
    cases = []
    for group, arm in conditions:
        path = args.plans / f"{group}.json"
        plan = read_json(path)
        sessions, private_page = basin_sessions(plan)
        sessions = with_known_lengths(plan, sessions, known)
        use = {"submitted": variants["submitted"]} if arm in ("plus33", "plus100") else variants
        for label, (fit, settings) in use.items():
            for cap in (100, 200):
                row, _ = replay_trial(plan, sessions, private_page, arm, fit, cap, settings)
                if label == "submitted":
                    ref = reference[group, arm, cap]
                    for a, b in (("formal_finish_s", "formal_finish_s"), ("common_base_output_tokens", "common_base_output_tokens"),
                                 ("model_throughput_tokens_s", "whole_run_TPS"), ("D_to_P_GET_GB", "D_Host_Store_GET_GB")):
                        if not isclose(row[a], ref[b], abs_tol=1e-6):
                            raise ValueError(f"Submitted result failed to reproduce: {group}/{arm}/{cap}/{a}")
                cases.append(dict(row, group=group, arm=arm, cap_Gbps=cap, scenario=label, plan_sha256=sha256(path)))
        print(f"Reviewed {group}/{arm}", flush=True)
    boundaries = review_abort_boundaries(read_json(args.plans / "short.json"), service, default,
                                         original["abort_stage_checks"])
    write_json(args.output, {"schema": "basin-window-model-audit/v1", "workload_seed": 216,
                            "service_fit": "1", "calibration": source["sources"],
                            "known_eos_sha256": sha256(args.known_eos), "submitted_sha256": sha256(args.submitted),
                            "cases": cases, "abort_boundary_checks": boundaries,
                            "scope": "CPU-only bounds. Prompt-only ignores decode output HBM occupancy while preserving "
                                     "output time and publication. Sensitivities change the model, not the hardware plan."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=Path("configs/basin-windows-v1"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--known-eos", type=Path, required=True)
    parser.add_argument("--submitted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
