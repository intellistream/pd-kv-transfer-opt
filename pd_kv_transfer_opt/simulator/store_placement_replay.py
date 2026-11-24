"""Compare historical and two-host Store placement using archived requests and CPU replay."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import count
from math import ceil, isclose
from pathlib import Path
from statistics import median, median_low

from pd_kv_transfer_opt.common import read_json, sha256, write_json
from pd_kv_transfer_opt.experiments.serve import read_deployment, store_pool_layout
from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union
from pd_kv_transfer_opt.profiling.profile_closed_sessions import NS
from pd_kv_transfer_opt.simulator.agent_cache import Store
from pd_kv_transfer_opt.simulator.agent_replay import (
    ATTENTION_BYTES, GB, PAGE, STATE_BYTES_PER_CHECKPOINT, Prefill, Replay,
)
from pd_kv_transfer_opt.simulator.freeinference_ab import calibration
from pd_kv_transfer_opt.simulator.get_criticality import METHODS, load_method_evidence, scenario
from pd_kv_transfer_opt.simulator.get_overlap_screen import batches
from pd_kv_transfer_opt.simulator.get_port_replay import mac_weights
from pd_kv_transfer_opt.simulator.r69_peer_replay import native_service, peer_tapes, replay_P
from pd_kv_transfer_opt.simulator.rdma_replay import drained_measurement, load_native_archive
from pd_kv_transfer_opt.simulator.terminal_long_sessions import source_setup
from pd_kv_transfer_opt.simulator.terminal_pressure import planned_recovery_start
from pd_kv_transfer_opt.workloads.freeinference import Session, Turn


ROLES = ("P", "D")


class PlacedStore(Store):
    """Keep unique keys in host pools. Prefer the writer host, then use spill capacity."""

    def __init__(self, service, capacity, *, host_eviction):
        if sum(capacity.values()) != service.store_bytes:
            raise ValueError("Placement comparisons require the calibrated aggregate Store capacity")
        super().__init__(service.store_bytes, high_watermark=service.store_high_watermark,
                         eviction_ratio=service.store_eviction_ratio, resident_scale=service.store_resident_scale)
        self.by_host = capacity
        self.host_eviction = host_eviction
        self.host_used = dict.fromkeys(ROLES, 0)
        self.host_peak = self.host_used.copy()
        self.host_evicted = self.host_used.copy()
        self.written_paths = Counter()
        self.watermark_ratio = service.store_high_watermark
        self.eviction_ratio = service.store_eviction_ratio

    def evict_until(self, size, limit, host=None):
        used = self.used if host is None else self.host_used[host]
        if used + size <= limit:
            return
        victims = []
        for key, location in self.objects.items():
            if key not in self.pins and (host is None or location == host):
                victims.append((key, location))
                used -= ceil(self.size(key) * self.resident_scale)
                if used + size <= limit:
                    break
        for key, location in victims:
            del self.objects[key]
            freed = ceil(self.size(key) * self.resident_scale)
            self.used -= freed
            self.host_used[location] -= freed
            self.host_evicted[location] += freed
            self.eviction_bytes += freed
            self.evicted += 1

    def publish(self, keys, writer):
        alternate = "D" if writer == "P" else "P"
        choices = (writer,) if self.host_eviction else (writer, alternate)
        for key in keys:
            if key in self.objects:
                self.objects.move_to_end(key)
                continue
            payload = self.size(key)
            size = ceil(payload * self.resident_scale)
            self.ever_published.add(key)
            if self.host_eviction:
                cap = self.by_host[writer]
                high = int(cap * self.watermark_ratio)
                if self.host_used[writer] + size > high:
                    self.evict_until(size, max(size, high - int(cap * self.eviction_ratio)), writer)
            elif self.used + size > self.high_watermark:
                self.evict_until(size, max(size, self.high_watermark - self.eviction_batch))
            dest = next((host for host in choices if self.host_used[host] + size <= self.by_host[host]), None)
            if dest is None:
                self.failed += 1
                continue
            self.objects[key] = dest
            self.used += size
            self.written += size
            self.host_used[dest] += size
            self.peak = max(self.peak, self.used)
            self.host_peak[dest] = max(self.host_peak[dest], self.host_used[dest])
            self.written_paths[f"{writer}_to_{dest}"] += payload


class PlacementReplay(Replay):
    def __init__(self, sessions, offers, service, cap, *, capacity, local_rate, overlap=False, host_eviction=False):
        super().__init__(sessions, offers, service, cap)
        self.store = PlacedStore(service, capacity, host_eviction=host_eviction)
        self.local_rate = local_rate
        self.overlap = overlap
        self.local_free_s = 0.
        self.local_costs = {}

    def put_from(self, target, keys):
        if not keys:
            return 0.
        writer = "P" if isinstance(target, Prefill) else "D"
        new_GB = sum(self.store.size(key) for key in keys if key not in self.store.objects) / GB
        target.put_free_s = (max(target.put_free_s, self.now) + self.service.put_setup_s
                             + new_GB / self.service.put_GB_s_per_P)
        self.defer(target.put_free_s, self.store.publish, keys, writer)
        return new_GB

    def queue_local(self, volume):
        self.local_free_s = max(self.local_free_s, self.now) + volume / self.local_rate
        return self.local_free_s

    def get_start(self, role, loads):
        super().get_start(role, loads)
        local, remote = 0, 0
        store = self.store.objects
        for job, attention, states, _ in loads:
            keep = (sum(store[key] == "P" for key in attention) * ATTENTION_BYTES
                    + sum(store[-key] == "P" for key in states) * STATE_BYTES_PER_CHECKPOINT)
            job.row["P_Host_Store_GET_bytes"] = keep
            job.row["D_Host_Store_GET_bytes"] = job.row["GET_bytes"] - keep
            local += keep
            remote += job.row["D_Host_Store_GET_bytes"]
        self.local_costs[role] = (local / GB, self.queue_local(local / GB) if local and self.overlap else None)
        self.gets[role].remaining_GB = remote / GB
        if not remote:
            self.get_finish(role)

    def get_finish(self, role):
        batch = self.gets.pop(role)
        local, ready = self.local_costs.pop(role)
        if local and ready is None:
            ready = self.queue_local(local)
        if ready is not None and ready > self.now:
            self.p[role].stage = "get_local"
            self.defer(ready, self.complete_get, role, batch)
        else:
            self.complete_get(role, batch)


def state_metrics(run):
    rows = list(run.requests.values())
    tail = max(rows, key=lambda job: job.row["EOS_s"])
    chain = [job for job in rows if job.session.name == tail.session.name]
    output = sum(job.turn.output for job in rows)
    first_offer = min(job.offered for job in rows)
    local = sum(job.row.get("P_Host_Store_GET_bytes", 0) for job in rows)
    original_get = sum(job.row["GET_bytes"] for job in rows)
    remote = original_get - local
    per_port = [sum((b - a) * rates[i] / 8 for a, b, rates in run.tx_epochs) for i in range(8)]
    if not isclose(sum(per_port), remote / GB, abs_tol=1e-5):
        raise ValueError("Cross-host GET payload did not match the modeled D ports")
    get_intervals = [[(job.row["GET_submit_s"], job.row["GET_done_s"]) for job in rows
                      if job.row["P"] == role and job.row["GET_bytes"]] for role in range(2)]
    return {"requests": len(rows), "output_tokens": output, "finish_s": run.now - first_offer,
            "throughput_tokens_s": output / (run.now - first_offer), "tail_id": tail.key,
            "tail_session_turns": len(chain),
            "tail_session_D_after_first_token_s": sum(job.row["EOS_s"] - job.row["D_first_s"] for job in chain),
            "P_HBM_APC_tokens": sum(job.row["P_local_pages"] for job in rows) * PAGE,
            "P_computed_tokens": sum(job.row["P_computed_tokens"] for job in rows),
            "D_HBM_APC_tokens": sum(job.row["D_local_pages"] for job in rows) * PAGE,
            "P_to_D_Pull_GB": sum(job.row["P_to_D_bytes"] for job in rows) / GB,
            "P_Host_Store_GET_GB": local / GB, "D_Host_Store_GET_GB": remote / GB,
            "D_GET_GB_by_port": per_port,
            "P_in_GET_wall_s_by_instance": [interval_union(rows) for rows in get_intervals],
            "Store_peak_GiB_by_host": {role: count / 2**30 for role, count in run.store.host_peak.items()},
            "Store_peak_total_GiB": run.store.peak / 2**30,
            "Store_evicted_GB_by_host": {role: count / GB for role, count in run.store.host_evicted.items()},
            "Store_put_GB_by_path": {key: count / GB for key, count in run.store.written_paths.items()},
            "Store_failed_puts": run.store.failed,
            "D_no_active_decode_with_P_pending_s": sum(b - a for a, b, width, pending, *_ in run.supply_epochs
                                                        if not sum(width) and sum(pending))}


def placement_assumptions(capacity, rate):
    return {
        "old_D_only": {"capacity": {"P": 0, "D": sum(capacity.values())}, "local_rate": rate},
        "local_first_serial": {"capacity": capacity, "local_rate": rate},
        "local_first_parallel": {"capacity": capacity, "local_rate": rate, "overlap": True},
        "local_first_serial_half_rate": {"capacity": capacity, "local_rate": rate / 2},
        "host_eviction_stress": {"capacity": capacity, "local_rate": rate, "host_eviction": True},
    }


def paired_states(models, calibration_summary, assumptions):
    blocks = {}
    for block, (service, sessions, offers) in models.items():
        order = (100, 200) if block == "1" else (200, 100)
        cases = {label: {} for label in assumptions}
        for cap in order:
            for label, settings in assumptions.items():
                run = PlacementReplay(sessions, offers, service, cap, **settings).run()
                row = state_metrics(run)
                if label == "old_D_only":
                    original = calibration_summary["results"][block]["original"][str(cap)]
                    for actual, expected in ((row["finish_s"], original["finish_s"]),
                                             (row["D_Host_Store_GET_GB"], original["GET_GB"])):
                        if not isclose(actual, expected, abs_tol=1e-5):
                            raise ValueError("D-only control failed to reproduce the legacy CPU replay")
                else:
                    baseline = cases["old_D_only"][cap]
                    row["gain_vs_D_only_pct"] = 100 * (row["throughput_tokens_s"] / baseline["throughput_tokens_s"] - 1)
                cases[label][cap] = row
        for case in cases.values():
            case["cap200_gain_over_cap100_pct"] = 100 * (case[200]["throughput_tokens_s"] /
                                                        case[100]["throughput_tokens_s"] - 1)
        blocks[block] = cases
        print(f"r43 block {block}: original arrivals and fixed observed EOS work replayed at both caps", flush=True)
    return blocks


def basin_sessions(plan):
    """Rebuild the exact adjacent LCP lengths and isolate distinct source tasks."""
    if plan["schema"] != "basin-small-replay/v1" or plan["ignore_eos"]:
        raise ValueError("Expected the frozen Basin validation workload")
    grouped = defaultdict(list)
    next_page = 1
    for row in plan["requests"]:
        task = row["session_id"]
        previous = grouped[task][-1][1].pages if grouped[task] else ()
        expected_parent = grouped[task][-1][0] if grouped[task] else None
        if row["predecessor"] != expected_parent:
            raise ValueError("Basin replay requires the complete original causal chain")
        pages = row["prompt_tokens"] // PAGE
        shared = row["source_lcp_tokens"] // PAGE
        if not 0 <= shared <= min(pages, len(previous)):
            raise ValueError("Basin source LCP goes beyond its prior or current prompt")
        keys = (*previous[:shared], *range(next_page, next_page + pages - shared))
        next_page += pages - shared
        grouped[task].append((row["id"], Turn(round(row["tool_gap_s"] * 1000), row["prompt_tokens"],
                                             row["source_completion_tokens"], keys, shared, "source_output_proxy")))
    return {name: Session(name, 0, 0, 0, tuple(turn for _, turn in rows)) for name, rows in grouped.items()}, next_page


def warm_basin_trial(sessions, first_private_page, schedule, arm, service, cap, settings):
    initial = schedule["initial_sessions"]
    priming = [replace(sessions[name], turns=sessions[name].turns[:spec["prime_turns"]])
               for name, spec in initial.items()]
    warm = PlacementReplay(priming, {name: item["prime_start_s"] for name, item in initial.items()},
                           service, cap, **settings)
    if service.publish_decode_kv:
        warm.generated_page_ids = count(first_private_page)
    warm.run()

    offers = {**schedule["base_offers_s"], **{name: item["initial_remaining_tool_s"]
                                             for name, item in initial.items()},
              **{name: schedule["extra_offers_s"][name] for name in schedule["arms"][arm]}}
    after_warmup = min(offers.values())
    actors = [replace(sessions[name], turns=sessions[name].turns[initial[name]["prime_turns"]:])
              if name in initial else sessions[name] for name in offers]
    live = PlacementReplay(actors, {name: at - after_warmup for name, at in offers.items()}, service, cap, **settings)
    origin = warm.now + after_warmup
    for old, new in zip(warm.p + warm.d, live.p + live.d, strict=True):
        new.cache = old.cache
        new.put_free_s = max(0., old.put_free_s - origin)
        if isinstance(new, Prefill):
            new.warm = old.warm
    live.store = warm.store
    live.generated_page_ids = warm.generated_page_ids
    live.store.written_paths.clear()
    live.store.host_evicted = dict.fromkeys(ROLES, 0)
    live.store.failed = 0
    live.store.host_peak = live.store.host_used.copy()
    live.store.peak = live.store.used
    for at, _, handler, args in sorted(warm.events):
        if handler != warm.store.publish:
            raise ValueError("A Basin priming request left pending work other than a Store publication")
        live.defer(max(0., at - origin), handler, *args)
    data = state_metrics(live.run())
    data["priming_calls"] = warm.expected_count
    data["priming_s"] = warm.now
    return data


def basin_sensitivity(plan, models, assumptions):
    sessions, private_page = basin_sessions(plan)
    result = {}
    for block, (service, _, _) in models.items():
        order = (100, 200) if block == "1" else (200, 100)
        result[block] = {}
        for arm in ("steady", "plus100"):
            cases = {}
            for label in ("old_D_only", "local_first_serial", "local_first_parallel", "host_eviction_stress"):
                cases[label] = {}
                for cap in order:
                    row = warm_basin_trial(sessions, private_page, plan["schedule"], arm, service, cap, assumptions[label])
                    if (row["requests"], row["priming_calls"]) != ({"steady": 592, "plus100": 689}[arm], 112):
                        raise ValueError("Basin replay changed the frozen cohort")
                    if label != "old_D_only":
                        base = cases["old_D_only"][cap]
                        row["gain_vs_D_only_pct"] = 100 * (row["throughput_tokens_s"] / base["throughput_tokens_s"] - 1)
                    cases[label][cap] = row
            result[block][arm] = cases
            print(f"Basin {arm} block {block}: replayed unchanged priming, original clocks and source output proxy", flush=True)
    return result


def trace_sensitivity(results, per_rank_rate):
    folder = results / METHODS
    manifest, digest, arms = load_native_archive(folder)
    events, receipts, methods_sha = load_method_evidence(folder, manifest)
    source = read_json(results / "2026-11-23-r69-peer-source-screen/summary.json")
    if (source["source_hashes"]["request"], source["source_hashes"]["method"]) != (digest, methods_sha):
        raise ValueError("The GET tape index belongs to a different trace")
    fits = {name: row["decode_fit"] for name, (_, row) in calibration(results).items()}
    out = {}
    for case, (raw, requests, _) in arms.items():
        cap = int(case[-3:])
        p_events, chunks, loads, joined = batches(events[case], requests, receipts[case]["P_GET_descriptor_bytes"])
        flows, tapes, clock, fixed = peer_tapes(chunks, loads, joined, source["cases"][case]["candidates_by_batch"])
        anchor, = (row["anchor"] for row in raw["client"] if row["type"] == "client_origin")
        shift_ns = (median_low(row["unix_ns"] - row["monotonic_ns"] for row in p_events)
                    - anchor["unix_ns"] + anchor["monotonic_ns"])
        weights = mac_weights(raw["telemetry"], cap, shift_ns, clock, flows, "pooled_MAC")["P0"]
        profile, _ = native_service(flows, cap, weights)
        since = planned_recovery_start(requests)
        baseline = scenario(requests, {}, fits, since)
        control = drained_measurement(requests)
        if not isclose(control["duration_s"], baseline["fixed_observed_downstream"]["duration_s"], abs_tol=1e-6):
            raise ValueError("The no-change trace did not reconstruct the measured drain")
        volume = sum(row["bytes"] for row in flows.values())
        case_out = {"cap_Gbps": cap, "historical_measured": control, "historical_replayed": baseline,
                    "archived_D_to_P_GET_GB": volume / GB, "scenarios": {}}
        for fraction in (0., .25, .50, .75):
            moved = {key: round(row["bytes"] * fraction) if key[0] not in fixed else 0 for key, row in flows.items()}
            rates = [("serial", "serial", per_rank_rate), ("parallel", "parallel", per_rank_rate)]
            if fraction == .50:
                rates.append(("serial_half_rate", "serial", per_rank_rate / 2))
            for label, order, rate in rates:
                advance, _, wire = replay_P(chunks, flows, tapes, clock, fixed, cap, profile, moved, local_rate=rate,
                                            placement=order, hold_dispatch=False, feedback=False)
                points = scenario(requests, advance, fits, since)
                for name, row in points.items():
                    row["gain_vs_D_only_pct"] = 100 * (row["throughput_tokens_s"] / baseline[name]["throughput_tokens_s"] - 1)
                case_out["scenarios"][f"{fraction:.2f}_{label}"] = {
                    "nominal_local_fraction_of_eligible_GET": fraction, "modeled_bytes_moved_to_P_GB": sum(moved.values()) / GB,
                    "local_GB_s_per_rank": rate, "remaining_D_to_P_GET_GB": sum(wire["wire_bytes_per_port"]) / GB,
                    "downstream": points}
        out[case] = case_out
        print(f"r69 {case}: unchanged work and explicit local-read scenarios replayed", flush=True)
    return {"request_trace_sha256": digest, "method_trace_sha256": methods_sha, "arms": out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--deployment", type=Path, default=Path("configs/ascend-2p2d-decode-apc.example.json"))
    parser.add_argument("--local-probe", type=Path, required=True)
    parser.add_argument("--basin-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layout = store_pool_layout(read_deployment(args.deployment))
    probe = read_json(args.local_probe)
    rates = {role: median(row["bytes"] / row["elapsed_ms"] / 10**6 for row in probe["samples"]
                          if row["layout"] == "near-223k" and row["arm"] == role)
             for role in ("local", "remote")}
    source, models = source_setup(args.results)
    assumptions = placement_assumptions(layout["capacity_bytes_by_role"], rates["local"])
    state = paired_states(models, source, assumptions)
    traced = trace_sensitivity(args.results, rates["local"])
    basin = {"plan_sha256": sha256(args.basin_plan),
             "output_lengths": "Observed outputs of the source provider, as a fixed proxy for unavailable Qwen EOS",
             "prompt_keys": "Exact archived adjacent LCP length within a task, with distinct task namespaces",
             "blocks": basin_sensitivity(read_json(args.basin_plan), models, assumptions)} if args.basin_plan else None
    write_json(args.output, {"target_layout": layout,
                             "source_probe": {"sha256": sha256(args.local_probe), "single_rank_GB_s": rates,
                                              "purpose": "Bounded transfer-rate anchor before multi-rank contention"},
                             "r43_calibration_sources": source["sources"],
                             "placement_assumptions": assumptions,
                             "scope": {"r43": "Fixed archived EOS work and original offers. Native simulated routing and APC "
                                              "can change with timing. New immutable Store keys prefer the first writer host. "
                                              "Global LRU allows spill and host eviction is a lower-capacity stress case. "
                                              "PUT rate, decode and P-to-D service calibration are held.",
                                       "r69": "Fixed archived routes, worker tape and EOS. Uniform fractions of eligible "
                                              "read payload are moved to P. Source residency is intentionally a scenario."},
                             "r43_state_replay": state, "r69_trace_sensitivity": traced, "basin_proxy_replay": basin})


if __name__ == "__main__":
    main()
