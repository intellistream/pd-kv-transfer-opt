"""Construct a native APC capacity-pressure replay from audited Terminal-Bench snapshots."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
import gzip
from itertools import product
from math import ceil, isclose
from pathlib import Path
from statistics import median_low

from pd_kv_transfer_opt.common import percentile, read_json, sha256, write_json
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BYTES
from pd_kv_transfer_opt.simulator.agent_cache import STATE_BYTES_PER_CHECKPOINT, STORE_ATTENTION_BYTES
from pd_kv_transfer_opt.simulator.agent_replay import GB, PAGE, Replay
from pd_kv_transfer_opt.simulator.agent_trace_gap import read_bytes
from pd_kv_transfer_opt.simulator.freeinference_ab import PUBLISHED_BASELINE, calibration, report, supply_pair, trace
from pd_kv_transfer_opt.simulator.terminal_prefill_calibration import clock_profiles
from pd_kv_transfer_opt.simulator.terminal_pressure import peak_window
from pd_kv_transfer_opt.simulator.terminal_supply import load_source
from pd_kv_transfer_opt.workloads.fetch_trajectories import REPO, REVISION
from pd_kv_transfer_opt.workloads.freeinference import Session, Turn


@dataclass(frozen=True)
class Source:
    key: str
    task: str
    prompt: int
    output: int
    body_bytes: int
    prompt_sha256: str
    eos_reference_count: int

    @property
    def pages(self):
        return self.prompt // PAGE


def sources(ledger, outcomes, sizes):
    eligible = defaultdict(list)
    for key, rows in outcomes.items():
        row = ledger[key]
        n = row["prompt_tokens"]
        if (key not in sizes or row["truncated"] or not 96 * PAGE <= n <= 165 * PAGE or len(rows) < 4
                or any(reason != "stop" for reason, _ in rows)):
            continue
        counts = sorted(count for _, count in rows)
        q = median_low(counts)
        # The retained output fits in the partial final page. The replay does
        # not need unknown generated block hashes to keep the shared set sized.
        if n % PAGE == 0 or n % PAGE + q >= PAGE or percentile(counts, .7) > 220:
            continue
        eligible[row["task"]].append(Source(key, row["task"], n, q, sizes[key],
                                            row["token_ids_uint32le_sha256"], len(counts)))
    return [min(group, key=lambda s: (abs(s.pages - 128), s.output, s.key))
            for _, group in sorted(eligible.items())]


def choose_domains(base, service):
    max_prompt_pages = max(s.pages for s in base)
    low = service.p_pages + max_prompt_pages + 8
    store_page_bytes = STORE_ATTENTION_BYTES + STATE_BYTES_PER_CHECKPOINT
    high = service.store_bytes // store_page_bytes
    wanted = low + min(12, (high - low) // 2)
    choices = []
    for counts in product(range(2, 7), repeat=len(base)):
        pages = sum(s.pages * n for s, n in zip(base, counts, strict=True))
        if low <= pages <= high:
            choices.append((max(counts) - min(counts), abs(pages - wanted), counts))
    if not choices:
        raise ValueError("Archived Terminal-Bench snapshots did not fit the capacity interval")
    _, _, copies = min(choices)
    domains = [source for step in range(max(copies))
               for source, count in zip(base, copies, strict=True) if step < count]
    return domains


def fitting_aliases(domains, budget):
    original, aliases = {}, {}
    pages = sum(source.pages for source in domains)
    for i, source in enumerate(domains):
        own = original.setdefault(source.key, i)
        aliases[i] = i
        if pages > budget and own != i:
            aliases[i] = own
            pages -= source.pages
    if pages > budget:
        raise ValueError("Same-prompt aliases cannot provide the HBM fit control")
    return aliases


def workload(domains, *, rounds, pressure_start, read_offer_GB_s, preparation_copies=2, aliases=None):
    offsets, body, sessions, rows = {}, {}, [], []
    keys, turns, next_key = {}, {}, 1
    for i, source in enumerate(domains):
        domain = aliases[i] if aliases else i
        if domain not in keys:
            keys[domain] = tuple(range(next_key, next_key + source.pages))
            next_key += source.pages
        turns[i] = Turn(0, source.prompt, source.output, keys[domain], source.pages, "stop")

    def offer(phase, cycle, index, replica, at):
        name = f"{phase}-{cycle:03}-{index:02}-{replica}"
        sessions.append(Session(name, len(sessions), 0, 0, (turns[index],)))
        offsets[name] = at
        body[f"{name}:0"] = domains[index].body_bytes
        rows.append({"name": name, "phase": phase, "cycle": cycle, "slot": index, "replica": replica,
                     "root_offered_s": at, "source": domains[index].key, "output_tokens": domains[index].output,
                     "cache_domain": f"terminal-pressure-{index if aliases is None else aliases[index]:02}"})

    cold = sorted(range(len(domains)), key=lambda i: (domains[i].body_bytes, i))
    for j, i in enumerate(cold):
        for replica in range(preparation_copies):
            offer("prepare", 0, i, replica, .012 * j)
    cursor = pressure_start
    for cycle in range(rounds):
        for i, source in enumerate(domains):
            for replica in range(2):
                offer("restore", cycle, i, replica, cursor)
            cursor += 2 * read_bytes(source.pages) / GB / read_offer_GB_s
    return sessions, offsets, body, rows


class PairedClients(Replay):
    """A client posts the next pair after both first tokens from the previous pair."""

    def __init__(self, plan, service, cap, residuals):
        sessions, offers, sizes, rows = plan
        prepare = [s for s, row in zip(sessions, rows, strict=True) if row["phase"] == "prepare"]
        super().__init__(prepare, {s.name: offers[s.name] for s in prepare}, service, cap,
                         observed_body_bytes=sizes, ingress_residual_s=residuals)
        copies = sessions[len(prepare):]
        self.pairs = deque(zip(copies[::2], copies[1::2], strict=True))
        self.minimum_clock = offers
        self.pending_first = set()
        self.next_opportunity = None
        self.expected_count = len(sessions)
        if self.pairs:
            self.defer(offers[self.pairs[0][0].name], self.post_pair)

    def post_pair(self):
        first, second = self.pairs.popleft()
        self.pending_first.update((first.name, second.name))
        self.offer(first, 0, self.now)
        self.offer(second, 0, self.now)
        if self.pairs:
            next_time = self.minimum_clock[self.pairs[0][0].name]
            self.next_opportunity = self.now + next_time - self.minimum_clock[first.name]

    def on_first_decode_token(self, job):
        if job.session.name in self.pending_first:
            self.pending_first.remove(job.session.name)
            if not self.pending_first and self.pairs:
                self.defer(max(self.now, self.next_opportunity), self.post_pair)


def audit_run(run, rows, pressure_start):
    result = report(run)
    reads = Counter()
    placements = defaultdict(set)
    phases = defaultdict(list)
    for row in rows:
        job = run.requests[f"{row['name']}:0"].row
        phases[row["phase"]].append(job)
        if row["phase"] == "restore":
            reads[row["cycle"]] += job["GET_bytes"] / GB
            placements[(row["cycle"], row["slot"])].add(job["P"])
    pressure_requests = phases["restore"]
    if any(job["EOS_s"] > pressure_start for job in phases["prepare"]):
        raise RuntimeError("The pressure phase began before a preparation request finished")
    interval = max(row["P_reply_s"] for row in pressure_requests) - pressure_start
    span_end = pressure_start + interval
    clocks = [(max(a, pressure_start), b, rates) for a, b, rates in run.tx_epochs if b > pressure_start]
    busy = sum(b - a for a, b, rates in clocks if any(rates))
    stages = run.supply_epochs
    active_area = sum(max(0, min(b, span_end) - max(a, pressure_start)) * sum(width)
                      for a, b, width, *_ in stages)
    queue = defaultdict(list)
    for row, job in zip((v for v in rows if v["phase"] == "restore"), pressure_requests, strict=True):
        queue[row["cycle"] * 10 // len(reads)].append(job["P_dispatch_s"] - job["P_admit_s"])
    result["pressure"] = {"start_s": pressure_start, "last_P_reply_minus_start_s": interval,
                          "GET_wire_busy_s": busy, "GET_wire_fraction_of_pressure_phase": busy / interval,
                          "D_to_P_peak_30s_Gbps": peak_window(clocks, 30),
                          "mean_D_running_sequences": active_area / interval,
                          "GET_GB_by_cycle": [reads[i] for i in sorted(reads)],
                          "both_P_admission_pairs": sum(len(roles) == 2 for roles in placements.values()),
                          "total_pairs": len(placements),
                          "P_queue_by_tenth_p95_s": [percentile(queue[i], .95) for i in sorted(queue)],
                          "P_queue_p95_s": percentile(
                              [row["P_dispatch_s"] - row["P_admit_s"] for row in pressure_requests], .95)}
    return result


def run_pair(plan, service, order, pressure_start, *, residuals=None, output=None, name="", paired_clients=False):
    sessions, offers, sizes, rows = plan
    runs, arms = {}, {}
    for cap in (100, 200) if order == "forward" else (200, 100):
        model = (PairedClients(plan, service, cap, residuals) if paired_clients else
                 Replay(sessions, offers, service, cap, observed_body_bytes=sizes,
                        ingress_residual_s=residuals))
        replay = model.run()
        arms[cap] = audit_run(replay, rows, pressure_start)
        runs[cap] = replay
        trace(output, name, order, cap, replay)
    if arms[100]["output_tokens"] != arms[200]["output_tokens"] or runs[100].requests.keys() != runs[200].requests.keys():
        raise RuntimeError("A link-cap intervention changed the frozen EOS reference")
    low, high = arms[100], arms[200]
    wire_gap = low["pressure"]["GET_wire_busy_s"] - high["pressure"]["GET_wire_busy_s"]
    gap = runs[100].now - runs[200].now
    result = {"arms": arms, "100_loss_relative_to_200": gap / runs[100].now,
              "same_D_to_P_bytes_on_every_port": all(
                  isclose(a, b, abs_tol=1e-6) for a, b in zip(low["D_to_P_GB_by_port"], high["D_to_P_GB_by_port"], strict=True)),
              "same_P_placement_requests": sum(job.row["P"] == runs[200].requests[key].row["P"]
                                               for key, job in runs[100].requests.items()),
              "D_supply_comparison": supply_pair(runs[100], runs[200], start_s=pressure_start),
              "saved_completion_s": gap, "saved_wire_service_s": wire_gap,
              "remaining_completion_gap_s": gap - wire_gap}
    if paired_clients:
        result["same_added_nonwire_ms_per_pair_budget_for_30pct_loss"] = (
            1000 * max(0, gap / .30 - runs[100].now) / low["pressure"]["total_pairs"])
    return result


def initial_sizing(domains, models, *, preparation_copies=2):
    rates = [100 * service.rail_efficiency / (64 * max(service.shares))
             for service, _ in models.values()]
    offer = .89 * 2 * min(rates)
    cold_plan, starts, sizes, _ = workload(domains, rounds=0, pressure_start=0,
                                           read_offer_GB_s=offer, preparation_copies=preparation_copies)
    cold_end = {name: Replay(cold_plan, starts, service, 100, observed_body_bytes=sizes).run().now
                for name, (service, _) in models.items()}
    pressure_start = ceil(max(cold_end.values()) + 2)
    wave_loss = 1 - max(rates) / offer
    per_cycle = 2 * sum(read_bytes(s.pages) for s in domains) / GB
    # Allow two initial cycles to populate or turn over the destination caches.
    ideal_work = .30 * (pressure_start + 5) * max(rates) / (wave_loss - .30)
    rounds = ceil(ideal_work / per_cycle) + 2
    return {"effective_100_GB_s": rates, "target_offered_GB_s_if_both_P_need_GET": offer,
            "cold_end_s": cold_end, "pressure_start_s": pressure_start, "ideal_pressure_wave_100_loss": wave_loss,
            "full_network_work_GB_per_cycle": per_cycle, "estimated_work_for_30pct_GB": ideal_work,
            "cycles": rounds}


def input_residuals(plan, results):
    m = read_json(results / "2026-11-22-r78-terminal-source-restore-stable-tail" / "manifest.json")
    fit = m["workload"]["arrival_order_screen"]["admission_fit"]
    profiles = clock_profiles(results, fit)
    _, _, _, rows = plan
    chosen, _ = max(profiles.items(), key=lambda item: max(item[1][1]) - min(item[1][1]))
    root, later = profiles[chosen]
    offsets = {}
    for i, row in enumerate(rows):
        residuals = root if row["phase"] == "prepare" else later
        offsets[f"{row['name']}:0"] = residuals[i % len(residuals)]
    return chosen, offsets


def paired_sizing(domains, models, offer):
    cold, starts, sizes, _ = workload(domains, rounds=0, pressure_start=0, read_offer_GB_s=offer,
                                     preparation_copies=1)
    root_ends = {name: Replay(cold, starts, service, 100, observed_body_bytes=sizes).run().now
                 for name, (service, _) in models.items()}
    start = ceil(max(root_ends.values()) + 2)
    budgets = {}
    for name, (s, _) in models.items():
        rate = 100 * s.rail_efficiency / (64 * max(s.shares))
        periods = {100: 0., 200: 0.}
        for source in domains:
            bytes_ = 2 * read_bytes(source.pages) / GB
            c = (s.ingress_base_s + source.body_bytes * s.ingress_s_per_byte
                 + s.get_setup_s + (source.prompt % PAGE) / s.warm_tokens_s
                 + s.decode_ingress_s + s.pull_setup_s + ceil(source.prompt / PAGE) * ATTENTION_BYTES / GB / s.pull_GB_s
                 + s.decode_first_s + (s.slots - 1) * s.decode_add_s)
            for cap in periods:
                min_offer_interval = bytes_ / offer
                periods[cap] += max(min_offer_interval, c + bytes_ / (cap / 100 * rate))
        exposure = .70 * periods[100] - periods[200]
        if exposure <= 0:
            raise ValueError("The calculated client-side pacing window cannot produce the target loss")
        budgets[name] = {"cycle_seconds": periods, "minimum_cycles": .30 * (start + 5) / exposure}
    return {"pressure_start_s": start, "cold_end_s": root_ends, "predicted": budgets,
            "cycles": ceil(max(v["minimum_cycles"] for v in budgets.values())) + 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ledger", "eos-receipt", "source-completions", "output", "trace"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    ledger, _, outcomes, sizes = load_source(args.ledger, args.eos_receipt, args.source_completions,
                                             REVISION, args.results)
    models = {name: (replace(service, release_suffix_first=True), meta)
              for name, (service, meta) in calibration(args.results).items()}
    service = models["forward"][0]
    selected = choose_domains(sources(ledger, outcomes, sizes), service)
    design = initial_sizing(selected, models)
    plan = workload(selected, rounds=design["cycles"], pressure_start=design["pressure_start_s"],
                    read_offer_GB_s=design["target_offered_GB_s_if_both_P_need_GET"])
    jit_case, ingress = input_residuals(plan, args.results)
    controls = fitting_aliases(selected, service.p_pages - 2 * max(s.pages for s in selected))
    ctrl_plan = workload(selected, rounds=design["cycles"], pressure_start=design["pressure_start_s"],
                         read_offer_GB_s=design["target_offered_GB_s_if_both_P_need_GET"], aliases=controls)
    offer = design["target_offered_GB_s_if_both_P_need_GET"]
    client_design = paired_sizing(selected, models, offer)
    client_options = {"rounds": client_design["cycles"], "pressure_start": client_design["pressure_start_s"],
                      "read_offer_GB_s": offer, "preparation_copies": 1}
    client_plan = workload(selected, **client_options)
    client_control = workload(selected, **client_options, aliases=controls)
    _, client_ingress = input_residuals(client_plan, args.results)
    out = {"source_repo": REPO, "source_revision": REVISION, "ledger_sha256": sha256(args.ledger),
           "eos_receipt_sha256": sha256(args.eos_receipt), "baseline_revision": PUBLISHED_BASELINE,
           "scope": "CPU constructed workload using exact Terminal-Bench snapshot fingerprints and observed natural EOS "
           "lengths. Fixed per-domain cache salts separate independent copies of unchanged prompts. Each request "
           "is an independent replay of its complete source prompt history. Cold preparation is timed.",
           "design": {**design, "num_domains": len(selected), "working_set_attention_pages": sum(s.pages for s in selected),
                      "P_capacity_pages": service.p_pages, "Store_capacity_GiB": service.store_bytes / 2**30,
                      "Store_dense_fit_GB": sum(s.pages for s in selected) * (
                          STORE_ATTENTION_BYTES + STATE_BYTES_PER_CHECKPOINT) / GB,
                      "ordered_sources": [asdict(s) for s in selected], "requests": len(plan[3]),
                      "fit_control_aliases": controls},
           "models": {name: {"service": asdict(s), **meta} for name, (s, meta) in models.items()},
           "nominal": {}, "archive_ingress_sensitivity": {"source": jit_case, "orders": {}},
           "paired_clients": {"design": client_design, "nominal": {}, "archive_ingress": {},
                              "same_requests_without_first_token_gate": {},
                              "rule": "Send two simultaneous requests for the same namespace and snapshot. "
                                      "Wait for both D first tokens and the minimum pair interval before posting "
                                      "the next pair. The plan and EOS work are frozen. Actual releases follow "
                                      "first-token completions. Routing uses unchanged native scores."}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output.with_name("workload.json"),
               {"open_loop": plan[3], "paired_clients_earliest_offer_if_unblocked": client_plan[3],
                "design": out["design"], "paired_client_design": client_design})
    with gzip.open(args.trace, "wt", encoding="utf-8", compresslevel=4) as file:
        for order, (s, _) in models.items():
            out["nominal"][order] = run_pair(plan, s, order, design["pressure_start_s"], output=file, name="nominal")
            print("nominal", order, out["nominal"][order]["100_loss_relative_to_200"], flush=True)
            out["archive_ingress_sensitivity"]["orders"][order] = run_pair(
                plan, s, order, design["pressure_start_s"], residuals=ingress, output=file, name="r78_ingress")
            print("ingress", order, out["archive_ingress_sensitivity"]["orders"][order][
                "100_loss_relative_to_200"], flush=True)
        out["fit_control"] = run_pair(ctrl_plan, service, "forward", design["pressure_start_s"],
                                      output=file, name="local_fit")
        for order, (s, _) in models.items():
            for label, residual in (("nominal", None), ("archive_ingress", client_ingress)):
                result = run_pair(client_plan, s, order, client_design["pressure_start_s"], residuals=residual,
                                  output=file, name=f"paired_{label}", paired_clients=True)
                out["paired_clients"][label][order] = result
                print("paired", label, order, result["100_loss_relative_to_200"], flush=True)
            out["paired_clients"]["same_requests_without_first_token_gate"][order] = run_pair(
                client_plan, s, order, client_design["pressure_start_s"],
                output=file, name="same_plan_open_loop")
        out["paired_clients"]["fit_control"] = run_pair(
            client_control, service, "forward", client_design["pressure_start_s"], residuals=client_ingress,
            output=file, name="paired_local_fit", paired_clients=True)
    repeated = run_pair(client_plan, service, "reverse", client_design["pressure_start_s"], paired_clients=True)
    out["paired_clients"]["same_service_ABBA_reproduced"] = all(
        repeated["arms"][cap]["finish_s"] == out["paired_clients"]["nominal"]["forward"]["arms"][cap]["finish_s"]
        for cap in (100, 200))
    out["workload_sha256"] = sha256(args.output.with_name("workload.json"))
    out["trace_sha256"] = sha256(args.trace)
    write_json(args.output, out)


if __name__ == "__main__":
    main()
