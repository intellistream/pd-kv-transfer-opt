"""Closed-loop CPU replay of the native two-P, two-D serving path."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from math import ceil, isclose

from pd_kv_transfer_opt.profiling.screen_restore_residency import (
    ATTENTION_BLOCK_TOKENS as PAGE,
    ATTENTION_BYTES,
    P_CAPACITY_TOKENS,
)
from pd_kv_transfer_opt.simulator.agent_cache import Lease, LocalCache, STATE_BYTES_PER_CHECKPOINT, Store
from pd_kv_transfer_opt.workloads.freeinference import Session


GB = 10**9
EPS = 1e-8
STORE_BYTES = 128 * 2**30


def prefill_score(body_bytes):
    return body_bytes / 4 * .0345 + 120.0745


@dataclass(frozen=True)
class Service:
    shares: tuple[float, ...]
    decode_first_s: float
    decode_add_s: float
    cold_tokens_s: float
    ingress_base_s: float
    ingress_s_per_byte: float
    body_bytes_per_token: float
    get_width: int
    get_setup_s: float = .03
    first_admission_s: float = .45
    rail_efficiency: float = .88
    warm_tokens_s: float = 24800
    chunk_tokens: int = 4096
    slots: int = 16
    p_pages: int = P_CAPACITY_TOKENS // PAGE
    d_pages: int = P_CAPACITY_TOKENS // PAGE
    store_bytes: int = STORE_BYTES
    checkpoint_stride: int = 1
    decode_ingress_s: float = .09
    pull_setup_s: float = .04
    pull_GB_s: float = 80
    put_setup_s: float = .01
    put_GB_s_per_P: float = 37.5
    release_suffix_first: bool = False


@dataclass(slots=True)
class Request:
    session: Session
    step: int
    offered: float
    body: float
    p_score: float
    row: dict = field(default_factory=dict)
    remaining: float = 0
    covered: int = 0
    p_lease: Lease | None = None
    d_lease: Lease | None = None

    @property
    def turn(self):
        return self.session.turns[self.step]

    @property
    def key(self):
        return f"{self.session.name}:{self.step}"

    @property
    def lookup_pages(self):
        return self.turn.pages[:(self.turn.prompt - 1) // PAGE]


@dataclass
class Prefill:
    cache: LocalCache
    waiting: deque = field(default_factory=deque)
    active: deque = field(default_factory=deque)
    stage: str = "idle"
    put_free_s: float = 0
    warm: bool = False


@dataclass
class Decode:
    cache: LocalCache
    waiting: deque = field(default_factory=deque)
    preparing: int = 0
    active: list = field(default_factory=list)


@dataclass
class GetBatch:
    rows: list
    remaining_GB: float


class Replay:
    def __init__(self, sessions, offers, service, cap, *, turn_offers=None, observed_body_bytes=None,
                 ingress_residual_s=None):
        if not sessions or cap <= 0 or len(service.shares) != 8 or not isclose(sum(service.shares), 1):
            raise ValueError("Replay requires whole sessions and eight calibrated rails")
        if set(offers) != {session.name for session in sessions} or min(offers.values()) != 0:
            raise ValueError("Replay start plan must cover each session and start at t=0")
        self.service = service
        self.GB_s = cap * service.rail_efficiency / (8 * len(service.shares) * max(service.shares))
        self.cap = cap
        self.p = [Prefill(LocalCache(service.p_pages, release_suffix_first=service.release_suffix_first))
                  for _ in range(2)]
        self.d = [Decode(LocalCache(service.d_pages, release_suffix_first=service.release_suffix_first))
                  for _ in range(2)]
        self.load_p = [0.0, 0.0]
        self.load_d = [0.0, 0.0]
        self.store = Store(service.store_bytes)
        self.gets = {}
        self.requests = {}
        self.events, self.serial = [], count()
        self.tx_epochs, self.supply_epochs = [], []
        self.put_GB = 0.0
        self.now = 0.0
        self.done = 0
        self.expected_count = sum(len(s.turns) for s in sessions)
        self.turn_offers = turn_offers or {}
        self.observed_body_bytes = observed_body_bytes or {}
        self.ingress_residual_s = ingress_residual_s or {}
        for session in sorted(sessions, key=lambda s: (s.source_start_ms, s.name)):
            self.offer(session, 0, offers[session.name])

    def defer(self, at, handler, *args):
        heappush(self.events, (at, next(self.serial), handler, args))

    def offer(self, session, step, at):
        key = f"{session.name}:{step}"
        at = max(at, self.turn_offers.get(key, at))
        body = self.observed_body_bytes.get(key, session.turns[step].prompt * self.service.body_bytes_per_token)
        job = Request(session, step, at, body, prefill_score(body))
        self.requests[key] = job
        ingress = max(0., self.service.ingress_base_s + body * self.service.ingress_s_per_byte
                      + self.ingress_residual_s.get(key, 0.))
        self.defer(at + ingress, self.p_arrive, job)

    def p_arrive(self, job):
        role = int(self.load_p[1] + EPS < self.load_p[0])
        self.load_p[role] += job.p_score
        job.row.update(P=role, offered_s=job.offered, P_admit_s=self.now)
        target = self.p[role]
        target.waiting.append(job)
        if not target.warm and target.stage == "idle":
            target.stage = "starting"
            self.defer(self.now + self.service.first_admission_s, self.p_start, role)

    def p_start(self, role):
        self.p[role].stage = "idle"

    def allocate_prefill(self, role, job):
        target = self.p[role]
        pages = job.lookup_pages
        local = target.cache.prefix(pages)
        remote, shadow = self.store.lookup(pages, local)
        lease = target.cache.reserve(pages[:local], job.turn.prompt)
        if lease is None:
            return None
        job.p_lease = lease
        attention = pages[local:remote]
        states = tuple(key for key in attention[-3:] if -key in self.store.objects)
        pins = self.store.acquire(attention, states)
        bytes_ = len(attention) * ATTENTION_BYTES + len(states) * STATE_BYTES_PER_CHECKPOINT
        newly_computed = job.turn.prompt - remote * PAGE
        job.remaining = newly_computed
        job.covered = remote
        adjacent = job.turn.adjacent_pages
        job.row.update(P_dispatch_s=self.now, P_local_pages=local, P_external_pages=len(attention),
                       P_lookup_pages=len(pages), P_computed_tokens=newly_computed,
                       P_adjacent_local_pages=min(adjacent, local),
                       P_adjacent_store_pages=max(0, min(adjacent, remote) - local),
                       P_adjacent_eviction_pages=max(0, min(adjacent, shadow) - remote),
                       P_store_capacity_lost_pages=shadow - remote, GET_bytes=bytes_,
                       GET_states=len(states), GET_submit_s=None, GET_done_s=None,
                       P_compute_s=0.0, P_compute_chunks=0)
        return job, attention, states, pins

    def p_tick(self, role):
        target = self.p[role]
        if target.stage != "idle":
            return
        loads = []
        available_tokens = self.service.chunk_tokens - sum(job.remaining for job in target.active)
        width = self.service.get_width if target.warm else 1
        # The native scheduler grants the chunk budget to running requests
        # before admitting another request from the waiting queue.
        while (available_tokens > 0 and target.waiting and len(target.active) < self.service.slots
               and len(loads) < width):
            part = self.allocate_prefill(role, target.waiting[0])
            if part is None:
                break
            target.active.append(target.waiting.popleft())
            loads.append(part)
            available_tokens -= part[0].remaining
        if any(part[1] for part in loads):
            target.stage = "get_setup"
            self.defer(self.now + self.service.get_setup_s, self.get_start, role, loads)
        elif target.active:
            self.compute(role)
        if loads:
            target.warm = True

    def get_start(self, role, loads):
        self.p[role].stage = "get"
        for job, attention, _, _ in loads:
            if attention:
                job.row["GET_submit_s"] = self.now
        self.gets[role] = GetBatch(loads, sum(job.row["GET_bytes"] for job, *_ in loads) / GB)

    def get_finish(self, role):
        target = self.p[role]
        batch = self.gets.pop(role)
        for job, attention, states, pins in batch.rows:
            if attention:
                job.row["GET_done_s"] = self.now
                target.cache.publish(job.p_lease, attention, states)
                self.store.release(pins)
        self.compute(role)

    def compute(self, role):
        target = self.p[role]
        slots = self.service.chunk_tokens
        updates, duration = [], 0.0
        for job in target.active:
            count = min(slots, job.remaining)
            warm_suffix = ((job.row["P_local_pages"] or job.row["P_external_pages"])
                           and job.row["P_computed_tokens"] <= self.service.chunk_tokens)
            rate = self.service.warm_tokens_s if warm_suffix else self.service.cold_tokens_s
            updates.append((job, count, count / rate))
            duration += count / rate
            slots -= count
            if not slots:
                break
        target.stage = "compute"
        self.defer(self.now + duration, self.compute_finish, role, updates)

    def compute_finish(self, role, updates):
        target = self.p[role]
        target.stage = "idle"
        for job, work, seconds in updates:
            job.row["P_compute_s"] += seconds
            job.row["P_compute_chunks"] += 1
            job.remaining -= work
            full = min(len(job.turn.pages), (job.turn.prompt - job.remaining) // PAGE)
            pages = job.turn.pages[job.covered:full]
            # The dense case exposes every complete checkpoint. The sparse
            # sensitivity exposes chunk-aligned and final eligible checkpoints.
            states = tuple(key for i, key in enumerate(pages, job.covered + 1)
                           if i % self.service.checkpoint_stride == 0 or i == full)
            target.cache.publish(job.p_lease, pages, states)
            self.put(role, (*pages, *(-key for key in states)))
            job.covered = full
            if job.remaining == 0:
                target.active.popleft()
                job.row["P_reply_s"] = self.now
                self.d_route(job)

    def put(self, role, keys):
        # Existing object keys only refresh residency. Data sent by another
        # producer before this submission commits may still be sent twice.
        new_GB = sum(self.store.size(key) for key in keys if key not in self.store.objects) / GB
        if not keys:
            return
        target = self.p[role]
        target.put_free_s = (max(target.put_free_s, self.now) + self.service.put_setup_s
                             + new_GB / self.service.put_GB_s_per_P)
        self.put_GB += new_GB
        self.defer(target.put_free_s, self.store.publish, keys)

    def d_route(self, job):
        role = int(self.load_d[1] + EPS < self.load_d[0])
        self.load_d[role] += job.body
        job.row["D"] = role
        self.defer(self.now + self.service.decode_ingress_s, self.d_arrive, role, job)

    def d_arrive(self, role, job):
        self.d[role].waiting.append(job)

    def d_tick(self, role):
        target = self.d[role]
        while target.waiting and target.preparing + len(target.active) < self.service.slots:
            job = target.waiting[0]
            pages = job.lookup_pages
            local = target.cache.prefix(pages)
            lease = target.cache.reserve(pages[:local], job.turn.prompt + job.turn.output)
            if lease is None:
                break
            target.waiting.popleft()
            target.preparing += 1
            job.d_lease = lease
            blocks = ceil(job.turn.prompt / PAGE) - local
            # Equal-TP Pull transfers each missing Attention rank copy and the
            # active recurrent state; compute fills any uncached partial tile.
            bytes_ = blocks * ATTENTION_BYTES + STATE_BYTES_PER_CHECKPOINT
            job.row.update(D_alloc_s=self.now, D_local_pages=local, P_to_D_bytes=bytes_)
            self.defer(self.now + self.service.pull_setup_s + bytes_ / GB / self.service.pull_GB_s,
                       self.d_ready, role, job)

    def d_ready(self, role, job):
        target = self.d[role]
        target.preparing -= 1
        job.row["D_KV_ready_s"] = self.now
        pages = job.turn.pages[job.row["D_local_pages"]:]
        target.cache.publish(job.d_lease, pages, pages)
        self.p[job.row["P"]].cache.release(job.p_lease)
        job.remaining = float(job.turn.output)
        target.active.append(job)

    def d_complete_tokens(self):
        for role, target in enumerate(self.d):
            for job in tuple(target.active):
                if "D_first_s" not in job.row and job.remaining <= job.turn.output - 1 + EPS:
                    job.row["D_first_s"] = self.now
                    self.load_p[job.row["P"]] -= job.p_score
                    self.on_first_decode_token(job)
                if job.remaining <= EPS:
                    target.active.remove(job)
                    target.cache.release(job.d_lease)
                    self.load_d[role] -= job.body
                    job.row["EOS_s"] = self.now
                    self.done += 1
                    step = job.step + 1
                    if step < len(job.session.turns):
                        gap = job.session.turns[step].tool_gap_ms / 1000
                        self.offer(job.session, step, self.now + gap)

    def on_first_decode_token(self, job):
        """Client-side arrival policies can observe an engine's first token."""

    def advance(self, end, periods):
        elapsed = end - self.now
        if not elapsed:
            return
        rates = tuple(self.GB_s * part * 8 if self.gets else 0 for part in self.service.shares)
        widths = tuple(len(target.active) for target in self.d)
        pending = tuple(len(target.waiting) + len(target.active) for target in self.p)
        if self.tx_epochs and self.tx_epochs[-1][2] == rates:
            self.tx_epochs[-1] = (self.tx_epochs[-1][0], end, rates)
        else:
            self.tx_epochs.append((self.now, end, rates))
        self.supply_epochs.append((self.now, end, widths, pending, tuple(target.stage for target in self.p),
                                   len(self.gets), tuple(len(target.waiting) + target.preparing for target in self.d)))
        for batch in self.gets.values():
            batch.remaining_GB -= elapsed * self.GB_s / len(self.gets)
        for target, period in zip(self.d, periods, strict=True):
            for job in target.active:
                job.remaining -= elapsed / period
        self.now = end

    def run(self):
        while self.done < self.expected_count:
            while self.events and self.events[0][0] <= self.now:
                _, _, handler, args = heappop(self.events)
                handler(*args)
            for role in range(2):
                self.d_tick(role)
                self.p_tick(role)
            times = [self.events[0][0]] if self.events else []
            if self.gets:
                times.append(self.now + max(0, min(b.remaining_GB for b in self.gets.values()))
                             * len(self.gets) / self.GB_s)
            periods = []
            for target in self.d:
                period = self.service.decode_first_s + self.service.decode_add_s * max(0, len(target.active) - 1)
                periods.append(period)
                for job in target.active:
                    remaining = job.remaining
                    if "D_first_s" not in job.row:
                        remaining -= job.turn.output - 1
                    times.append(self.now + max(0, remaining) * period)
            if not times:
                raise RuntimeError("No event can release the remaining native KVs or requests")
            self.advance(min(times), periods)
            for role, batch in tuple(self.gets.items()):
                if batch.remaining_GB <= EPS:
                    self.get_finish(role)
            self.d_complete_tokens()
        if any(self.store.pins) or any(engine.cache.live or engine.cache.reserved for engine in self.p + self.d):
            raise RuntimeError("A completed replay has live KV leases")
        return self
