"""Opt-in cold-request ordering on the P scheduler with per-arm receipts."""

from functools import wraps
import importlib.abc
import importlib.machinery
from itertools import combinations
import logging
import os
from pathlib import Path
import re
import sys
import time

from pd_kv_transfer_opt.common import read_json, write_json
from pd_kv_transfer_opt.profiling.scoped_native_observability import emit


MODES = ("native", "sjf", "history")
P_ROLES = ("P0", "P1")
GATE_S = 5
MAX_PROMOTIONS = 2
P_SECONDS_PER_TOKEN = 0.00006
HINT_ROOT = "pd_kv_cold_root"
HINT_CASE = "pd_kv_order_case"
HINT_MODE = "pd_kv_order_mode"
HINT_SUFFIX = "pd_kv_suffix_s"
TARGET = "vllm.v1.core.sched.scheduler"


def select_promotions(snapshots):
    """The same visible-information score used by the offline screen."""
    keys = [item["id"] for role in P_ROLES for item in snapshots[role]["waiting"]]

    def finish(choice):
        ranks = {key: index for index, key in enumerate(choice)}
        tails = []
        for role in P_ROLES:
            snapshot = snapshots[role]
            queue = sorted(snapshot["waiting"], key=lambda item: ranks.get(item["id"], len(choice)))
            p_finish = snapshot["running_tokens"] * P_SECONDS_PER_TOKEN
            for item in queue:
                p_finish += item["prompt_tokens"] * P_SECONDS_PER_TOKEN
                tails.append(p_finish + item["suffix_s"])
        return max(tails, default=0.0)

    choices = (choice for size in range(MAX_PROMOTIONS + 1) for choice in combinations(keys, size))
    return min(choices, key=lambda choice: (finish(choice), len(choice)))


def reorder_waiting(waiting, candidates, mode, selected=()):
    """Return a permutation with the noncandidate positions intact."""
    remaining = iter(candidates)
    if mode == "sjf":
        remaining = iter(sorted(candidates, key=lambda request: request.num_prompt_tokens))
    elif mode == "history":
        ranks = {root: rank for rank, root in enumerate(selected)}
        remaining = iter(sorted(candidates, key=lambda request: ranks.get(extras(request)[HINT_ROOT], len(ranks))))
    keys = {request.request_id for request in candidates}
    return [next(remaining) if request.request_id in keys else request for request in waiting]


def extras(request):
    params = request.sampling_params
    return (params.extra_args if params is not None else None) or {}


def atomic_json(path, item):
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, item)
    temporary.replace(path)


def optional_json(path):
    try:
        return read_json(path)
    except FileNotFoundError:
        return None


class Arm:
    def __init__(self, name, mode, role, directory):
        if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name) is None or mode not in MODES:
            raise ValueError("Invalid prefill-order arm")
        self.name, self.mode, self.role = name, mode, role
        self.dir = directory / name
        self.peer = next(other for other in P_ROLES if other != role)
        self.first_ns = time.monotonic_ns()
        self.gate_ns = None
        self.snapshot = None
        self.undispatched = {}
        self.decision_done = mode == "native"
        if not self.decision_done:
            atomic_json(self.dir / f"{role}-first.json", self.first_ns)

    def fields(self):
        return {"case": self.name, "mode": self.mode, "instance": self.role}

    def check_gate(self):
        now = time.monotonic_ns()
        if now < self.first_ns + GATE_S * 10**9:
            return False
        if self.gate_ns is None:
            peer_ns = optional_json(self.dir / f"{self.peer}-first.json")
            if peer_ns is None:
                return False
            self.gate_ns = max(self.first_ns, peer_ns) + GATE_S * 10**9
        return now >= self.gate_ns

    def decide(self, scheduler, waiting_status):
        if not self.check_gate():
            return
        waiting = list(scheduler.waiting)
        candidates = [r for r in waiting if r.request_id in self.undispatched
                      and r.num_computed_tokens == 0 and r.status == waiting_status]
        if self.snapshot is None:
            self.snapshot = {
                "running_tokens": sum(max(0, request.num_prompt_tokens - request.num_computed_tokens)
                                      for request in scheduler.running),
                "waiting": [{"id": extras(request)[HINT_ROOT], "prompt_tokens": request.num_prompt_tokens,
                             "suffix_s": extras(request)[HINT_SUFFIX]} for request in candidates],
            }
            if self.mode == "history":
                atomic_json(self.dir / f"{self.role}-queue.json", self.snapshot)
        selected = ()
        if self.mode == "history":
            peer = optional_json(self.dir / f"{self.peer}-queue.json")
            if peer is None:
                return
            selected = select_promotions({self.role: self.snapshot, self.peer: peer})
        revised = reorder_waiting(waiting, candidates, self.mode, selected)
        if revised != waiting:
            scheduler.waiting.clear()
            scheduler.waiting.extend(revised)
        self.decision_done = True
        candidate_ids = {r.request_id for r in candidates}
        emit("prefill_order_decision", **self.fields(), selected=list(selected),
             before=[extras(r)[HINT_ROOT] for r in candidates],
             after=[extras(r)[HINT_ROOT] for r in revised if r.request_id in candidate_ids],
             active=len(scheduler.running), gate_ns=self.gate_ns)


def attach(scheduler, module, role, directory):
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue

    if not isinstance(scheduler.waiting, FCFSRequestQueue):
        raise TypeError("Cold-prefill ordering requires the pinned FCFS waiting queue")
    original_add, original_schedule = scheduler.add_request, scheduler.schedule
    arm = None

    @wraps(original_add)
    def add(request):
        nonlocal arm
        hint = extras(request)
        result = original_add(request)
        if HINT_ROOT not in hint:
            return result
        if arm is None or arm.name != hint[HINT_CASE]:
            arm = Arm(hint[HINT_CASE], hint[HINT_MODE], role, directory)
        elif arm.mode != hint[HINT_MODE]:
            raise ValueError("Prefill-order mode changed in-flight")
        arm.undispatched[request.request_id] = (hint[HINT_ROOT], time.monotonic_ns())
        return result

    @wraps(original_schedule)
    def schedule():
        if arm is not None and not arm.decision_done:
            arm.decide(scheduler, module.RequestStatus.WAITING)
        output = original_schedule()
        if arm is not None and arm.undispatched:
            for internal_id, count in output.num_scheduled_tokens.items():
                if count and (root := arm.undispatched.pop(internal_id, None)) is not None:
                    emit("prefill_order_root_started", **arm.fields(), request_id=root[0],
                         arrival_ns=root[1], first_chunk_tokens=count)
        return output

    scheduler.add_request, scheduler.schedule = add, schedule
    emit("prefill_order_hook_attached", instance=role)


def install(module):
    if getattr(module, "_pd_kv_prefill_order", False):
        return
    original = module.Scheduler.__init__
    role = os.environ["PDKV_PREFILL_ORDER_INSTANCE"]
    directory = Path(os.environ["PDKV_PREFILL_ORDER_DIR"])

    @wraps(original)
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        attach(self, module, role, directory)

    module.Scheduler.__init__ = init
    module._pd_kv_prefill_order = True


class Loader:
    def __init__(self, downstream):
        self.downstream = downstream

    def create_module(self, spec):
        return self.downstream.create_module(spec)

    def exec_module(self, module):
        self.downstream.exec_module(module)
        install(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == TARGET:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader:
                spec.loader = Loader(spec.loader)
            return spec


_installed = False


class InstallFilter(logging.Filter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global _installed
        if _installed or os.environ.get("PDKV_PREFILL_ORDER_INSTANCE") not in P_ROLES:
            return
        sys.meta_path.insert(0, Finder())
        if TARGET in sys.modules:
            install(sys.modules[TARGET])
        _installed = True
