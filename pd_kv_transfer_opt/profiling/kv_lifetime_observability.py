"""Observe actual scheduler page checks and P→D page references during a replay."""

from collections import Counter
from functools import wraps
import os
import sys
import time

from pd_kv_transfer_opt.profiling.scoped_native_observability import emit


ENGINE = "vllm.v1.engine.core"
SECOND = 1_000_000_000


def attach_prefill(scheduler):
    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    pool = manager.block_pool
    allocate = manager.allocate_slots
    free = manager.free
    count_needed = coordinator.get_num_blocks_to_allocate
    count_free = pool.get_num_free_blocks
    connector_finished = scheduler._connector_finished
    transfer_finished = scheduler._update_from_kv_xfer_finished
    schedule = scheduler.schedule
    held = {}
    held_refs = Counter()
    attempts = Counter()
    blocked = {}
    active = False
    need = None
    capacity = None
    prefix = ()
    full_gate = False
    releasing = set()
    last_sample = 0
    min_free = pool.num_gpu_blocks
    min_slack = pool.num_gpu_blocks

    @wraps(count_needed)
    def need_pages(request_id, num_tokens, new_computed_blocks, num_encoder_tokens,
                   total_computed_tokens, num_tokens_main_model, apply_admission_cap=False):
        nonlocal need, capacity, prefix, full_gate
        count = count_needed(request_id, num_tokens, new_computed_blocks, num_encoder_tokens,
                             total_computed_tokens, num_tokens_main_model, apply_admission_cap)
        if active:
            need = count
            capacity = None
            prefix = new_computed_blocks
            full_gate = apply_admission_cap
        return count

    @wraps(count_free)
    def free_count():
        nonlocal capacity
        value = count_free()
        if active and need is not None and capacity is None:
            capacity = value
        return value

    def detachable():
        return {block_id for block_id, count in held_refs.items()
                if pool.blocks[block_id].ref_cnt == count}

    @wraps(allocate)
    def allocate_slots(request, *args, **kwargs):
        nonlocal active, need, capacity, min_free, min_slack
        request_id = request.request_id
        need = capacity = None
        active = True
        try:
            result = allocate(request, *args, **kwargs)
        finally:
            active = False
        attempts["calls"] += 1
        if need is None or capacity is None:
            attempts["unobserved_checks"] += 1
            return result

        reserve = 0 if full_gate else kwargs.get("reserved_blocks", 0)
        slack = capacity - reserve - need
        min_free = min(min_free, capacity)
        min_slack = min(min_slack, slack)
        if result is not None:
            if request_id in blocked:
                episode = blocked.pop(request_id)
                emit("p_pool_allocation_resumed", request_id=request_id, attempts=episode["attempts"],
                     first_fail_ns=episode["first_ns"], last_fail_ns=episode["last_ns"],
                     free_blocks=capacity, need_blocks=need)
            return result

        attempts["failures"] += 1
        now = time.monotonic_ns()
        episode = blocked.setdefault(request_id, {"first_ns": now, "last_log_ns": 0, "attempts": 0})
        episode["last_ns"] = now
        episode["attempts"] += 1
        signature = (capacity, need, reserve, full_gate, len(held_refs), len(held))
        if signature == episode.get("signature") and now - episode["last_log_ns"] < SECOND:
            return result
        episode["signature"] = signature
        episode["last_log_ns"] = now
        reclaimable = detachable()
        # A newly freed APC hit would have to be touched again by this allocation.
        prefix_ids = {block.block_id for group in prefix for block in group if not block.is_null}
        emit("p_pool_denied", request_id=request_id, at_ns=now, first_fail_ns=episode["first_ns"],
             computation=request.num_computed_tokens, gate="full_sequence" if full_gate else "current_chunk",
             free_blocks=capacity, need_blocks=need, reserved_blocks=reserve, attempts=episode["attempts"],
             holder_count=len(held), held_blocks=len(held_refs),
             detachable_blocks=len(reclaimable), detachable_outside_prefix=len(reclaimable - prefix_ids),
             pending_holders={key: len(blocks.keys() & reclaimable) for key, blocks in held.items()},
             running=len(scheduler.running), waiting=len(scheduler.waiting))
        return result

    @wraps(connector_finished)
    def retain(request):
        delayed, params = connector_finished(request)
        if delayed and params and params.get("do_remote_prefill"):
            request_id = request.request_id
            refs = Counter(block.block_id for group in manager.get_blocks(request_id).blocks
                           for block in group if not block.is_null)
            held[request_id] = refs
            held_refs.update(refs)
            emit("p_handoff_retain", request_id=request_id, referenced_blocks=len(refs),
                 exclusively_owned_blocks=sum(pool.blocks[key].ref_cnt == cnt for key, cnt in refs.items()),
                 free_blocks=count_free())
        return delayed, params

    @wraps(transfer_finished)
    def complete(output):
        nonlocal releasing
        releasing = output.finished_sending or set()
        try:
            return transfer_finished(output)
        finally:
            releasing = set()

    @wraps(free)
    def free_slots(request):
        request_id = request.request_id
        refs = held.pop(request_id, None)
        if refs is None:
            return free(request)
        before = count_free()
        result = free(request)
        for block_id, count in refs.items():
            held_refs[block_id] -= count
            if not held_refs[block_id]:
                del held_refs[block_id]
        after = count_free()
        emit("p_handoff_release", request_id=request_id, referenced_blocks=len(refs),
             free_before=before, free_after=after, returned_blocks=after - before,
             cause="connector_finished_sending" if request_id in releasing else "other")
        return result

    @wraps(schedule)
    def schedule_batch():
        nonlocal last_sample, min_free, min_slack
        output = schedule()
        now = time.monotonic_ns()
        first = [request.req_id for request in output.scheduled_new_reqs
                 if output.num_scheduled_tokens.get(request.req_id, 0)]
        if first:
            emit("p_allocation_first_dispatch", at_ns=now, request_ids=first)
        if now - last_sample >= SECOND:
            emit("p_pool_sample", calls=attempts["calls"], failures=attempts["failures"],
                 unobserved_checks=attempts["unobserved_checks"], min_attempt_free=min_free,
                 min_attempt_slack=min_slack, free_blocks=count_free(), holder_count=len(held),
                 held_blocks=len(held_refs), detachable_blocks=len(detachable()),
                 running=len(scheduler.running), waiting=len(scheduler.waiting))
            last_sample = now
            min_free = min_slack = pool.num_gpu_blocks
        return output

    coordinator.get_num_blocks_to_allocate = need_pages
    pool.get_num_free_blocks = free_count
    manager.allocate_slots = allocate_slots
    manager.free = free_slots
    scheduler._connector_finished = retain
    scheduler._update_from_kv_xfer_finished = complete
    scheduler.schedule = schedule_batch
    emit("p_pool_attached", capacity_blocks=pool.num_gpu_blocks, free_blocks=count_free(),
         full_sequence_gate=scheduler.scheduler_reserve_full_isl,
         group_specs=[type(group.kv_cache_spec).__name__ for group in scheduler.kv_cache_config.kv_cache_groups])


def attach_decode(scheduler):
    add_request = scheduler.add_request
    allocate = scheduler.kv_cache_manager.allocate_slots
    transfer_finished = scheduler._update_from_kv_xfer_finished
    remote_ids = {}
    allocated = set()

    @wraps(add_request)
    def add(request):
        result = add_request(request)
        remote = (request.kv_transfer_params or {}).get("remote_request_id")
        if remote is not None:
            remote_ids[request.request_id] = remote
            emit("d_paged_receive_arrival", request_id=request.request_id, p_request_id=remote)
        return result

    @wraps(allocate)
    def allocate_slots(request, *args, **kwargs):
        result = allocate(request, *args, **kwargs)
        key = request.request_id
        if result is not None and key in remote_ids and key not in allocated:
            allocated.add(key)
            emit("d_paged_receive_allocated", request_id=key, p_request_id=remote_ids[key],
                 external_tokens=kwargs.get("num_external_computed_tokens", 0),
                 load_async=kwargs.get("delay_cache_blocks", False))
        return result

    @wraps(transfer_finished)
    def complete(output):
        result = transfer_finished(output)
        for key in output.finished_recving or ():
            if key in remote_ids:
                emit("d_paged_receive_ready", request_id=key, p_request_id=remote_ids.pop(key))
                allocated.discard(key)
        return result

    scheduler.add_request = add
    scheduler.kv_cache_manager.allocate_slots = allocate_slots
    scheduler._update_from_kv_xfer_finished = complete
    emit("d_pool_attached")


def install(module):
    cls = module.EngineCore
    initial = cls.__init__
    if getattr(initial, "_pd_kv_lifetime_observed", False):
        return

    @wraps(initial)
    def init(self, *args, **kwargs):
        initial(self, *args, **kwargs)
        if os.environ["ISSUE23_ROLE"] == "P":
            attach_prefill(self.scheduler)
        else:
            attach_decode(self.scheduler)

    init._pd_kv_lifetime_observed = True
    cls.__init__ = init


class Loader:
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        install(module)


class Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != ENGINE:
            return None
        for finder in sys.meta_path[sys.meta_path.index(self) + 1:]:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader)
                return spec
        return None


class InstallFilter:
    def __init__(self):
        if not any(isinstance(finder, Finder) for finder in sys.meta_path):
            sys.meta_path.insert(0, Finder())
            if ENGINE in sys.modules:
                install(sys.modules[ENGINE])

    def filter(self, _record):
        return True
