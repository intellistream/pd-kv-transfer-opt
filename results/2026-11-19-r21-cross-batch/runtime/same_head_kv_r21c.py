"""r20 synchronous packed path with one guarded cross-batch experiment."""

from contextvars import ContextVar
from functools import wraps
import sys
import time

import r21_prefetch_c as prefetch
import prefill_chunk_observability as chunks
import same_head_kv_r16 as packed
import scoped_native_observability as native_observer
from prefill_chunk_observability import batch_key, chunk_rows
from r21_stage import zero_overlap


RUNNER = 'vllm_ascend.worker.model_runner_v1'
PLAN = ContextVar('r21_descriptor_collection', default=None)
BATCH = ContextVar('r21_worker_batch', default=None)
CALL = ContextVar('r21_store_call', default=None)
emit = native_observer.emit
lookahead = None


def with_context(event, **fields):
    if PLAN.get() is not None and event in ('worker_load_begin', 'worker_load_end'):
        return
    if batch := BATCH.get():
        fields = {**batch[0], **fields}
    if call := CALL.get():
        fields = {**call, **fields}
    emit(event, **fields)


def install_get(module):
    original = module.MooncakeBackend.get

    @wraps(original)
    def get(self, keys, addresses, sizes):
        capture = PLAN.get()
        if capture is None or self is not capture['backend']:
            return original(self, keys, addresses, sizes)
        capture['jobs'].append((capture['request_id'], self, keys, addresses, sizes))
        return [0] * len(keys)

    module.MooncakeBackend.get = get
    module.MooncakeBackend._r18_physical_get = original


def install_worker(module):
    original = module.KVPoolWorker.start_load_kv

    def collect(worker, metadata):
        capture = {'jobs': [], 'backend': worker.m_store}
        db = worker.token_database
        mask = db.load_mask
        own_mask = 'load_mask' in vars(db)
        ordinal = iter(r.req_id for r in prefetch.loadable(metadata))

        def load_mask(*args, **kwargs):
            capture['request_id'] = next(ordinal)
            return mask(*args, **kwargs)

        db.load_mask = load_mask
        token = PLAN.set(capture)
        try:
            result = original(worker, metadata)
        finally:
            PLAN.reset(token)
            if own_mask:
                db.load_mask = mask
            else:
                del db.load_mask
        if next(ordinal, None) is not None:
            raise RuntimeError('Packed preflight missed a scheduler load request')
        jobs = capture['jobs']
        plans = [packed.transfer_plan(keys, addrs, sizes, worker._same_head_groups, worker.tp_rank)
                 for _, _, keys, addrs, sizes in jobs]
        return result, jobs, plans

    @wraps(original)
    def load(self, metadata):
        global lookahead
        requests = prefetch.loadable(metadata)
        mode = prefetch.mode()
        if not requests or mode == 'baseline':
            return original(self, metadata)
        if mode not in ('packed', 'prepare-only', 'early-get') or not (
                self.kv_role == 'kv_producer' and self.tp_size == 4 and self.num_kv_head == 2
                and self.put_step == 2 and self.pcp_size == self.dcp_size == self.pp_size == 1
                and not self.load_async and not self.use_layerwise and self.backend_name == 'mooncake'):
            raise RuntimeError('Unexpected r21 worker mode or descriptor shape')
        begin = time.monotonic_ns()
        ids = [request.req_id for request in requests]
        with_context('worker_load_begin', request_ids=ids, load_async=False, tp_rank=self.tp_rank)
        if not hasattr(self, '_same_head_pair'):
            self._same_head_groups = packed.attention_groups(self, module)
            self._same_head_pair = packed.Pair(self.tp_rank)
            self._r21_verified_modes = set()
        pair = self._same_head_pair
        if prefetch.is_p0() and lookahead is None:
            lookahead = prefetch.Lookahead(self, collect, emit)
        context, runner, zero_blocks = BATCH.get()
        worker_take = None
        plan_begin = time.monotonic_ns()
        if lookahead is not None and len(requests) == 1:
            worker_take = lookahead.take(context['batch'], requests[0])
        if worker_take is not None:
            result, jobs, plans = worker_take['result'], [worker_take['job']], [worker_take['plan']]
        else:
            result, jobs, plans = collect(self, metadata)
        own = [(job[0], plan[1], len(plan[2])) for job, plan in zip(jobs, plans, strict=True)]
        gathered = [None, None]
        guard_begin = time.monotonic_ns()
        pair.dist.all_gather_object(gathered, own, group=pair.group)
        same_requests = [r[0] for r in gathered[0]] == [r[0] for r in gathered[1]]
        with_context('r18_packed_preflight', tp_rank=self.tp_rank, requests=len(ids), nonempty_gets=len(jobs),
                     same_request_order=same_requests, plan_ns=guard_begin-plan_begin,
                     wait_ns=time.monotonic_ns()-guard_begin)
        for n, (job, plan) in enumerate(zip(jobs, plans, strict=True)):
            rid, backend, keys, addrs, sizes = job
            keep, digest, segments, _ = plan
            packed_ok = same_requests and bool(segments) and gathered[0][n] == gathered[1][n]
            fields = {'call_ordinal': n, 'request_id': rid, 'tp_rank': self.tp_rank}
            token = CALL.set(fields)
            try:
                with_context('r18_restore_decision', mode='packed' if packed_ok else 'native',
                             native_bytes=sum(map(sum, sizes)), keys=len(keys), attention_segments=len(segments))
                selected = tuple([values[i] for i in keep] for values in (keys, addrs, sizes)) if packed_ok else (keys, addrs, sizes)
                selected_keys, selected_addrs, selected_sizes = selected
                amount = sum(map(sum, selected_sizes))
                physical = type(backend)._r18_physical_get
                early = worker_take is not None and worker_take['early']
                if lookahead is not None and worker_take is None:
                    hint = prefetch.restore(prefetch.path() / 'candidate')
                    if hint is not None and hint['request_id'] == rid:
                        zeroer, overlap = zero_overlap(runner, zero_blocks, selected_addrs, selected_sizes)
                        with_context('r21_plan', mode=mode, predecessor_batch=hint['predecessor_batch'],
                                     prepare_begin_ns=plan_begin, prepare_end_ns=guard_begin,
                                     zero_overlap_bytes=overlap, zeroer_active=zeroer, zero_block_count=len(zero_blocks),
                                     total_bytes=amount, staging_bytes=0, dispatch_ns=hint['dispatch_ns'],
                                     prior_used_block_overlap=hint['block_alias'], safe_live_write_ns=time.monotonic_ns(),
                                     reason='normal_worker_path')
                scatter = {}
                if early:
                    if not packed_ok or amount != prefetch.TARGET_RANK_BYTES:
                        raise RuntimeError('r21 staged bytes no longer match packed restore')
                    scatter = lookahead.stage.scatter(selected_addrs, selected_sizes)
                    success = True
                else:
                    response = physical(backend, *selected)
                    success = response is not None and len(response) == len(selected_keys) and not any(response)
                if worker_take is not None:
                    with_context('r21_consume', mode=mode, predecessor_batch=worker_take['fields']['predecessor_batch'],
                                 wait_ns=worker_take['wait_ns'], stage_copy_ns=scatter.get('end_ns', 0)-scatter.get('begin_ns', 0),
                                 stage_copy_device_ms=scatter.get('device_elapsed_ms'),
                                 old_forward_device_ms=worker_take['old_forward_device_ms'],
                                 preparation_saved_ns=worker_take['prepare_ns'], used_early_get=early,
                                 safe_live_write_ns=scatter.get('begin_ns', time.monotonic_ns()))
                if not packed_ok:
                    if not success:
                        raise RuntimeError('Native GET failed')
                    continue
                started = time.monotonic_ns()
                pair.exchange(digest, bytes(row[3] for row in segments), success)
                admitted = time.monotonic_ns()
                operations, device_ms = pair.copy_packed(segments)
                returned = time.monotonic_ns()
                with_context('same_head_local_copy', request_ids=[rid], payload_bytes=sum(
                    size for owner, _, size, _, _ in segments if owner != pair.rank), pair_wait_ns=admitted-started,
                    verify=False, method='packed', copy_begin_ns=admitted, copy_end_ns=returned, operations=operations,
                    device_elapsed_ms=device_ms, elapsed_ns=returned-admitted)
                if mode not in self._r21_verified_modes:
                    omitted = next((i for i in range(len(keys)) if i not in set(keep)), None)
                    if omitted is not None:
                        sample = [(addrs[omitted][0], min(sizes[omitted][0], 4096))]
                        old = packed.read_samples(sample)
                        checked = physical(backend, [keys[omitted]], [addrs[omitted]], [sizes[omitted]])
                        if checked is None or len(checked) != 1 or any(checked) or old != packed.read_samples(sample):
                            raise RuntimeError('r21 same-head sample differs from the independent GET')
                        with_context('same_head_verify', request_ids=ids, sample_bytes=sample[0][1], samples=1,
                                     method='packed', extra_get_bytes=sum(sizes[omitted]), run_mode=mode)
                    self._r21_verified_modes.update(('packed', 'prepare-only', 'early-get'))
            finally:
                CALL.reset(token)
        with_context('worker_load_end', request_ids=ids, load_async=False, tp_rank=self.tp_rank,
                     elapsed_ns=time.monotonic_ns()-begin)
        return result

    module.KVPoolWorker.start_load_kv = load


def install_runner(module):
    original_execute = module.NPUModelRunner.execute_model
    original_forward = module.NPUModelRunner._model_forward

    @wraps(original_execute)
    def execute(self, output, *args, **kwargs):
        if not output.total_num_scheduled_tokens:
            return original_execute(self, output, *args, **kwargs)
        context = {'batch': batch_key(chunk_rows(output)), 'tp_rank': self.device.index,
                   'worker_enter_ns': time.monotonic_ns()}
        token = BATCH.set((context, self, output.new_block_ids_to_zero or ()))
        try:
            return original_execute(self, output, *args, **kwargs)
        finally:
            BATCH.reset(token)

    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        context = BATCH.get()
        markers = lookahead.begin(self, context[0]['batch'], module.torch) if context and lookahead else None
        started = time.monotonic_ns()
        result = original_forward(self, *args, **kwargs)
        submitted = time.monotonic_ns()
        if markers is not None:
            event = module.torch.npu.Event(enable_timing=True)
            event.record()
            markers['last'] = event
        if context:
            with_context('p_chunk_rank_forward', forward_enter_ns=started, forward_cpu_return_ns=submitted)
        return result

    module.NPUModelRunner.execute_model = execute
    module.NPUModelRunner._model_forward = forward


TARGETS = {
    packed.WORKER: install_worker, packed.BACKEND: install_get,
    RUNNER: install_runner,
}


def install(module):
    if not getattr(module, '_issue23_r21', False):
        TARGETS[module.__name__](module)
        module._issue23_r21 = True


class Loader:
    def __init__(self, downstream):
        self.downstream = downstream

    def create_module(self, spec):
        return self.downstream.create_module(spec)

    def exec_module(self, module):
        self.downstream.exec_module(module)
        install(module)


class Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        for finder in sys.meta_path[sys.meta_path.index(self)+1:]:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader)
                return spec
        return None


class InstallFilter(packed.BaseFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not any(isinstance(finder, Finder) for finder in sys.meta_path):
            observe_scheduler = chunks.observe_scheduler

            def attach(scheduler):
                observe_scheduler(scheduler)
                prefetch.attach_scheduler(scheduler, emit)

            chunks.observe_scheduler = attach
            native_observer.emit = with_context
            sys.meta_path.insert(0, Finder())
            for name in TARGETS:
                if name in sys.modules:
                    install(sys.modules[name])
