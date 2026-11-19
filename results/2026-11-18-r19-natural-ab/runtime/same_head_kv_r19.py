"""Guard r16 packed restore for partial APC hits and trace worker gates."""

from contextvars import ContextVar
from functools import wraps
import sys
import os
from pathlib import Path
import time

import same_head_kv_r16 as packed
import scoped_native_observability as native_observer
from prefill_chunk_observability import batch_key, chunk_rows


RUNNER = 'vllm_ascend.worker.model_runner_v1'
PLAN = ContextVar('r18_descriptor_collection', default=None)
BATCH = ContextVar('r18_worker_batch', default=None)
CALL = ContextVar('r18_store_call', default=None)
emit = native_observer.emit


def with_context(event, **fields):
    if PLAN.get() is not None and event in ('worker_load_begin', 'worker_load_end'):
        return
    if batch := BATCH.get():
        fields = {**batch, **fields}
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

    @wraps(original)
    def load(self, metadata):
        ids = [r.req_id for r in metadata.requests if r.load_spec is not None and r.load_spec.can_load]
        if not ids:
            return original(self, metadata)
        mode_file = Path(os.environ['ISSUE23_SAME_HEAD_MODE'])
        mode = mode_file.read_text().strip() if mode_file.exists() else 'baseline'
        if mode == 'baseline':
            return original(self, metadata)
        if mode != 'packed':
            raise RuntimeError(f'Unexpected r19 runtime mode {mode!r}')
        if not (self.kv_role == 'kv_producer' and self.tp_size == 4 and self.num_kv_head == 2
                and self.put_step == 2 and self.pcp_size == self.dcp_size == self.pp_size == 1
                and not self.load_async and not self.use_layerwise and self.backend_name == 'mooncake'):
            raise RuntimeError('Unexpected R18 descriptor shape')
        begin = time.monotonic_ns()
        with_context('worker_load_begin', request_ids=ids, load_async=False, tp_rank=self.tp_rank)
        if not hasattr(self, '_same_head_pair'):
            self._same_head_groups = packed.attention_groups(self, module)
            self._same_head_pair = packed.Pair(self.tp_rank)
        pair = self._same_head_pair
        capture = {'jobs': [], 'backend': self.m_store}
        db = self.token_database
        mask = db.load_mask
        own_mask = 'load_mask' in vars(db)
        ordinal = iter(ids)

        def load_mask(*args, **kwargs):
            capture['request_id'] = next(ordinal)
            return mask(*args, **kwargs)

        db.load_mask = load_mask
        token = PLAN.set(capture)
        plan_begin = time.monotonic_ns()
        try:
            result = original(self, metadata)
        finally:
            PLAN.reset(token)
            if own_mask:
                db.load_mask = mask
            else:
                del db.load_mask
        if next(ordinal, None) is not None:
            raise RuntimeError('Packed preflight did not see all scheduler load requests')
        jobs = capture['jobs']
        plans = [packed.transfer_plan(keys, addr, sizes, self._same_head_groups, self.tp_rank)
                 for _, _, keys, addr, sizes in jobs]
        own = [(job[0], plan[1], len(plan[2])) for job, plan in zip(jobs, plans, strict=True)]
        gathered = [None, None]
        guard_begin = time.monotonic_ns()
        pair.dist.all_gather_object(gathered, own, group=pair.group)
        same_requests = [row[0] for row in gathered[0]] == [row[0] for row in gathered[1]]
        with_context('r18_packed_preflight', tp_rank=self.tp_rank, requests=len(ids),
                     nonempty_gets=len(jobs), same_request_order=same_requests,
                     plan_ns=guard_begin - plan_begin, wait_ns=time.monotonic_ns() - guard_begin)
        for n, (job, plan) in enumerate(zip(jobs, plans, strict=True)):
            req, backend, keys, addr, sizes = job
            keep, digest, segments, _ = plan
            mode = 'packed' if same_requests and segments and gathered[0][n] == gathered[1][n] else 'native'
            token = CALL.set({'call_ordinal': n, 'request_id': req, 'tp_rank': self.tp_rank})
            try:
                orig_get = type(backend)._r18_physical_get
                native_bytes = sum(sum(row) for row in sizes)
                with_context('r18_restore_decision', mode=mode, native_bytes=native_bytes,
                             keys=len(keys), attention_segments=len(segments))
                if mode == 'native':
                    status = orig_get(backend, keys, addr, sizes)
                    if status is None or len(status) != len(keys) or any(status):
                        raise RuntimeError('Native partial-hit GET failed')
                    continue
                status = orig_get(backend, [keys[i] for i in keep], [addr[i] for i in keep], [sizes[i] for i in keep])
                success = status is not None and len(status) == len(keep) and not any(status)
                started = time.monotonic_ns()
                pair.exchange(digest, bytes(row[3] for row in segments), success)
                admitted = time.monotonic_ns()
                operations, device_ms = pair.copy_packed(segments)
                elapsed = time.monotonic_ns() - admitted
                received = sum(size for owner, _, size, _, _ in segments if owner != pair.rank)
                with_context('same_head_local_copy', request_ids=[req], payload_bytes=received,
                             elapsed_ns=elapsed, device_elapsed_ms=device_ms, pair_wait_ns=admitted-started,
                             operations=operations, verify=False, method='packed', copy_begin_ns=admitted,
                             copy_end_ns=admitted+elapsed)
                if not getattr(self, '_r18_first_packed_sample_checked', False):
                    omitted = next((i for i in range(len(keys)) if i not in set(keep)), None)
                    if omitted is not None:
                        sample = [(addr[omitted][0], min(sizes[omitted][0], 4096))]
                        old = packed.read_samples(sample)
                        checked = orig_get(backend, [keys[omitted]], [addr[omitted]], [sizes[omitted]])
                        if checked is None or len(checked) != 1 or any(checked) or old != packed.read_samples(sample):
                            raise RuntimeError('R18 packed partial-hit sample differs from native GET')
                        with_context('same_head_verify', request_ids=[req], sample_bytes=sample[0][1],
                                     samples=1, method='packed', extra_get_bytes=sum(sizes[omitted]))
                    self._r18_first_packed_sample_checked = True
            finally:
                CALL.reset(token)
        with_context('worker_load_end', request_ids=ids, load_async=False, tp_rank=self.tp_rank,
                     elapsed_ns=time.monotonic_ns() - begin)
        return result

    module.KVPoolWorker.start_load_kv = load


def install_runner(module):
    original_execute = module.NPUModelRunner.execute_model
    original_forward = module.NPUModelRunner._model_forward

    @wraps(original_execute)
    def execute(self, output, *args, **kwargs):
        if not output.total_num_scheduled_tokens:
            return original_execute(self, output, *args, **kwargs)
        tp_rank = self.device.index
        event = {'batch': batch_key(chunk_rows(output)), 'tp_rank': tp_rank, 'worker_enter_ns': time.monotonic_ns()}
        token = BATCH.set(event)
        try:
            return original_execute(self, output, *args, **kwargs)
        finally:
            BATCH.reset(token)

    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        started = time.monotonic_ns()
        result = original_forward(self, *args, **kwargs)
        if BATCH.get():
            with_context('p_chunk_rank_forward', forward_enter_ns=started, forward_cpu_return_ns=time.monotonic_ns())
        return result

    module.NPUModelRunner.execute_model = execute
    module.NPUModelRunner._model_forward = forward


TARGETS = {packed.WORKER: install_worker, packed.BACKEND: install_get, RUNNER: install_runner}


def install(module):
    if not getattr(module, '_issue23_r18', False):
        TARGETS[module.__name__](module)
        module._issue23_r18 = True


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
        for finder in sys.meta_path[sys.meta_path.index(self) + 1:]:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader)
                return spec
        return None


class InstallFilter(packed.BaseFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not any(isinstance(finder, Finder) for finder in sys.meta_path):
            native_observer.emit = with_context
            sys.meta_path.insert(0, Finder())
            for name in TARGETS:
                if name in sys.modules:
                    install(sys.modules[name])
