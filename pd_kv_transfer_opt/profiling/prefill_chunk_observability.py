"""Low-overhead scheduler, worker and NPU forward timestamps."""
from collections import deque
import functools
import hashlib
import importlib.abc
import importlib.machinery
import os
import sys
import time

from pd_kv_transfer_opt.profiling.scoped_native_observability import InstallFilter as NativeInstallFilter, emit


def chunk_rows(output):
    offsets = {req.req_id: req.num_computed_tokens for req in output.scheduled_new_reqs}
    cache = output.scheduled_cached_reqs
    offsets.update(zip(cache.req_ids, cache.num_computed_tokens, strict=True))
    return [(rid, offsets.get(rid), tokens) for rid, tokens in sorted(output.num_scheduled_tokens.items())]


def batch_key(rows):
    return hashlib.blake2b(repr(rows).encode(), digest_size=12).hexdigest()


def observe_scheduler(scheduler, event_prefix='p_chunk'):
    original_add = scheduler.add_request
    original_schedule = scheduler.schedule
    original_update = scheduler.update_from_output

    @functools.wraps(original_add)
    def add(request):
        arrival = time.monotonic_ns()
        new = request.request_id not in scheduler.requests
        result = original_add(request)
        if new:
            emit(f'{event_prefix}_arrival', request_id=request.request_id, arrival_ns=arrival,
                 prompt_tokens=request.num_prompt_tokens)
        return result

    @functools.wraps(original_schedule)
    def schedule():
        begin = time.monotonic_ns()
        output = original_schedule()
        dispatch = time.monotonic_ns()
        if output.total_num_scheduled_tokens:
            rows = chunk_rows(output)
            emit(f'{event_prefix}_dispatch', batch=batch_key(rows), rows=rows, begin_ns=begin,
                 dispatch_ns=dispatch, running=len(scheduler.running), waiting=len(scheduler.waiting),
                 budget=scheduler.max_num_scheduled_tokens, total=output.total_num_scheduled_tokens,
                 preempted=sorted(output.preempted_req_ids or ()))
        return output

    @functools.wraps(original_update)
    def update(scheduler_output, model_runner_output):
        ready = time.monotonic_ns()
        if scheduler_output.total_num_scheduled_tokens:
            rows = chunk_rows(scheduler_output)
            emit(f'{event_prefix}_ready', batch=batch_key(rows), ready_ns=ready, rows=rows)
        return original_update(scheduler_output, model_runner_output)

    scheduler.add_request = add
    scheduler.schedule = schedule
    scheduler.update_from_output = update
    emit(f'{event_prefix}_scheduler_attached', schedule_module=original_schedule.__module__,
         scheduler_class=type(scheduler).__name__)


def observe_engine(module, event_prefix='p_chunk'):
    cls = module.EngineCore
    original_init = cls.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        observe_scheduler(self.scheduler, event_prefix)

    cls.__init__ = init


def observe_runner(module, event_prefix='p_chunk'):
    cls = module.NPUModelRunner
    original_execute = cls.execute_model
    original_forward = cls._model_forward
    original_sample = cls.sample_tokens
    pending = deque()
    active = None
    timing_available = True
    graph_setting = ('ISSUE23_D_GRAPH_REPLAY_OBSERVABILITY' if event_prefix == 'd_step'
                     else 'ISSUE23_P_GRAPH_REPLAY_OBSERVABILITY')
    observe_graph = os.environ.get(graph_setting) == '1'
    replay_calls = 0
    if observe_graph:
        graph_class = module.torch.npu.NPUGraph
        original_replay = graph_class.replay

        @functools.wraps(original_replay)
        def replay(self):
            nonlocal replay_calls
            result = original_replay(self)
            if active is not None:
                replay_calls += 1
            return result

        graph_class.replay = replay

    def disable_timing(error):
        nonlocal timing_available
        timing_available = False
        pending.clear()
        emit(f'{event_prefix}_timing_unavailable', error=type(error).__name__, message=str(error)[:300])

    def flush():
        try:
            while pending and pending[0][1].query():
                first, last, key, entered, begin, submitted = pending.popleft()
                emit(f'{event_prefix}_device', batch=key, worker_enter_ns=entered, forward_enter_ns=begin,
                     forward_cpu_return_ns=submitted, forward_device_ms=first.elapsed_time(last))
        except (RuntimeError, AttributeError) as error:
            disable_timing(error)

    @functools.wraps(original_execute)
    def execute(self, scheduler_output, *args, **kwargs):
        nonlocal active
        if self.device.index != 0 or not scheduler_output.total_num_scheduled_tokens:
            return original_execute(self, scheduler_output, *args, **kwargs)
        flush()
        active = (batch_key(chunk_rows(scheduler_output)), time.monotonic_ns())
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            active = None

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        if active is None:
            return original_forward(self, *args, **kwargs)
        key, entered = active
        if observe_graph:
            graph_mode = module.get_forward_context().cudagraph_runtime_mode.name
            padded_tokens = args[0] if args else kwargs['num_tokens_padded']
            replay_before = replay_calls
        first = None
        if timing_available:
            try:
                first = module.torch.npu.Event(enable_timing=True)
                first.record()
            except (RuntimeError, AttributeError) as error:
                first = None
                disable_timing(error)
        begin = time.monotonic_ns()
        result = original_forward(self, *args, **kwargs)
        submitted = time.monotonic_ns()
        if observe_graph:
            emit(f'{event_prefix}_graph', batch=key, runtime_mode=graph_mode, padded_tokens=int(padded_tokens),
                 aclgraph_replays=replay_calls - replay_before)
        if first is not None:
            try:
                last = module.torch.npu.Event(enable_timing=True)
                last.record()
                pending.append((first, last, key, entered, begin, submitted))
            except (RuntimeError, AttributeError) as error:
                disable_timing(error)
        if first is None or not timing_available:
            emit(f'{event_prefix}_device', batch=key, worker_enter_ns=entered, forward_enter_ns=begin,
                 forward_cpu_return_ns=submitted, forward_device_ms=None)
        return result

    @functools.wraps(original_sample)
    def sample(self, *args, **kwargs):
        result = original_sample(self, *args, **kwargs)
        if self.device.index == 0:
            flush()
        return result

    cls.execute_model = execute
    cls._model_forward = forward
    cls.sample_tokens = sample


TARGETS = {
    'vllm.v1.engine.core': observe_engine,
    'vllm_ascend.worker.model_runner_v1': observe_runner,
}


class Loader:
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        install(module)


def install(module):
    if not getattr(module, '_issue23_prefill_chunks', False):
        event_prefix = 'd_step' if os.environ.get('ISSUE23_ROLE') == 'D' else 'p_chunk'
        TARGETS[module.__name__](module, event_prefix)
        module._issue23_prefill_chunks = True
        emit(f'{event_prefix}_observer_installed', module=module.__name__)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in TARGETS:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader:
                spec.loader = Loader(spec.loader)
            return spec


_installed = False


class InstallFilter(NativeInstallFilter):
    def __init__(self, *args, **kwargs):
        global _installed
        super().__init__(*args, **kwargs)
        role = os.environ.get('ISSUE23_ROLE')
        enabled = role == 'P' or (role == 'D' and os.environ.get('ISSUE23_D_STEP_OBSERVABILITY') == '1')
        if not enabled or _installed:
            return
        sys.meta_path.insert(0, Finder())
        for name in TARGETS:
            if name in sys.modules:
                install(sys.modules[name])
        _installed = True
