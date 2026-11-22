"""Sample native Decode steps on the engine and on each TP worker."""

from collections import deque
from functools import wraps
import logging
import sys
from time import monotonic_ns

from pd_kv_transfer_opt.profiling.prefill_chunk_observability import batch_key, chunk_rows
from pd_kv_transfer_opt.profiling.scoped_native_observability import emit, span


ENGINE = "vllm.v1.engine.core"
RUNNER = "vllm_ascend.worker.model_runner_v1"
STRIDE = 16


def sampled_rows(output):
    offsets = output.scheduled_cached_reqs.num_computed_tokens
    if (offsets and offsets[0] % STRIDE == 0 and output.total_num_scheduled_tokens
            == len(output.num_scheduled_tokens)):
        return chunk_rows(output)
    return None


def observe_engine(module):
    cls = module.EngineCore
    initial = cls.__init__

    @wraps(initial)
    def init(self, *args, **kwargs):
        initial(self, *args, **kwargs)
        scheduler = self.scheduler
        schedule = scheduler.schedule
        update = scheduler.update_from_output
        previous_dispatch = None

        @wraps(schedule)
        def dispatch():
            nonlocal previous_dispatch
            begin = monotonic_ns()
            output = schedule()
            end = monotonic_ns()
            if output.total_num_scheduled_tokens:
                if rows := sampled_rows(output):
                    emit("d_step_dispatch", batch=batch_key(rows), rows=rows, schedule_begin_ns=begin,
                         dispatch_ns=end, previous_dispatch_ns=previous_dispatch,
                         running=len(scheduler.running), waiting=len(scheduler.waiting))
                previous_dispatch = end
            return output

        @wraps(update)
        def publish(output, model_result):
            rows = sampled_rows(output)
            begin = monotonic_ns()
            result = update(output, model_result)
            if rows:
                emit("d_step_result", batch=batch_key(rows), result_ns=begin, update_end_ns=monotonic_ns())
            return result

        scheduler.schedule = dispatch
        scheduler.update_from_output = publish
        emit("d_step_attached", batch_queue_size=self.batch_queue_size, async_scheduling=self.async_scheduling,
             step_sample_stride=STRIDE)

    cls.__init__ = init


def observe_runner(module):
    cls = module.NPUModelRunner
    execute = cls.execute_model
    forward = cls._model_forward
    sample = cls.sample_tokens
    replay = module.torch.npu.NPUGraph.replay
    pending = deque()
    previous_enter = None
    active = None
    awaiting_sample = None
    timing_available = True

    def disable_timing(error):
        nonlocal timing_available
        if timing_available:
            emit("d_step_device_timing_unavailable", error=type(error).__name__)
        timing_available = False

    def mark():
        if timing_available:
            try:
                event = module.torch.npu.Event(enable_timing=True)
                event.record()
                return event
            except (RuntimeError, AttributeError) as error:
                disable_timing(error)
        return None

    def flush():
        while pending:
            fields, (first, forward_end, sample_end) = pending[0]
            if first is not None and forward_end is not None and sample_end is not None and timing_available:
                try:
                    if not sample_end.query():
                        break
                    fields["forward_device_ms"] = first.elapsed_time(forward_end)
                    fields["after_forward_stream_ms"] = forward_end.elapsed_time(sample_end)
                except (RuntimeError, AttributeError) as error:
                    disable_timing(error)
            pending.popleft()
            emit("d_step_worker", **fields)

    @wraps(replay)
    def graph_replay(self):
        result = replay(self)
        if active is not None:
            active[0]["graph_replays"] += 1
        return result

    @wraps(execute)
    def run(self, output, *args, **kwargs):
        nonlocal previous_enter, active, awaiting_sample
        if pending:
            flush()
        if not output.total_num_scheduled_tokens:
            return execute(self, output, *args, **kwargs)
        enter = monotonic_ns()
        prior, previous_enter = previous_enter, enter
        if not (rows := sampled_rows(output)):
            return execute(self, output, *args, **kwargs)
        fields = {"batch": batch_key(rows), "rank": self.device.index, "worker_enter_ns": enter,
                  "previous_worker_enter_ns": prior, "graph_replays": 0}
        marks = [None, None]
        active = fields, marks
        try:
            result = execute(self, output, *args, **kwargs)
            fields["execute_return_ns"] = monotonic_ns()
            if result is None:
                awaiting_sample = active
            else:
                pending.append((fields, (*marks, mark())))
            return result
        finally:
            active = None

    @wraps(forward)
    def model_forward(self, *args, **kwargs):
        if active is None:
            return forward(self, *args, **kwargs)
        fields, marks = active
        fields["graph_mode"] = module.get_forward_context().cudagraph_runtime_mode.name
        fields["padded_tokens"] = int(args[0] if args else kwargs["num_tokens_padded"])
        marks[0] = mark()
        fields["forward_enter_ns"] = monotonic_ns()
        with span("decode_forward_" + fields["batch"]):
            result = forward(self, *args, **kwargs)
        fields["forward_return_ns"] = monotonic_ns()
        marks[1] = mark()
        return result

    @wraps(sample)
    def sample_tokens(self, *args, **kwargs):
        nonlocal awaiting_sample
        if awaiting_sample is None:
            return sample(self, *args, **kwargs)
        fields, marks = awaiting_sample
        awaiting_sample = None
        fields["sample_enter_ns"] = monotonic_ns()
        with span("decode_sample_" + fields["batch"]):
            result = sample(self, *args, **kwargs)
        fields["sample_return_ns"] = monotonic_ns()
        pending.append((fields, (*marks, mark())))
        flush()
        return result

    module.torch.npu.NPUGraph.replay = graph_replay
    cls.execute_model = run
    cls._model_forward = model_forward
    cls.sample_tokens = sample_tokens


TARGETS = {ENGINE: observe_engine, RUNNER: observe_runner}


def install(module):
    if not getattr(module, "_pd_decode_step_observed", False):
        TARGETS[module.__name__](module)
        module._pd_decode_step_observed = True


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


class InstallFilter(logging.Filter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not any(isinstance(finder, Finder) for finder in sys.meta_path):
            sys.meta_path.insert(0, Finder())
            for name in TARGETS:
                if name in sys.modules:
                    install(sys.modules[name])

    def filter(self, record):
        return True
