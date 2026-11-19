"""Opt-in P-only timing from HTTP intake to the EngineCore input queue."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import logging
import os
import sys
from threading import local
from time import monotonic_ns

from pd_kv_transfer_opt.profiling.prefill_chunk_observability import batch_key, chunk_rows
from pd_kv_transfer_opt.profiling.scoped_native_observability import emit


_http_span = ContextVar("admission_http_span", default=None)
_frontend_start = ContextVar("admission_frontend_start", default=None)
_result_context = ContextVar("admission_result_context", default=None)


class ObserveHTTP:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope["method"] != "POST"
                or scope["path"] not in ("/v1/chat/completions", "/v1/completions")):
            return await self.app(scope, receive, send)

        span = {"http_enter_ns": monotonic_ns()}
        token = _http_span.set(span)

        async def read_body():
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body"):
                span.setdefault("http_body_end_ns", monotonic_ns())
            return message

        try:
            return await self.app(scope, read_body, send)
        finally:
            _http_span.reset(token)


def observe_http(module):
    build_app = module.build_app

    @wraps(build_app)
    def build(*args, **kwargs):
        app = build_app(*args, **kwargs)
        if app.middleware_stack is None:
            app.add_middleware(ObserveHTTP)
        else:
            # vLLM prebuilds this stack when instrumenting the metrics endpoint.
            app.middleware_stack = ObserveHTTP(app.middleware_stack)
        return app

    module.build_app = build


def observe_render(cls, name):
    original = getattr(cls, name)

    @wraps(original)
    async def render(self, *args, **kwargs):
        span = _http_span.get()
        if span is None:
            return await original(self, *args, **kwargs)
        span["api_begin_ns"] = monotonic_ns()
        result = await original(self, *args, **kwargs)
        span["api_end_ns"] = monotonic_ns()
        return result

    setattr(cls, name, render)


def observe_chat(module):
    observe_render(module.OpenAIServingChat, "render_chat_request")


def observe_completion(module):
    observe_render(module.OpenAIServingCompletion, "render_completion_request")


def observe_async_llm(module):
    cls = module.AsyncLLM
    original_add = cls.add_request
    original_submit = cls._add_request

    @wraps(original_add)
    async def add(self, *args, **kwargs):
        token = _frontend_start.set(monotonic_ns())
        try:
            return await original_add(self, *args, **kwargs)
        finally:
            _frontend_start.reset(token)

    @wraps(original_submit)
    async def submit(self, request, *args, **kwargs):
        prepared = monotonic_ns()
        identity = {"request_id": request.request_id, "external_request_id": request.external_req_id}
        span = _http_span.get()
        if span is not None:
            emit("p_admission_http", **identity, http_enter_ns=span["http_enter_ns"],
                 http_body_end_ns=span.get("http_body_end_ns"))
            if "api_end_ns" in span:
                emit("p_admission_api", **identity, api_begin_ns=span["api_begin_ns"],
                     api_end_ns=span["api_end_ns"])
        frontend_start = _frontend_start.get()
        if frontend_start is not None:
            emit("p_admission_frontend_add", **identity, frontend_begin_ns=frontend_start,
                 frontend_end_ns=prepared)
        return await original_submit(self, request, *args, **kwargs)

    cls.add_request = add
    cls._add_request = submit


def observe_client(module):
    cls = module.AsyncMPClient
    original = cls._send_input
    add_type = module.EngineCoreRequestType.ADD

    @wraps(original)
    def send_input(self, request_type, request, *args, **kwargs):
        if request_type != add_type:
            return original(self, request_type, request, *args, **kwargs)
        start = monotonic_ns()
        result = original(self, request_type, request, *args, **kwargs)
        emit("p_admission_ipc_send", request_id=request.request_id,
             send_begin_ns=start, send_return_ns=monotonic_ns())
        return result

    cls._send_input = send_input


def observe_queue(input_queue, add_type):
    original_put, original_get = input_queue.put, input_queue.get
    locked_put, locked_get = input_queue._put, input_queue._get
    stamps = local()

    def put_under_lock(item):
        locked_put(item)
        if item[0] == add_type:
            stamps.put = monotonic_ns()

    def get_under_lock():
        item = locked_get()
        if item[0] == add_type:
            stamps.get = monotonic_ns()
        return item

    def put(item, *args, **kwargs):
        result = original_put(item, *args, **kwargs)
        if item[0] == add_type:
            emit("p_admission_enqueue", request_id=item[1][0].request_id, event_ns=stamps.put)
        return result

    def get(*args, **kwargs):
        item = original_get(*args, **kwargs)
        if item[0] == add_type:
            emit("p_admission_dequeue", request_id=item[1][0].request_id, event_ns=stamps.get)
        return item

    # Record the queue boundary under its existing lock and write the trace after release.
    input_queue._put, input_queue._get = put_under_lock, get_under_lock
    input_queue.put, input_queue.get = put, get


def observe_update(scheduler):
    original = scheduler.update_from_output

    @wraps(original)
    def update(scheduler_output, model_runner_output):
        if not scheduler_output.total_num_scheduled_tokens:
            return original(scheduler_output, model_runner_output)
        key = batch_key(chunk_rows(scheduler_output))
        start = monotonic_ns()
        result = original(scheduler_output, model_runner_output)
        emit("p_admission_result_update", batch=key, update_begin_ns=start, update_end_ns=monotonic_ns())
        return result

    scheduler.update_from_output = update


def observe_engine(module):
    cls = module.EngineCore
    proc = module.EngineCoreProc
    original_init = cls.__init__
    original_context = cls.log_iteration_details
    original_preprocess = cls.preprocess_add_request
    original_input = proc.process_input_sockets
    add_type = module.EngineCoreRequestType.ADD

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        observe_update(self.scheduler)

    @wraps(original_context)
    @contextmanager
    def result_context(self, scheduler_output):
        if scheduler_output is None or not scheduler_output.total_num_scheduled_tokens:
            with original_context(self, scheduler_output):
                yield
            return
        pending = len(self.batch_queue) if self.batch_queue is not None else 0
        key = batch_key(chunk_rows(scheduler_output))
        with original_context(self, scheduler_output):
            token = _result_context.set((key, self.batch_queue_size, pending))
            try:
                yield
            finally:
                _result_context.reset(token)

    @wraps(original_preprocess)
    def preprocess(self, request):
        decoded = monotonic_ns()
        result = original_preprocess(self, request)
        emit("p_admission_ipc_recv", request_id=request.request_id,
             decode_end_ns=decoded, preprocess_end_ns=monotonic_ns())
        return result

    @wraps(original_input)
    def input_sockets(self, *args, **kwargs):
        observe_queue(self.input_queue, add_type)
        return original_input(self, *args, **kwargs)

    cls.__init__ = init
    cls.log_iteration_details = result_context
    cls.preprocess_add_request = preprocess
    proc.process_input_sockets = input_sockets
    if os.environ.get("ISSUE23_P_ADMISSION_MODE_FILE"):
        from pd_kv_transfer_opt.profiling.prefill_admission_drain import install_on_engine

        install_on_engine(module)


def observe_executor(module):
    cls = module.FutureWrapper
    original = cls.result

    @wraps(original)
    def result(self, *args, **kwargs):
        context = _result_context.get()
        if context is None:
            return original(self, *args, **kwargs)
        batch, capacity, pending = context
        done = self.done()
        start = monotonic_ns()
        output = original(self, *args, **kwargs)
        emit("p_admission_result_wait", batch=batch, wait_begin_ns=start, wait_end_ns=monotonic_ns(),
             batch_queue_size=capacity, batch_queue_pending=pending, future_done_at_entry=done)
        return output

    cls.result = result


TARGETS = {
    "vllm.entrypoints.openai.api_server": observe_http,
    "vllm.entrypoints.openai.chat_completion.serving": observe_chat,
    "vllm.entrypoints.openai.completion.serving": observe_completion,
    "vllm.v1.engine.async_llm": observe_async_llm,
    "vllm.v1.engine.core_client": observe_client,
    "vllm.v1.engine.core": observe_engine,
    "vllm.v1.executor.multiproc_executor": observe_executor,
}


def install(module):
    if not getattr(module, "_issue23_p_admission", False):
        TARGETS[module.__name__](module)
        module._issue23_p_admission = True
        emit("p_admission_observer_installed", module=module.__name__)


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


def install_admission():
    """Install after the existing P trace filter, without adding a second native hook."""
    if os.environ.get("ISSUE23_ROLE") != "P" or os.environ.get("ISSUE23_P_ADMISSION_OBSERVABILITY") != "1":
        return
    if any(isinstance(finder, Finder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, Finder())
    for name in TARGETS:
        if name in sys.modules:
            install(sys.modules[name])


class InstallFilter(logging.Filter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_admission()
