"""Test whether admitting queued ADDs during an existing P result wait helps."""

from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
import threading
import time

from pd_kv_transfer_opt.profiling.prefill_chunk_observability import batch_key, chunk_rows
from pd_kv_transfer_opt.profiling.scoped_native_observability import emit


POLL_MS = 5
POLL_NS = POLL_MS * 1_000_000


class AdmissionDrain:
    """Run only the normal ADD handler on the engine thread during local response waits."""

    def __init__(self, engine, add_type):
        self.engine = engine
        self.add_type = add_type
        self.owner_thread = threading.get_ident()
        self.batch = None
        self.mode = None
        self.next_poll_ns = 0
        readers = getattr(engine.model_executor, "response_mqs", ())
        self.available = bool(readers) and all(reader._is_local_reader for reader in readers)
        if not self.available:
            emit("p_admission_drain_unavailable", reason="requires local multiprocess response queues")
            return
        for reader in readers:
            condition = reader._spin_condition
            condition.wait = self.wrap_wait(condition.wait)
        emit("p_admission_drain_ready", response_queues=len(readers), poll_ms=POLL_MS)

    def eligible(self, item):
        kind, payload = item
        if kind != self.add_type:
            return False
        req, _ = payload
        return (
            not req.resumable
            and not req.abort_immediately
            and not req.use_structured_output
            and req.request_id not in self.engine.scheduler.requests
        )

    def drain_one(self):
        engine = self.engine
        if not engine.is_running() or not engine.aborts_queue.empty():
            return
        inputs = engine.input_queue
        with inputs.mutex:
            if not inputs.queue or not self.eligible(inputs.queue[0]):
                return
        # The input thread only appends, and this is the same consumer as the
        # normal engine loop. Use its normal get path for queue accounting.
        item = inputs.get_nowait()
        begin_ns = time.monotonic_ns()
        engine._handle_client_request(*item)
        emit("p_admission_early_add", request_id=item[1][0].request_id, batch=self.batch,
             begin_ns=begin_ns, end_ns=time.monotonic_ns())

    def wrap_wait(self, original_wait):
        @wraps(original_wait)
        def wait(timeout_ms=None):
            if self.batch is None or threading.get_ident() != self.owner_thread:
                return original_wait(timeout_ms=timeout_ms)
            now = time.monotonic_ns()
            if now >= self.next_poll_ns:
                self.next_poll_ns = now + POLL_NS
                self.drain_one()
            # The native acquire loop still owns the deadline and cancellation.
            # Bound an idle sleep so new ADDs can be seen during a long wait.
            bounded = POLL_MS if timeout_ms is None else min(timeout_ms, POLL_MS)
            return original_wait(timeout_ms=bounded)
        return wait


def read_mode(path):
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return "blocked"
    return "drain" if value == "drain" else "blocked"


def install_on_engine(module):
    path = os.environ.get("ISSUE23_P_ADMISSION_MODE_FILE")
    if not path or getattr(module, "_issue23_admission_drain", False):
        return
    mode_file = Path(path)
    cls = module.EngineCore
    original = cls.log_iteration_details

    @wraps(original)
    @contextmanager
    def log_iteration_details(self, scheduler_output):
        with original(self, scheduler_output):
            if (type(self) is not module.EngineCoreProc or scheduler_output is None
                    or not scheduler_output.total_num_scheduled_tokens):
                yield
                return
            state = getattr(self, "_issue23_admission_drain_state", None)
            if state is None:
                state = AdmissionDrain(self, module.EngineCoreRequestType.ADD)
                self._issue23_admission_drain_state = state
            mode = read_mode(mode_file)
            if state.mode != mode:
                state.mode = mode
                emit("p_admission_drain_mode", mode=mode)
            if mode == "drain" and state.available:
                state.batch = batch_key(chunk_rows(scheduler_output))
                state.next_poll_ns = 0
            try:
                yield
            finally:
                state.batch = None

    cls.log_iteration_details = log_iteration_details
    module._issue23_admission_drain = True
