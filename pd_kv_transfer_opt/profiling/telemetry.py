"""Nonblocking host-event writer for temporary engine/connector hooks.

No device synchronization is introduced. Device execution needs separate CANN
events. A full queue drops events and marks the trace incomplete in its footer.
"""

from __future__ import annotations

from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import queue
import socket
import threading
import time

from pd_kv_transfer_opt.common import clock_anchor, output_path


class TraceWriter:
    def __init__(self, path: str | Path, run_id: str, instance_id: str,
                 rank: int = 0, capacity: int = 8192):
        self.path = output_path(path)
        self.run_id, self.instance_id, self.rank = run_id, instance_id, rank
        self.pid, self.host = os.getpid(), socket.gethostname()
        self.events = queue.Queue(maxsize=capacity)
        self.ids = itertools.count()
        self.dropped = 0
        self.failure = None
        self.closed = False
        self.stream = self.path.open("x", encoding="utf-8")
        self.stream.write(json.dumps({"type": "trace_header", "schema": "rdma-lab-events/v1",
                                      "run_id": run_id, "host": self.host, "pid": self.pid,
                                      "anchor": clock_anchor()}) + "\n")
        self.worker = threading.Thread(target=self._drain, daemon=True)
        self.worker.start()

    def emit(self, name: str, **fields) -> None:
        if self.closed:
            raise RuntimeError("TraceWriter is closed")
        event = {**fields, "type": "event", "name": name, "schema": "rdma-lab-events/v1",
                 "run_id": self.run_id, "instance_id": self.instance_id, "rank": self.rank,
                 "host": self.host, "pid": self.pid, "thread_id": threading.get_ident(),
                 "monotonic_ns": time.monotonic_ns(), "event_seq": next(self.ids)}
        try:
            self.events.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    @contextmanager
    def span(self, name: str, **fields):
        span_id = f"{self.pid}:{next(self.ids)}"
        self.emit(name, phase="begin", span_id=span_id, **fields)
        outcome = "ok"
        try:
            yield span_id
        except BaseException:
            outcome = "error"
            raise
        finally:
            self.emit(name, phase="end", span_id=span_id, outcome=outcome, **fields)

    def _drain(self):
        while True:
            item = self.events.get()
            if item is None:
                return
            try:
                self.stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            except Exception as exc:
                self.failure = repr(exc)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.events.put(None)
        self.worker.join()
        try:
            self.stream.write(json.dumps({"type": "trace_footer", "dropped_events": self.dropped,
                                          "writer_failure": self.failure, "anchor": clock_anchor()}) + "\n")
        finally:
            self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
