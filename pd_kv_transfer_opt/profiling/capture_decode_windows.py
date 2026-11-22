"""Collect short native NPU traces after a Decode instance reaches a measured batch width."""

import json
import time
import urllib.request


def metric(reported, metric_name):
    prefix = "vllm:" + metric_name
    return sum(float(value) for line in reported.splitlines() if line.startswith((prefix + "{", prefix + " "))
               for _, value in (line.rsplit(None, 1),))


def capture_windows(role, endpoint, case_dir, finished, request_count, capture_s=0.85):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path, method="GET", timeout=3):
        with opener.open(urllib.request.Request(endpoint + path, method=method), timeout=timeout) as reply:
            return reply.read().decode()

    def capture(window, running, forwarded):
        timing = {"window": window, "role": role, "running_at_trigger": running,
                  "decode_forwarded_at_trigger": forwarded, "request_ns": time.monotonic_ns()}
        try:
            call("/start_profile", "POST", timeout=25)
            timing["started_ns"] = time.monotonic_ns()
            finished.wait(capture_s)
        except OSError as error:
            timing["start_error"] = type(error).__name__
        finally:
            timing["stop_request_ns"] = time.monotonic_ns()
            try:
                call("/stop_profile", "POST", timeout=100)
            except OSError as error:
                timing["stop_error"] = type(error).__name__
            timing["stopped_ns"] = time.monotonic_ns()
        return timing

    def forwarded_count():
        return sum(line.count('"decode_begin"') for path in (case_dir / "proxy").glob("proxy-events-*.jsonl")
                   for line in path.read_text(encoding="utf-8").splitlines())

    start = time.monotonic()
    completed = set()
    solo_seen = False
    while not finished.wait(1.1) and len(completed) < 2:
        elapsed = time.monotonic() - start
        if elapsed < 45:
            continue
        try:
            reported = call("/metrics")
            running = metric(reported, "num_requests_running")
            waiting = metric(reported, "num_requests_waiting")
        except (OSError, ValueError):
            continue
        forwarded = forwarded_count()
        name = None
        if "multi" not in completed and elapsed < 195 and running >= 4:
            name = "multi"
        solo = running == 1 and waiting == 0 and forwarded == request_count
        if "solo" not in completed and solo and solo_seen:
            name = "solo"
        solo_seen = solo
        if name is not None:
            outcome = capture(name, running, forwarded)
            with (case_dir / f"profile-windows-{role}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(outcome) + "\n")
            completed.add(name)
