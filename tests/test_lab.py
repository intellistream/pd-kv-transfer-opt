from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]

from pd_kv_transfer_opt.experiments.analyze import summarize
from pd_kv_transfer_opt.experiments.build_matrix import build
from pd_kv_transfer_opt.profiling.analyze_multiwave import aligned_tx_intervals, cap_pressure, supply_time
from pd_kv_transfer_opt.profiling.check_trace import audit
from pd_kv_transfer_opt.common import read_json, read_jsonl
from pd_kv_transfer_opt.experiments.compare import compare
from pd_kv_transfer_opt.experiments.serve import model_commands, read_deployment, store_pool_layout
from pd_kv_transfer_opt.simulator.geometry import geometry
from pd_kv_transfer_opt.experiments.loadgen import drive, request_one, validate_workload
from pd_kv_transfer_opt.simulator.lifecycle import Registry
from pd_kv_transfer_opt.simulator.network_sim import Flow, max_min_rates, simulate
from pd_kv_transfer_opt.profiling.telemetry import TraceWriter


class MultiwaveProfilingTests(unittest.TestCase):
    def test_port_rates_follow_each_ports_own_timestamps(self):
        def port(number, second, gigabytes):
            return {"port": number, "monotonic_ns": second * 10**9,
                    "counters": {"mac_tx_total_oct_num": gigabytes * 10**9}}

        samples = [
            {"monotonic_ns": 0, "ports": [port(0, 0, 0), port(1, 2, 0)]},
            {"monotonic_ns": 3 * 10**9, "ports": [port(0, 5, 50), port(1, 7, 0)]},
            {"monotonic_ns": 8 * 10**9, "ports": [port(0, 10, 50), port(1, 12, 50)]},
        ]
        intervals = list(aligned_tx_intervals(samples, 0))
        self.assertEqual(intervals, [(2, 5, 80, (80, 0)), (5, 7, 0, (0, 0)), (7, 10, 80, (0, 80))])
        self.assertEqual(cap_pressure(intervals, 9, 90), {
            "D_cap_measurement_window_s": 7,
            "D_MAC_TX_at_least_85pct_cap_s": 5,
        })

    def test_ready_slot_deficit_distinguishes_handoff_from_d_wait(self):
        requests = [
            {"P_begin_s": 0, "P_HTTP_return_s": 4, "D_KV_ready_s": 5, "D_end_s": 8},
            {"P_begin_s": 0, "P_HTTP_return_s": 2, "D_KV_ready_s": 3, "D_end_s": 6},
        ]
        result = supply_time(requests, 0, 8, decode_slots=1)
        self.assertEqual(result["D_below_slots_while_P_pending_s"], 3)
        self.assertEqual(result["D_handoff_below_slots_while_P_pending_s"], 2)


class DeploymentTests(unittest.TestCase):
    def test_both_roles_contribute_eight_gib_per_worker(self):
        config = read_deployment(ROOT / "configs" / "ascend-2p2d-decode-apc.example.json")
        layout = store_pool_layout(config)
        self.assertEqual(layout["model_workers_by_role"], {"P": 8, "D": 8})
        self.assertEqual(layout["capacity_bytes_by_role"], {"P": 64 * 1024**3, "D": 64 * 1024**3})
        self.assertEqual(layout["total_capacity_bytes"], 128 * 1024**3)
        for role in ("P", "D"):
            store, _ = model_commands(config, role, ROOT / "runs")
            self.assertEqual(store["global_segment_size"], 8 * 1024**3)
            self.assertEqual(store["local_buffer_size"], 32 * 1024**2)
            self.assertIs(store["preferred_segment"], True)
            self.assertIs(store["prefer_alloc_in_same_node"], True)

        for old_size in (0, 16 * 1024**3):
            with self.subTest(old_size=old_size):
                stale = dict(config, store_pool=dict(config["store_pool"], global_segment_size=old_size))
                with self.assertRaisesRegex(ValueError, "store_pool must match"):
                    model_commands(stale, "D", ROOT / "runs")

    def test_deployment_uses_separate_lookup_ids_and_opt_in_profiling(self):
        config = read_deployment(ROOT / "configs" / "ascend-2p2d.example.json")
        for role in ("P", "D"):
            _, commands = model_commands(config, role, ROOT / "runs")
            lookup_ids = []
            for command in commands:
                argv = command["argv"]
                transfer = json.loads(argv[argv.index("--kv-transfer-config") + 1])
                store = transfer["kv_connector_extra_config"]["connectors"][1]
                lookup_ids.append(store["kv_connector_extra_config"]["lookup_rpc_port"])
                prefix_flag = "--enable-prefix-caching" if role == "P" else "--no-enable-prefix-caching"
                self.assertIn(prefix_flag, argv)
                self.assertNotIn("--profiler-config", argv)
                self.assertNotIn("--enforce-eager", argv)
            self.assertEqual(lookup_ids, list(map(str, config[role]["api_ports"])))

        config["P"]["max_num_seqs"] = 32
        del config["P"]["enable_prefix_caching"]
        _, commands = model_commands(config, "P", ROOT / "runs", observe=True)
        argv = commands[0]["argv"]
        self.assertEqual(argv[argv.index("--max-num-seqs") + 1], "32")
        self.assertNotIn("--no-enable-prefix-caching", argv)
        self.assertIn("--profiler-config", argv)


class NetworkTests(unittest.TestCase):
    def test_single_and_zero_byte(self):
        result = simulate([Flow("a", 0, 100, ("pd",)), Flow("z", 1, 0, ("pd",))], {"pd": 100})
        self.assertEqual([row["complete_s"] for row in result["flows"]], [1, 1])

    def test_simultaneous_flows_share_bandwidth(self):
        result = simulate([Flow("a", 0, 100, ("pd",)), Flow("b", 0, 100, ("pd",))], {"pd": 100})
        self.assertEqual([row["complete_s"] for row in result["flows"]], [2, 2])

    def test_arrival_changes_active_flow_rate(self):
        result = simulate([Flow("a", 0, 100, ("pd",)), Flow("b", .5, 100, ("pd",))], {"pd": 100})
        self.assertEqual([row["complete_s"] for row in result["flows"]], [1.5, 2])

    def test_full_duplex_does_not_share_an_invented_capacity(self):
        result = simulate([Flow("a", 0, 100, ("pd",)), Flow("b", 0, 100, ("dp",))], {"pd": 100, "dp": 100})
        self.assertEqual([row["complete_s"] for row in result["flows"]], [1, 1])

    def test_shared_endpoint_couples_directions(self):
        result = simulate([Flow("a", 0, 100, ("pd", "mem")), Flow("b", 0, 100, ("dp", "mem"))],
                          {"pd": 100, "dp": 100, "mem": 100})
        self.assertEqual([row["complete_s"] for row in result["flows"]], [2, 2])

    def test_progressive_fill_multiple_bottlenecks(self):
        rates = max_min_rates([Flow("a", 0, 1, ("x", "y")), Flow("b", 0, 1, ("x",)),
                               Flow("c", 0, 1, ("y",))], {"x": 10, "y": 100})
        self.assertEqual(rates, {"a": 5, "b": 5, "c": 95})

    def test_fifo_limit_is_not_a_bandwidth_penalty(self):
        result = simulate([Flow("a", 0, 100, ("pd",)), Flow("b", 0, 100, ("pd",))],
                          {"pd": 100}, {"default": 1})
        self.assertEqual([row["complete_s"] for row in result["flows"]], [1, 2])
        self.assertEqual([row["queue_wait_s"] for row in result["flows"]], [0, 1])

    def test_invalid_inputs(self):
        for flows, capacities in [([Flow("a", 0, 1, ("pd",))], {"pd": 0}),
                                   ([Flow("a", 0, -1, ("pd",))], {"pd": 1}),
                                   ([Flow("a", 0, 1, ("missing",))], {"pd": 1})]:
            with self.subTest(flows=flows), self.assertRaises(ValueError):
                simulate(flows, capacities)


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if body["prompt"] == "empty":
            self.wfile.write(b"data: [DONE]\n\n")
            return
        first = {"choices": [{"index": 0, "text": "ab", "token_ids": [10, 11], "finish_reason": None}]}
        self.wfile.write(("data: " + json.dumps(first) + "\n\n").encode())
        self.wfile.flush()
        if body["prompt"] == "slow":
            time.sleep(.05)
        last = {"choices": [{"index": 0, "text": "c", "token_ids": [12], "finish_reason": "length"}]}
        usage = {"choices": [], "usage": {"completion_tokens": 3, "prompt_tokens": 2}}
        for frame in (last, usage):
            self.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), StreamHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_multi_token_chunk_and_empty_http_200(self):
        events = []
        row = {"id": "r", "prompt": "ok", "offered_s": 0, "ready_s": 0, "max_tokens": 3}
        record = request_one(row, self.endpoint, "mock", time.monotonic(), 2, True, False, events.append)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["streamed_token_ids"], 3)
        self.assertEqual(len(events), 2)
        failed = request_one({**row, "prompt": "empty"}, self.endpoint, "mock", time.monotonic(), 2, True, False, events.append)
        self.assertEqual(failed["status"], "error")

    def test_successor_waits_for_full_decode(self):
        rows = [{"id": "a", "prompt": "slow", "offered_s": 0, "max_tokens": 3},
                {"id": "b", "predecessor": "a", "prompt": "ok", "offered_s": 0, "max_tokens": 3, "tool_gap_s": .01}]
        validate_workload(rows)
        events = []
        asyncio.run(drive(rows, self.endpoint, "mock", 2, 2, True, False, events.append))
        ends = {row["id"]: row for row in events if row["type"] == "request_end"}
        self.assertGreaterEqual(ends["b"]["dispatch_s"], ends["a"]["end_s"] + .01)
        self.assertGreater(ends["a"]["end_s"], ends["a"]["first_s"])

    def test_open_loop_queue_is_preserved(self):
        rows = [{"id": key, "prompt": "slow", "offered_s": 0, "max_tokens": 3} for key in ("a", "b")]
        events = []
        asyncio.run(drive(rows, self.endpoint, "mock", 1, 2, True, False, events.append))
        ends = [row for row in events if row["type"] == "request_end"]
        self.assertEqual([row["ready_s"] for row in ends], [0, 0])
        self.assertGreater(ends[1]["dispatch_s"], ends[0]["first_s"])


class AccountingTests(unittest.TestCase):
    def test_finite_throughput_uses_planned_arrival_and_drain(self):
        manifest = {"run_id": "unit", "window_start": 0, "seconds": 6, "selected_requests": 2}
        first = {"type": "request_end", "id": "a", "status": "ok", "offered_s": 5,
                 "ready_s": 5, "dispatch_s": 5.5, "first_s": 6, "end_s": 7, "completion_tokens": 3}
        last = dict(first, id="b", offered_s=5.1, ready_s=5.1, end_s=9, completion_tokens=9)
        result = summarize([last, first], manifest)["finite_cohort"]
        self.assertEqual(result["offer_to_last_end_s"], 4)
        self.assertEqual(result["output_tokens"], 12)
        self.assertEqual(result["output_tokens_per_s"], 3)
        self.assertIsNone(summarize([last], manifest)["finite_cohort"]["output_tokens_per_s"])

    def test_window_boundaries_and_unknown_token_counts(self):
        manifest = {"run_id": "unit", "window_start": 0, "seconds": 1, "selected_requests": 1}
        end = {"type": "request_end", "id": "a", "status": "ok", "offered_s": 0,
               "ready_s": 0, "dispatch_s": 0, "first_s": .1, "end_s": 2,
               "completion_tokens": 3, "streamed_token_ids": 3, "all_content_chunks_have_token_ids": True}
        rows = [{"type": "stream_chunk", "id": "a", "at_s": .1, "n_tokens": 2},
                {"type": "stream_chunk", "id": "a", "at_s": 1, "n_tokens": 1}, end]
        result = summarize(rows, manifest)
        self.assertEqual(result["window"]["client_delivered_tokens_s"], 2)
        self.assertEqual(result["window"]["completed_request_tokens_s_boundary_biased"], 0)
        slo = summarize(rows, manifest, 3, 3)["window"]
        self.assertEqual(slo["slo_goodput_requests_s"], 0)
        self.assertEqual(slo["offered_cohort_slo_attainment"], 1)
        rows[0]["n_tokens"] = None
        end["all_content_chunks_have_token_ids"] = False
        self.assertIsNone(summarize(rows, manifest)["window"]["client_delivered_tokens_s"])

    def test_trace_pairing_and_drops(self):
        temporary_root = ROOT / "runs" / "test-tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            path = Path(directory) / "events.jsonl"
            with TraceWriter(path, "test", "P0") as writer:
                with writer.span("fetch", request_id="a", bytes=10):
                    pass
            rows = read_jsonl(path)
            self.assertTrue(audit(rows)["valid_host_span_accounting"])
            rows[-1]["dropped_events"] = 1
            self.assertFalse(audit(rows)["valid_host_span_accounting"])

    def test_matrix_balance_and_reproducibility(self):
        a = build(2, ["heldout"], 2, 7)
        self.assertEqual(a, build(2, ["heldout"], 2, 7))
        self.assertEqual(len(a["runs"]), 30)
        for block in range(6):
            self.assertEqual({row["arm"] for row in a["runs"] if row["block_id"] == block},
                             {"base-before", "base-after", "scheduler", "system", "joint"})

    def test_model_geometry_and_run_bootstrap(self):
        result = geometry(read_json(ROOT / "configs" / "qwen35-config.json"), 4, 32768)
        self.assertEqual(result["logical_attention_bytes_per_token"], 20480)
        self.assertEqual(result["estimated_aggregate_stored_attention_bytes_per_token"], 40960)
        self.assertEqual(result["logical_recurrent_bytes_per_snapshot"], 60 * 1024**2)
        result = compare([{"block_id": str(i), "control": 100, "treatment": 110} for i in range(5)], samples=100)
        self.assertAlmostEqual(result["treatment_relative_change_percent"], 10)


class LifecycleTests(unittest.TestCase):
    def test_destination_backpressure_does_not_pin_source_forever(self):
        state = Registry({"P": 100, "D": 50})
        state.allocate("p", "kv", 1, "P", 60)
        state.publish("p")
        self.assertIsNone(state.start_copy("f", "p", "d", "D"))
        self.assertEqual(state.allocations["p"].pins, 0)
        self.assertEqual(state.used("D"), 0)

    def test_inflight_capacity_source_lifetime_and_checkpoint_bundle(self):
        state = Registry({"P": 100, "D": 100})
        for key in ("attention", "ssm"):
            state.allocate("p-" + key, key, 7, "P", 40)
            state.publish("p-" + key)
            state.start_copy(key, "p-" + key, "d-" + key, "D")
        self.assertEqual(state.used("D"), 80)
        with self.assertRaises(ValueError):
            state.free("p-attention")
        state.finish_copy("attention")
        expected = {"d-attention": ("attention", 7), "d-ssm": ("ssm", 7)}
        self.assertIsNone(state.acquire_bundle(expected))
        state.finish_copy("ssm")
        bundle = state.acquire_bundle(expected)
        with self.assertRaises(ValueError):
            state.free("d-ssm")
        state.release_bundle(bundle)
        with self.assertRaises(KeyError):
            state.release_bundle(bundle)
        state.free("d-ssm")
        self.assertEqual(state.used("D"), 40)

    def test_aborted_copy_releases_only_its_allocation(self):
        state = Registry({"P": 100, "D": 100})
        state.allocate("old", "same-hash", 1, "P", 40)
        state.publish("old")
        state.start_copy("f", "old", "destination", "D")
        state.allocate("new", "same-hash", 2, "P", 40)
        state.publish("new")
        state.finish_copy("f", success=False)
        self.assertEqual(state.used("D"), 0)
        self.assertEqual(state.allocations["old"].pins, 0)
        self.assertEqual(state.allocations["new"].pins, 0)
        with self.assertRaises(ValueError):
            state.finish_copy("f")

    def test_wrong_hybrid_checkpoint_is_rejected(self):
        state = Registry({"D": 100})
        state.allocate("d", "ssm", 3, "D", 40)
        state.publish("d")
        with self.assertRaises(ValueError):
            state.acquire_bundle({"d": ("ssm", 4)})


if __name__ == "__main__":
    unittest.main()
