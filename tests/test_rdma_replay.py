"""CPU regressions for cap, byte conservation and successor timing."""

import unittest

from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BYTES, STATE_BYTES
from pd_kv_transfer_opt.simulator.rdma_audit import compare_decode_inputs
from pd_kv_transfer_opt.simulator.rdma_replay import replay_egress, run_bandwidth
from pd_kv_transfer_opt.simulator.terminal_magnitude import magnitude_requirements, source_options, wave_times
from pd_kv_transfer_opt.simulator.terminal_pressure import overlap, peak_window
from pd_kv_transfer_opt.simulator.terminal_supply import ColdPrefixes, protect_order, run_successors, scored_prefill, successor_jobs
from pd_kv_transfer_opt.simulator.terminal_wave import decode_deliveries, prefill_wave, sample_ports


class RdmaReplay(unittest.TestCase):
    def test_local_earlier_root_covers_only_shared_attention_in_a_cross_P_read(self):
        ledger = {key: {"id": key, "task": "t", "step_id": index, "prompt_tokens": 4096,
                        "source_prompt_lcp_to_previous": shared}
                  for index, (key, shared) in enumerate((("r0", 0), ("s0", 3072), ("r1", 1024), ("s1", 3072)))}
        pairs = [{"task": "t", "root": root, "successor": child, "checkpoint_blocks": 3,
                  "potential_GET_bytes": 3 * ATTENTION_BYTES + STATE_BYTES} for root, child in (("r0", "s0"), ("r1", "s1"))]
        placement = {"r0": 0, "r1": 1}
        coverage = ColdPrefixes(ledger).earlier_root_attention(pairs, placement)
        self.assertEqual(coverage, {"s0": ATTENTION_BYTES, "s1": ATTENTION_BYTES})
        times = {"s0": 1, "s1": 2}
        jobs = successor_jobs(times, placement, {"s0": 1, "s1": 1}, times, {"s0": 100, "s1": 100},
                              {"wire_fraction": 1}, {row["successor"]: row for row in pairs}, ledger, {"s0": 1, "s1": 2},
                              local_attention=coverage)
        self.assertEqual(jobs["s0"]["GET_GB"], (2 * ATTENTION_BYTES + STATE_BYTES) / 1e9)
        self.assertEqual(jobs["s1"]["GET_GB"], 0)

    def test_overlapping_root_stays_in_decode_but_is_not_a_successor_delivery(self):
        jobs = {"successor": {"P": "P0", "P_admit_s": 1.0, "GET_GB": 12.5, "suffix_tokens": 0,
                              "output_tokens": 2, "decode_score": 1}}
        roots = ({"root": {"P_admit_s": 0, "output_tokens": 200, "decode_score": 2}}, {"root": 0.1})
        state = {"successor_handoff_s": .1, "root_handoff_s": .2, "sample_period_s": .2, "common_D_tail_residual_s": 0}
        prefill = {"GET_batch_width": 1, "first_GET_launch_s": 0, "batch_nonwire_s": 0, "warm_tokens_s": 1,
                   "rail_efficiency": 1}
        decode = {"first_sequence_period_s": .1, "additional_sequence_period_s": .0}
        run = run_successors(jobs, 100, [1 / 8] * 8, 1, state, prefill, decode, root_work=roots)
        self.assertEqual(run["sim"]["output_tokens"], 202)
        self.assertEqual(run["report"]["D_first_tokens_in_first_10_s"], 1)
        self.assertEqual(run["report"]["tail"]["id"], "root")
        self.assertAlmostEqual(run["report"]["first_D_token_s"], 2.1)

    def test_native_idle_prefill_uses_ordinal_after_credits_return(self):
        gates = {"a": 0, "b": .1, "c": .2, "d": 1}
        weights = {"a": 500, "b": 200, "c": 100, "d": 500}
        self.assertEqual(scored_prefill(gates, weights, {"b": .3, "c": .4, "a": .5}), {"a": 0, "b": 1, "c": 1, "d": 0})

    def test_ingress_protection_changes_only_gaps_that_can_invert(self):
        nominal = {"a": 0, "b": .022, "c": .1}
        profiles = [{"a": .038, "b": 0, "c": 0}, {"a": 0, "b": .006, "c": 0}]
        offers, admits = protect_order(list(nominal), nominal, nominal, profiles)
        self.assertEqual(offers, admits)
        self.assertEqual(admits["a"], 0)
        self.assertAlmostEqual(admits["b"], .040)
        self.assertAlmostEqual(admits["c"] - admits["b"], .078)

    def test_synchronous_prefill_readers_share_the_rails_and_sample_conserves_wire(self):
        jobs = {f"p{i}": {"P": f"P{i}", "P_admit_s": 0.0, "GET_GB": 12.5, "suffix_tokens": 0}
                for i in range(2)}
        service = {"GET_batch_width": 1, "rail_efficiency": 1, "first_GET_launch_s": 0,
                   "batch_nonwire_s": 0, "warm_tokens_s": 1}
        for cap, expected in ((100, 2), (200, 1)):
            ends, trace, _, wire = prefill_wave(jobs, cap, [1/8] * 8, service)
            self.assertEqual(ends, {"p0": expected, "p1": expected})
            self.assertEqual(wire, [25/8] * 8)
            for port in range(8):
                sampled = sample_ports(trace, 0, .35)
                self.assertAlmostEqual(sum((b-a) * rates[port] / 8 for a,b,rates in sampled), wire[port])

    def test_first_singleton_load_can_reply_before_the_bulk_batch(self):
        jobs = {str(i): {"P": "P0", "P_admit_s": 0.0, "GET_GB": 12.5, "suffix_tokens": 0} for i in range(4)}
        service = {"GET_batch_width": 3, "rail_efficiency": 1, "first_GET_launch_s": 0,
                   "batch_nonwire_s": 0, "warm_tokens_s": 1}
        old, _, _, old_wire = prefill_wave(jobs, 100, [1/8] * 8, service)
        first, _, calls, wire = prefill_wave(jobs, 100, [1/8] * 8, {**service, "first_GET_batch_width": 1})
        self.assertEqual(old["0"], 3)
        self.assertEqual(first, {"0": 1, "1": 4, "2": 4, "3": 4})
        self.assertEqual([len(call["requests"]) for call in calls], [1, 3])
        self.assertEqual(wire, old_wire)

    def test_native_decode_score_counts_pending_handshake_and_releases_at_EOS(self):
        jobs = {key: {"P_admit_s": t, "decode_score": score, "output_tokens": length}
                for key, t, score, length in (("a", 0, 300, 2), ("b", .1, 200, 10), ("c", .7, 300, 2))}
        replies = {key: row["P_admit_s"] for key, row in jobs.items()}
        step = {"first_sequence_period_s": .1, "additional_sequence_period_s": 0.0}
        events = decode_deliveries(jobs, replies, step, 16, .5)["per_request"]
        self.assertEqual({key: row["D"] for key,row in events.items()}, {"a": "D0", "b": "D1", "c": "D0"})

    def test_one_hot_rail_extends_the_window_and_preserves_each_ports_bytes(self):
        epochs = [(0.0, 2.0, (20.0, 4.0, 0, 0, 0, 0, 0, 0))]
        _, slow = replay_egress(epochs, 100)
        _, fast = replay_egress(epochs, 200)
        self.assertEqual(slow["D_port_TX_bytes"], fast["D_port_TX_bytes"])
        self.assertEqual(slow["D_port_TX_bytes"][:2], [5e9, 1e9])
        self.assertAlmostEqual(slow["capacity_wait_on_shared_egress_clock_s"], 1.2)
        self.assertEqual(fast["capacity_wait_on_shared_egress_clock_s"], 0)

    def test_within_sample_bursts_change_wait_without_creating_bytes(self):
        epochs = [(0.0, 2.0, (10.0, 0, 0, 0, 0, 0, 0, 0))]
        _, uniform = replay_egress(epochs, 100)
        _, burst = replay_egress(epochs, 100, use_short_bursts=True)
        self.assertEqual(uniform["D_port_TX_bytes"], burst["D_port_TX_bytes"])
        self.assertEqual(uniform["capacity_wait_on_shared_egress_clock_s"], 0)
        self.assertAlmostEqual(burst["capacity_wait_on_shared_egress_clock_s"], .8)

    def test_prefill_wait_propagates_through_eos_to_a_successor(self):
        def request(prior, ready, output_tokens, role):
            return {"D": role, "predecessor": prior, "offered_s": 0, "ready_s": ready, "tool_gap_s": 0,
                    "P_start_s": ready, "P_end_s": ready + 1, "first_s": ready + 1, "end_s": ready + 1.1,
                    "output_tokens": output_tokens, "phases_s": {"ready_to_P": 0, "P_HTTP": 1}}

        requests = {"root": request(None, 0, 2, "D0"), "successor": request("root", 1.1, 1, "D1")}
        service = {"first_sequence_period_s": .1, "additional_sequence_period_s": 0}
        intervals = [(0, 1, (25, 0, 0, 0, 0, 0, 0, 0)), (1, 2.1, (0,) * 8)]
        slow = run_bandwidth(requests, service, intervals, 100, 16)
        fast = run_bandwidth(requests, service, intervals, 200, 16)
        self.assertAlmostEqual(slow["duration_s"], 3.1)
        self.assertAlmostEqual(fast["duration_s"], 2.1)
        self.assertEqual(slow["output_tokens"], fast["output_tokens"])
        self.assertEqual(fast["tail_id"], "successor")

    def test_peak_fixed_window_can_start_inside_a_measured_interval(self):
        intervals = [(0, 2, (4,)), (2, 4, (16,)), (4, 6, (2,))]
        self.assertAlmostEqual(peak_window(intervals, 3), 12)

    def test_rank_get_overlaps_count_once_inside_each_sampling_interval(self):
        self.assertAlmostEqual(overlap(1, 4, [(0, 2), (1.5, 3.5), (3, 5), (6, 7)]), 3)

    def test_prefill_attribution_keeps_eos_and_placement_on_both_anchors(self):
        low = {"root": {"session_id": "root", "predecessor": None, "P": "P0", "D": "D0", "offered_s": 0,
                        "tool_gap_s": 0, "ready_s": 0, "first_s": 2, "end_s": 2.1, "output_tokens": 2,
                        "phases_s": {"P_HTTP": 2}}}
        high = {"root": {**low["root"], "first_s": 1, "end_s": 1.1, "phases_s": {"P_HTTP": 1}}}
        service = {"first_sequence_period_s": .1, "additional_sequence_period_s": 0}
        report = compare_decode_inputs(low, high, service, 16)
        terms = report["symmetric_seconds_saved_by_factor"]
        self.assertAlmostEqual(terms["P_HTTP"], 1)
        self.assertTrue(all(value == 0 for key, value in terms.items() if key != "P_HTTP"))
        for cap, elapsed in (("100", 1.1), ("200", 2.1)):
            scenario = report["only_P_HTTP_taken_from_other_cap"][cap]
            self.assertEqual(scenario["output_tokens"], 2)
            self.assertAlmostEqual(scenario["duration_s"], elapsed)

    def test_volume_required_for_a_throughput_loss_accounts_for_both_caps(self):
        pairs = [{"estimated_root_output_tokens": 100, "estimated_successor_output_tokens": 1000}]
        costs = {"cold_tokens_s": 1000, "root_slack_s": 0, "wire_fraction": .8, "P_nonwire_per_request_s": 1,
                 "peak_D_tokens_s": 100, "port_skew": 1, "exposed_fraction": .8, "final_D_residual_s": 4.5}
        for target, high_can_supply in ((.1, True), (.4, False)):
            with self.subTest(target=target):
                shape = wave_times(pairs, 200, 2000, costs, target, favorable=True)
                required = magnitude_requirements(shape, 1, target)["min_actual_D_egress_GB_at_same_preparation_and_output"]
                exact = wave_times(pairs, required, 2000, costs, target, favorable=True)
                self.assertAlmostEqual(exact["estimated_100_low_throughput_loss_fraction"], target)
                self.assertEqual(exact["100_P_lags_peak_D_while_200_keeps_up"], high_can_supply)
                self.assertEqual(exact["arms"]["100"]["output_tokens"], exact["arms"]["200"]["output_tokens"])

    def test_source_windows_preserve_adjacent_dependencies_and_never_reuse_a_request(self):
        ledger, sizes, counts, outcomes = {}, {}, {}, {}
        for index, shared in enumerate((0, 67_000, 68_000, 69_000)):
            key = f"task:{index}"
            ledger[key] = {"id": key, "task": "task", "truncated": False, "prompt_tokens": 68_000 + index * 1024,
                           "step_id": index, "source_prompt_lcp_to_previous": shared,
                           "predecessor": f"task:{index - 1}" if index else None}
            sizes[key], counts[key] = 200_000, 120
            outcomes[key] = [("stop", 100)] * 4
        pair_group = max(source_options(ledger, counts, outcomes, sizes, 500)["task"], key=lambda option: len(option.pairs))
        self.assertEqual([(p["root"], p["successor"]) for p in pair_group.pairs],
                         [("task:0", "task:1"), ("task:2", "task:3")])
        self.assertEqual(pair_group.cold_tokens, 68_000 + 70_048 - 66_560)
