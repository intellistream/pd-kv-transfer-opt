"""Small CPU-only contracts for trace-conditioned causal queue replay."""

import unittest

from pd_kv_transfer_opt.simulator.trace_replay import simulate


def row(key, task, p=0, d=2, p_time=1.0):
    return {"id": key, "task": task, "offered_arrival_seconds": 0.0,
            "route_seconds": 0.1, "route": {"p": p, "d": d},
            "P_submission_seconds": 0.0, "D_submission_seconds": 0.0,
            "pipeline_timing": {"P_started_seconds": 0.0, "D_started_seconds": 0.0},
            "P_wire": {"first_seconds": p_time, "complete_seconds": p_time},
            "D_wire": {"first_seconds": .3, "complete_seconds": 2.0},
            "offered_to_first_seconds": 0.0}


class Simulation(unittest.TestCase):
    def test_stage_arithmetic_and_dependency_preserve_complete_output_causality(self):
        rows = [row("a", "session"), row("b", "session")]
        result = simulate(rows, ["a", "b"], {})["per_request"]
        self.assertAlmostEqual(result["a"]["first"], 1.4)
        self.assertAlmostEqual(result["a"]["end"], 3.1)
        self.assertEqual(result["b"]["ready"], result["a"]["end"])

    def test_two_P_are_independent_and_D_FIFO_uses_P_completion_order(self):
        rows = [row("slow", "s0", p=0, p_time=2), row("fast", "s1", p=1, p_time=.1)]
        result = simulate(rows, ["slow", "fast"], {})["per_request"]
        self.assertLess(result["fast"]["first"], result["slow"]["first"])
        self.assertAlmostEqual(result["slow"]["D_queue"], .2)

    def test_incomplete_or_duplicate_stream_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate([row("a", "s")], ["a", "b"], {})


if __name__ == "__main__":
    unittest.main()
