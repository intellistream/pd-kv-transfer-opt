import unittest

from pd_kv_transfer_opt.simulator.agent_cache import LocalCache, Store
from pd_kv_transfer_opt.simulator.agent_replay import Replay, Service
from pd_kv_transfer_opt.simulator.freeinference_ab import report
from pd_kv_transfer_opt.simulator.terminal_rotation_pressure import PairedClients, Source, workload
from pd_kv_transfer_opt.workloads.freeinference import Session, Turn


class AgentReplay(unittest.TestCase):
    def test_live_pages_survive_pressure_and_store_uses_complete_checkpoints(self):
        cache = LocalCache(2)
        live = cache.reserve((), 2048)
        cache.publish(live, (1, 2), (2,))
        self.assertEqual(cache.prefix((1, 2)), 2)
        self.assertIsNone(cache.reserve((), 1024))
        cache.release(live)
        other = cache.reserve((), 1024)
        self.assertEqual(cache.prefix((1, 2)), 0)
        cache.release(other)

        store = Store(Store.size(1) + Store.size(-1))
        store.publish((1,))
        self.assertEqual(store.lookup((1,), 0), (0, 0))
        store.publish((-1,))
        pins = store.acquire((1,), (1,))
        store.publish((2,))
        self.assertEqual(store.failed, 1)
        self.assertEqual(store.lookup((1,), 0), (1, 1))
        store.release(pins)
        store.publish((2, -2))
        self.assertEqual(store.lookup((1,), 0), (0, 1))

    def test_cap_only_changes_same_bytes_and_successor_waits_for_the_natural_output(self):
        pages = (1, 2, 3)
        root = Turn(0, 3073, 3, pages, 0, "tool_calls")
        child = Turn(625, 3073, 2, pages, 3, "stop")
        first = Session("a", 0, 0, 0, (root, child))
        next_root = Session("b", 1, 0, 0, (root,))
        service = Service((.125,) * 8, .05, .001, 4096, .002, 0, 4, 4, first_admission_s=0)
        runs = []
        for cap in (100, 200):
            run = Replay([first, next_root], {"a": 0, "b": .80}, service, cap).run()
            a, c, b = (run.requests[name].row for name in ("a:0", "a:1", "b:0"))
            self.assertEqual((a["P"], b["P"]), (0, 1))
            self.assertAlmostEqual(c["offered_s"], a["EOS_s"] + .625)
            self.assertEqual(b["P_external_pages"], 3)
            self.assertEqual(c["P_local_pages"], 3)
            summary = report(run)
            self.assertEqual(summary["output_tokens"], 8)
            self.assertAlmostEqual(sum(summary["D_to_P_GB_by_port"]), summary["GET_GB"])
            runs.append((b["GET_bytes"], b["GET_done_s"] - b["GET_submit_s"]))
        self.assertEqual(runs[0][0], runs[1][0])
        self.assertAlmostEqual(runs[0][1], 2 * runs[1][1])

    def test_client_pair_gate_waits_for_first_tokens_and_drains_completed_eos(self):
        source = Source("fixture", "fixture", 3073, 50, 12000, "test", 4)
        service = Service((.125,) * 8, .02, .001, 4096, .002, 0, 4, 4, first_admission_s=0)
        plan = workload([source], rounds=2, pressure_start=3, read_offer_GB_s=16, preparation_copies=1)
        model = PairedClients(plan, service, 100, {"restore-000-00-1:0": .003}).run()
        first = [model.requests[f"restore-000-00-{r}:0"].row for r in range(2)]
        second = [model.requests[f"restore-001-00-{r}:0"].row for r in range(2)]
        self.assertEqual({r["P"] for r in second}, {0, 1})
        self.assertAlmostEqual(second[0]["offered_s"], max(r["D_first_s"] for r in first))
        self.assertLess(second[0]["offered_s"], min(r["EOS_s"] for r in first))
        self.assertEqual(report(model)["output_tokens"], 5 * source.output)


if __name__ == "__main__":
    unittest.main()
