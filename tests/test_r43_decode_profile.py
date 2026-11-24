import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pd_kv_transfer_opt.common import write_json
from pd_kv_transfer_opt.profiling.analyze_r43_decode import analyze_window
from pd_kv_transfer_opt.profiling.r43_decode_capture import capture_one


def write_rank(folder, rank):
    folder.mkdir()
    rows, calls = [], {}
    trace = [{"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "Ascend Hardware"}}]

    def event(name, start, duration, pid=1, **args):
        trace.append({"ph": "X", "name": name, "pid": pid, "tid": 8, "ts": start, "dur": duration, "args": args})

    for cycle, batch in enumerate(("prefix", "small", "wide")):
        base = 12000 * cycle + 2 * rank
        for op in range(82):
            kind = "allreduce" if op < 81 else "allgather"
            at = base + (1000 + 100 * op if op < 81 else 9400)
            calls[f"{kind}-{cycle}-{op}"] = {"Communication Time Info": {
                "Start Timestamp(us)": at, "Elapse Time(ms)": .01, "Synchronization Time(ms)": 0.}}
            rows.append((at, 10, kind, "COMMUNICATION", "N/A"))
        rows.extend([(base + 200, 150, "MatMul", "AI_CORE", .5),
                     (base + 400, 100, "Attention", "AI_CORE", 1.8),
                     (base + 520, 50, "Add", "AI_CORE", "N/A")])
        event("AscendCL@hcom_allReduce_", base + 750, 10)
        event("AscendCL@aclrtSynchronizeEvent", base + 300, 80)
        event("issue23::decode_graph_" + batch, base + 1020, 20)
        event("AscendCL@aclmdlRIExecuteAsync", base + 1025, 5, connection_id=cycle)
        event("MODEL_EXECUTE", base + 1040, 50, pid=2, connection_id=cycle)
    with (folder / "kernel_details.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Start Time(us)", "Duration(us)", "Type", "Accelerator Core", "memory_bound"))
        writer.writerows(rows)
    write_json(folder / "communication.json", {"Step": {"collective": calls}})
    write_json(folder / "trace_view.json", {"traceEvents": trace})


class R43DecodeProfileTests(unittest.TestCase):
    def test_raw_steps_join_by_actual_width_and_host_overlap_counts_once(self):
        with tempfile.TemporaryDirectory() as scratch:
            folders = [Path(scratch) / str(rank) for rank in range(4)]
            for rank, folder in enumerate(folders):
                write_rank(folder, rank)
            prompts = {f"r{i}": 128 for i in range(10)}
            steps = {key: {"mode": "FULL", "replayed_all_ranks": True, "padded_tokens": padded,
                           "rows": [(rid, 140, 1) for rid in list(prompts)[:width]],
                           "cost": {"rank0_worker_gap_ms": 12.}}
                     for key, width, padded in (("small", 2, 2), ("wide", 10, 16))}
            report = analyze_window(folders, steps, prompts)
        self.assertEqual(report["stats"]["1-4"]["exact_widths"], {2: 1})
        self.assertEqual(report["stats"]["9-16"]["exact_widths"], {10: 1})
        for row in report["steps"]:
            self.assertAlmostEqual(row["arithmetic_candidate_ms"], .15)
            self.assertAlmostEqual(row["memory_candidate_ms"], .1)
            self.assertAlmostEqual(row["mixed_or_unknown_kernel_ms"], .05)
            self.assertAlmostEqual(row["explicit_host_sync_ms"], .03)
            self.assertAlmostEqual(row["save_memory_candidate_0.2_ms"], .02)
            self.assertAlmostEqual(row["period_ms"], sum(row[key] for key in (
                "arithmetic_candidate_ms", "memory_candidate_ms", "mixed_or_unknown_kernel_ms", "copy_ms",
                "device_event_wait_ms", "explicit_host_sync_ms", "unknown_after_visible_host_ms",
                "collective_after_last_ms")))

    def test_capture_receipt_requires_all_four_new_worker_folders(self):
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch)
            root = output / "native-process-profile" / "D0"
            root.mkdir(parents=True)
            (root / "old_rank0_1_ascend_pt").mkdir()
            actions = []

            def rpc(role, endpoint, action):
                actions.append(action)
                if action == "stop":
                    for rank in range(4):
                        (root / f"new_rank{rank}_7_ascend_pt").mkdir()
                return {"status": 200}

            result = {}
            with patch("pd_kv_transfer_opt.profiling.r43_decode_capture.profile_rpc", side_effect=rpc):
                capture_one("D0", "unused", output, result, 0.)
        self.assertEqual(actions, ["start", "stop"])
        self.assertEqual(set(result["ranks"]), {"0", "1", "2", "3"})


if __name__ == "__main__":
    unittest.main()
