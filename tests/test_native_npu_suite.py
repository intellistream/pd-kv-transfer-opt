import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pd_kv_transfer_opt.experiments.loadgen import request_one
from pd_kv_transfer_opt.experiments import native_npu_suite as suite


class NativeSuiteTest(unittest.TestCase):
    def test_compact_request_preserves_token_ids_and_eos(self):
        row = {"id": "check", "offered_s": 0, "ready_s": 0, "prompt": [100, 200, 300], "max_tokens": 4}
        with patch("urllib.request.urlopen", side_effect=OSError("offline")) as opening:
            request_one(row, "http://127.0.0.1:18400", "mock", time.monotonic(), 2, True, False,
                        lambda _event: None, compact_json=True)
        self.assertEqual(opening.call_args.args[0].data,
                         b'{"model":"mock","prompt":[100,200,300],"max_tokens":4,"temperature":0,'
                         b'"stream":true,"stream_options":{"include_usage":true},"return_token_ids":true}')

    def test_both_hosts_restore_and_bad_readback_prevents_load(self):
        for bad_readback in (False, True):
            with self.subTest(bad_readback=bad_readback), TemporaryDirectory() as temporary:
                self.exercise_arm(Path(temporary), bad_readback)

    def exercise_arm(self, root, bad_readback):
        history = []
        summary = {"counts": {"ok": 254}, "quality": {"client_accounting_complete": True},
                   "finite_cohort": {"output_tokens_per_s": 42}}
        config = suite.read_deployment(suite.BASELINE)
        arm = suite.Arm("r43-test", 100, "requests.jsonl", "unused", requests=254)
        hosts = {role: Mock(ssh=role, hccn_tool="hccn_tool", sudo=True) for role in ("P", "D")}
        for role, host in hosts.items():
            host.python.side_effect = lambda module, *args, **_kwargs: [module, *args]
            host.run.return_value.stdout = json.dumps(summary).encode()
            host.put_json.side_effect = lambda name, *_args: history.append(Path(name).name)

        class Job:
            def __init__(self, host, argv, path, **_kwargs):
                self.host, self.output = host, path
                self.process = SimpleNamespace(poll=lambda: 0, returncode=0)
                history.append(path.name)
                if path.name == "client.log":
                    assert "P-shaping-readback.json" in history and "D-shaping-readback.json" in history
                    assert host.ssh == "D" and argv[argv.index("--endpoint") + 1].startswith("http://127.0.0.1:")
                    assert "--compact-json" in argv and "--ignore-eos" not in argv

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                self.close()

            def ensure_running(self):
                pass

            def close(self):
                history.append(f"close:{self.output.name}")

            def event(self, kind):
                if kind == "restored":
                    return {"restored": True}
                per_port = 25_000 if bad_readback and self.host.ssh == "D" else 12_500
                return {"aggregate_Gbps": 100, "ports": {str(p): {"configured_Mbps": per_port,
                       "current_Mbps": per_port} for p in range(8)}}

        with (patch.object(suite, "Job", Job), patch.object(suite, "idle"), patch.object(suite, "host_archive") as archives,
              patch.object(suite, "state_ready", return_value={"stage": "ready"}),
              patch.object(suite, "store_snapshot", return_value={"key_count": 0}),
              patch.object(suite, "telemetry_ready", return_value=True),
              patch.object(suite, "proxy_state", return_value={"request_num": 0})):
            args = (arm, config, str(suite.BASELINE), {"upstream_proxy": "upstream.py"}, hosts, "test", root / "trial")
            if bad_readback:
                with self.assertRaisesRegex(ValueError, "D egress ports"):
                    suite.run_arm(*args)
                self.assertNotIn("client.log", history)
            else:
                self.assertEqual(suite.run_arm(*args), 42)
            self.assertEqual([event for event in history if event.endswith("shaping.log") and event.startswith("close:")],
                             ["close:D-shaping.log", "close:P-shaping.log"])
            self.assertEqual(archives.call_count, 2)


if __name__ == "__main__":
    unittest.main()
