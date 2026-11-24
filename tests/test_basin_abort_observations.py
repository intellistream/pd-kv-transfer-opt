import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pd_kv_transfer_opt.experiments.basin_burst_validation import verified_shaping
from pd_kv_transfer_opt.profiling.kv_lifetime_observability import attach_abort


class Scheduler:
    def __init__(self, *requests):
        self.requests = {request.request_id: request for request in requests}

    def _free_request(self, request, delay_free_blocks=False):
        if not delay_free_blocks:
            self._free_blocks(request)
            return None
        return {"do_remote_prefill": True}

    def _free_blocks(self, request):
        del self.requests[request.request_id]


class NativeAbortObservations(unittest.TestCase):
    def test_records_abort_before_deferred_release_and_later_source_release(self):
        aborted = SimpleNamespace(request_id="d", status=SimpleNamespace(name="FINISHED_ABORTED"),
                                  num_output_tokens=17, kv_transfer_params={"remote_request_id": "p"})
        source = SimpleNamespace(request_id="p", status=SimpleNamespace(name="FINISHED_STOPPED"))
        scheduler = Scheduler(aborted, source)
        attach_abort(scheduler, observe_source=True)
        with patch("pd_kv_transfer_opt.profiling.kv_lifetime_observability.emit") as emit:
            scheduler._free_request(aborted, delay_free_blocks=True)
            scheduler._free_request(source, delay_free_blocks=True)
            self.assertEqual(set(scheduler.requests), {"d", "p"})
            self.assertEqual([call.args[0] for call in emit.call_args_list],
                             ["engine_abort_accepted", "engine_abort_finish_result", "engine_source_kv_held"])
            self.assertEqual(emit.call_args_list[0].kwargs["output_tokens"], 17)
            self.assertTrue(emit.call_args_list[1].kwargs["blocks_still_held"])
            scheduler._free_blocks(aborted)
            scheduler._free_blocks(source)
            self.assertEqual([call.args[0] for call in emit.call_args_list[-2:]],
                             ["engine_abort_kv_release", "engine_source_kv_release"])
        self.assertFalse(scheduler.requests)

    def test_matching_shaping_receipt_covers_all_eight_ports_and_preserves_bytes(self):
        receipt = {"requested_aggregate_Gbps": 200, "readback": {
            str(port): {"configured_Mbps": 25000, "current_Mbps": 25000} for port in range(8)}}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "readback.json"
            raw = json.dumps(receipt, indent=2).encode()
            path.write_bytes(raw)
            self.assertEqual(verified_shaping(path, 200), raw)
            with self.assertRaises(ValueError):
                verified_shaping(path, 100)
            receipt["readback"]["4"]["current_Mbps"] = 12500
            path.write_bytes(json.dumps(receipt).encode())
            with self.assertRaises(ValueError):
                verified_shaping(path, 200)
