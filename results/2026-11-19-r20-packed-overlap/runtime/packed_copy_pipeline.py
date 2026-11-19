"""A single in-flight packed copy on the existing same-head NPU stream."""

from dataclasses import dataclass
import time

from same_head_kv_r16 import PACKED_CHUNK_BYTES, packed_plan


@dataclass
class PendingCopy:
    pair: object
    before: object
    after: object
    # Keep descriptor tensors alive until scatter has finished on the NPU.
    descriptors: tuple
    operations: int
    record_start_ns: int
    enqueue_end_ns: int

    def probe(self):
        start = time.monotonic_ns()
        before_ready = self.before.query()
        after_ready = self.after.query()
        return [start, time.monotonic_ns(), before_ready, after_ready]

    def finish(self):
        wait_start = time.monotonic_ns()
        self.pair.stream.synchronize()
        wait_end = time.monotonic_ns()
        return self.before.elapsed_time(self.after), wait_start, wait_end


def submit(pair, segments):
    """Queue the r16 pack/broadcast/scatter sequence and retain its descriptors."""
    from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel

    torch = pair.torch
    if not hasattr(pair, "wire"):
        pair.wire = torch.empty(PACKED_CHUNK_BYTES, dtype=torch.uint8, device=pair.comm.device)
    chunks, local_addrs, wire_addrs, lengths = packed_plan(segments, pair.wire.data_ptr())
    before = torch.npu.Event(enable_timing=True)
    after = torch.npu.Event(enable_timing=True)
    try:
        with torch.npu.stream(pair.stream):
            local = torch.tensor(local_addrs, dtype=torch.int64, device=pair.comm.device)
            wire = torch.tensor(wire_addrs, dtype=torch.int64, device=pair.comm.device)
            sizes = torch.tensor(lengths, dtype=torch.int32, device=pair.comm.device)
            record_start = time.monotonic_ns()
            before.record(pair.stream)
            for owner, begin, end, nbytes in chunks:
                local_chunk = local[begin:end]
                wire_chunk = wire[begin:end]
                size_chunk = sizes[begin:end]
                if owner == pair.rank:
                    batch_memcpy_kernel[(end - begin,)](local_chunk, wire_chunk, size_chunk, BLOCK_SIZE=8192)
                pair.comm.broadcast(pair.wire[:nbytes], src=owner, stream=pair.stream)
                if owner != pair.rank:
                    batch_memcpy_kernel[(end - begin,)](wire_chunk, local_chunk, size_chunk, BLOCK_SIZE=8192)
            after.record(pair.stream)
    except BaseException:
        pair.stream.synchronize()
        raise
    return PendingCopy(pair, before, after, (local, wire, sizes), len(chunks), record_start, time.monotonic_ns())
