"""One registered staging slot per P worker and exact zero-span accounting."""

import time


ALIGNMENT = 4096


def _flatten(addresses, sizes):
    rows = []
    for pointers, lengths in zip(addresses, sizes, strict=True):
        row = []
        for pointer, length in zip(pointers, lengths, strict=True):
            pointer, length = int(pointer), int(length)
            if pointer <= 0 or not 0 < length < 2**31:
                raise ValueError("Invalid transfer pointer or length")
            row.append((pointer, length))
        rows.append(row)
    return rows


class Stage:
    """Hold one physical GET until the normal KV load can publish its data."""

    def __init__(self, worker, capacity=384 * 1024**2):
        import torch
        from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
        from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel

        self.torch = torch
        self.kernel = batch_memcpy_kernel
        backend = worker.m_store
        backend.set_device()
        if backend._use_fabric_mem or global_te.transfer_engine is None:
            raise RuntimeError("Staging requires the initialized non-fabric Mooncake engine")
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("Staging capacity must be positive")
        self.device = torch.device(f"npu:{torch.npu.current_device()}")
        # The allocation and registration stay live as long as the worker owns
        # the slot. The registration includes any padding before the first slice.
        allocation_size = self.capacity + ALIGNMENT - 1
        self.buffer = torch.empty(allocation_size, dtype=torch.uint8, device=self.device)
        pointer = self.buffer.data_ptr()
        self.start = (pointer + ALIGNMENT - 1) & -ALIGNMENT
        # GlobalTE.register_buffer is a one-time startup wrapper; this memory
        # needs its own registration on the very same TransferEngine.
        with global_te.register_buffer_lock:
            result = global_te.transfer_engine.register_memory(pointer, allocation_size)
        if result != 0:
            raise RuntimeError(f"Mooncake staging registration failed with code {result}")

    def _plan(self, addresses, sizes):
        rows = _flatten(addresses, sizes)
        intervals = sorted((pointer, pointer + size) for row in rows for pointer, size in row)
        if not intervals or any(end > nxt for (_, end), (nxt, _) in zip(intervals, intervals[1:])):
            raise ValueError("Staged GET needs nonempty, disjoint live destinations")
        offset = 0
        staged = []
        for row in rows:
            ptrs = []
            for _, size in row:
                offset = (offset + ALIGNMENT - 1) & -ALIGNMENT
                ptrs.append(self.start + offset)
                offset += size
            staged.append(ptrs)
        if offset > self.capacity:
            raise ValueError(f"Staged GET needs {offset} bytes in a {self.capacity}-byte arena")
        return rows, staged

    def layout(self, addresses, sizes):
        """Return an aligned address for each unchanged descriptor length."""
        return self._plan(addresses, sizes)[1]

    def scatter(self, addresses, sizes):
        """Copy completed staging data into live pages on the current NPU stream."""
        begin = time.monotonic_ns()
        rows, staged = self._plan(addresses, sizes)
        dst, lengths = zip(*(elem for row in rows for elem in row), strict=True)
        src = [pointer for row in staged for pointer in row]
        torch = self.torch
        src_gpu = torch.tensor(src, dtype=torch.int64, device=self.device)
        dst_gpu = torch.tensor(dst, dtype=torch.int64, device=self.device)
        lengths_gpu = torch.tensor(lengths, dtype=torch.int32, device=self.device)
        start = torch.npu.Event(enable_timing=True)
        done = torch.npu.Event(enable_timing=True)
        start.record()
        self.kernel[(len(src),)](src_gpu, dst_gpu, lengths_gpu, BLOCK_SIZE=8192)
        done.record()
        done.synchronize()
        return {"begin_ns": begin, "end_ns": time.monotonic_ns(),
                "device_elapsed_ms": start.elapsed_time(done),
                "bytes": sum(lengths), "descriptors": len(src)}


def _union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def zero_overlap(runner, zero_blocks, addresses, sizes):
    """Return (active device zeroer, bytes of live destinations it would overwrite)."""
    zeroer = getattr(runner, "_kv_block_zeroer", None)
    meta = getattr(zeroer, "_meta", None)
    if meta is None:
        return False, 0
    if not zero_blocks:
        return True, 0
    segment_tensor, elements, _, nsegments = meta
    cached = getattr(runner, "_r21_zero_geometry", None)
    if cached is None or cached[0] is not segment_tensor:
        bases = tuple(segment_tensor.cpu().tolist())
        if len(bases) != nsegments or elements <= 0:
            raise RuntimeError("Unexpected Ascend zero-kernel geometry")
        # AscendKVBlockZeroer addresses are absolute, in bytes, with int32 pages.
        cached = (segment_tensor, bases, 4 * elements)
        runner._r21_zero_geometry = cached
    _, bases, page = cached
    to_zero = _union((base + block * page, base + (block + 1) * page)
                     for base in bases for block in zero_blocks)
    live = _union((pointer, pointer + size) for row in _flatten(addresses, sizes) for pointer, size in row)
    i = j = total = 0
    while i < len(live) and j < len(to_zero):
        left, right = live[i], to_zero[j]
        total += max(0, min(left[1], right[1]) - max(left[0], right[0]))
        if left[1] <= right[1]:
            i += 1
        else:
            j += 1
    return True, total
