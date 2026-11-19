"""Opt-in synchronous AscendStore experiment for TP4 and two KV heads.

Used as a logging filter in the r14 research launcher. The Store, block
allocator, and P→D connector continue to own their normal state.
"""

from contextvars import ContextVar
import ctypes
from datetime import timedelta
from functools import wraps
import hashlib
from pathlib import Path
import os
import re
import sys
import time

if __package__:
    from pd_kv_transfer_opt.profiling.rdma_full_observability import InstallFilter as BaseFilter
    from pd_kv_transfer_opt.profiling.scoped_native_observability import emit
else:
    from rdma_full_observability import InstallFilter as BaseFilter
    from scoped_native_observability import emit


WORKER = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker"
BACKEND = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend"
ACTIVE = ContextVar("same_head_store_load", default=None)
GROUP = re.compile(r"@group:(\d+)(?:@|$)")
STRIPE_BLOCKS = 16
PACKED_CHUNK_BYTES = 128 * 1024**2


class Pair:
    def __init__(self, rank):
        import torch
        import torch.distributed as dist
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

        if dist.get_world_size() != 4 or dist.get_rank() != rank:
            raise RuntimeError("Same-head experiment requires one TP4 world per P instance")
        groups = [
            dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=60))
            for ranks in ([0, 1], [2, 3])
        ]
        self.group = groups[rank // 2]
        self.rank = rank % 2
        self.dist = dist
        self.torch = torch
        self.comm = PyHcclCommunicator(self.group, torch.npu.current_device())
        if not self.comm.available or self.comm.disabled:
            raise RuntimeError("A same-host HCCL communicator is required")
        self.stream = torch.npu.Stream()
        emit("same_head_pair_ready", tp_rank=rank, peer_rank=rank ^ 1)

    def exchange(self, digest, joins, success):
        messages = [None, None]
        self.dist.all_gather_object(messages, (digest, joins, success), group=self.group)
        peer_digest, peer_joins, peer_success = messages[1 - self.rank]
        if peer_digest != digest or len(peer_joins) != len(joins):
            raise RuntimeError("Same-head peers disagree on the Attention object or layout")
        if not success or not peer_success:
            raise RuntimeError("A same-head native Store GET failed before state publication")
        return peer_joins

    def copy(self, segments, peer_joins):
        from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
            aclrtStream_t, buffer_type, hcclDataTypeEnum,
        )

        ranges = []
        for (owner, addr, size, joins_previous, _), peer_join in zip(segments, peer_joins, strict=True):
            if joins_previous and peer_join:
                old_owner, old_addr, old_size = ranges[-1]
                ranges[-1] = old_owner, old_addr, old_size + size
            else:
                ranges.append((owner, addr, size))
        if not ranges:
            return 0, 0.0

        before = self.torch.npu.Event(enable_timing=True)
        after = self.torch.npu.Event(enable_timing=True)
        with self.torch.npu.stream(self.stream):
            before.record(self.stream)
            stream_ptr = aclrtStream_t(self.stream.npu_stream)
            for owner, addr, size in ranges:
                self.comm.hccl.hcclBroadcast(
                    buffer_type(addr), size, hcclDataTypeEnum.hcclUint8,
                    owner, self.comm.comm, stream_ptr,
                )
            after.record(self.stream)
        self.stream.synchronize()
        return len(ranges), before.elapsed_time(after)

    def copy_packed(self, segments):
        from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel

        if not segments:
            return 0, 0.0
        if not hasattr(self, "wire"):
            self.wire = self.torch.empty(PACKED_CHUNK_BYTES, dtype=self.torch.uint8, device=self.comm.device)
        chunks, local_addrs, wire_addrs, lengths = packed_plan(segments, self.wire.data_ptr())
        torch = self.torch
        before = torch.npu.Event(enable_timing=True)
        after = torch.npu.Event(enable_timing=True)
        with torch.npu.stream(self.stream):
            local = torch.tensor(local_addrs, dtype=torch.int64, device=self.comm.device)
            wire = torch.tensor(wire_addrs, dtype=torch.int64, device=self.comm.device)
            sizes = torch.tensor(lengths, dtype=torch.int32, device=self.comm.device)
            before.record(self.stream)
            for owner, begin, end, nbytes in chunks:
                local_chunk = local[begin:end]
                wire_chunk = wire[begin:end]
                size_chunk = sizes[begin:end]
                if owner == self.rank:
                    batch_memcpy_kernel[(end - begin,)](local_chunk, wire_chunk, size_chunk, BLOCK_SIZE=8192)
                self.comm.broadcast(self.wire[:nbytes], src=owner, stream=self.stream)
                if owner != self.rank:
                    batch_memcpy_kernel[(end - begin,)](wire_chunk, local_chunk, size_chunk, BLOCK_SIZE=8192)
            after.record(self.stream)
        self.stream.synchronize()
        return len(chunks), before.elapsed_time(after)


def packed_plan(segments, wire_address):
    chunks, local_addrs, wire_addrs, lengths = [], [], [], []
    for source in (0, 1):
        start = len(local_addrs)
        chunk_bytes = 0
        for owner, addr, size, _, _ in segments:
            if owner != source:
                continue
            if not (addr and 0 < size <= PACKED_CHUNK_BYTES):
                raise RuntimeError("Invalid buffer for packed same-head copy")
            if chunk_bytes and chunk_bytes + size > PACKED_CHUNK_BYTES:
                chunks.append((source, start, len(local_addrs), chunk_bytes))
                start = len(local_addrs)
                chunk_bytes = 0
            local_addrs.append(addr)
            wire_addrs.append(wire_address + chunk_bytes)
            lengths.append(size)
            chunk_bytes += size
        if chunk_bytes:
            chunks.append((source, start, len(local_addrs), chunk_bytes))
    return chunks, local_addrs, wire_addrs, lengths


def attention_groups(worker, module):
    result = set()
    for group_id, group in enumerate(worker.kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        specs = (spec.kv_cache_specs.values()
                 if isinstance(spec, module.UniformTypeKVCacheSpecs) else (spec,))
        if not worker.group_uses_align_state[group_id] and all(
            isinstance(item, module.FullAttentionSpec) for item in specs
        ):
            result.add(group_id)
    if not result:
        raise RuntimeError("No supported same-head Attention groups found")
    return result


def transfer_plan(keys, addrs, sizes, group_ids, rank):
    """Keep native Store order for selected reads and derive a common copy order."""
    by_group = {group: [] for group in sorted(group_ids)}
    rows = list(zip(keys, addrs, sizes, strict=True))
    shift = rank % len(rows)
    canonical = rows[-shift:] + rows[:-shift] if shift else rows
    for key, addr, size in canonical:
        group = GROUP.search(key)
        if group is None:
            raise RuntimeError("Unrecognized AscendStore key")
        group = int(group[1])
        if group in by_group:
            by_group[group].append((key, addr, size))

    kept = set()
    segments = []
    verify_segments = []
    signature = hashlib.sha256()
    for group, group_rows in by_group.items():
        for first in range(0, len(group_rows), STRIPE_BLOCKS):
            stripe = group_rows[first:first + STRIPE_BLOCKS]
            owner = (first // STRIPE_BLOCKS) % 2
            for key, addr, size in stripe:
                signature.update(f"{key}:{tuple(size)}\n".encode())
                if len(addr) != len(size) or len(addr) != len(stripe[0][1]):
                    raise RuntimeError("Inconsistent Attention descriptor layout")
                if owner == rank % 2:
                    kept.add(key)
                if owner != rank % 2 and (first == 0 or first + STRIPE_BLOCKS >= len(group_rows)
                                           or first // STRIPE_BLOCKS == len(group_rows) // (2 * STRIPE_BLOCKS)):
                    verify_segments.append((addr[0], min(size[0], 4096)))
            for col in range(len(stripe[0][1])):
                last_end = None
                for _, addr, size in stripe:
                    if size[col] <= 0:
                        raise RuntimeError("Empty Attention buffer")
                    segments.append((owner, addr[col], size[col], last_end == addr[col], last_end is not None))
                    last_end = addr[col] + size[col]
    attention_keys = {key for group in by_group.values() for key, _, _ in group}
    sampled = verify_segments[::max(1, len(verify_segments) // 8)][:8]
    keep_indexes = [i for i, key in enumerate(keys) if key not in attention_keys or key in kept]
    return keep_indexes, signature.digest(), segments, sampled


def read_samples(segments):
    memcpy = ctypes.CDLL("libascendcl.so").aclrtMemcpy
    memcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    memcpy.restype = ctypes.c_int
    values = []
    for addr, size in segments:
        out = ctypes.create_string_buffer(size)
        result = memcpy(out, size, addr, size, 2)
        if result:
            raise RuntimeError(f"ACL device-to-host sample copy failed with {result}")
        values.append(out.raw)
    return values


def deduplicated_get(original, backend, keys, addrs, sizes, context):
    worker, request_ids, mode = context
    pair = worker._same_head_pair
    keep, digest, segments, samples = transfer_plan(
        keys, addrs, sizes, worker._same_head_groups, worker.tp_rank,
    )
    if len(keys) > len(set(keys)):
        raise RuntimeError("Duplicate descriptors in one Store GET")
    status = original(backend, [keys[i] for i in keep], [addrs[i] for i in keep], [sizes[i] for i in keep])
    success = status is not None and len(status) == len(keep) and not any(status)
    started = time.monotonic_ns()
    peer_joins = pair.exchange(digest, bytes(row[3] for row in segments), success)
    admitted = time.monotonic_ns()
    packed = mode in {"packed", "verify-packed"}
    if packed:
        operations, device_ms = pair.copy_packed(segments)
    else:
        operations, device_ms = pair.copy(segments, peer_joins)
    elapsed = time.monotonic_ns() - admitted
    received = sum(size for owner, _, size, _, _ in segments if owner != pair.rank)
    verifying = mode in {"verify", "verify-packed"}
    emit("same_head_local_copy", request_ids=request_ids, tp_rank=worker.tp_rank,
         payload_bytes=received, elapsed_ns=elapsed, device_elapsed_ms=device_ms,
         pair_wait_ns=admitted - started, operations=operations, verify=verifying,
         method="packed" if packed else "direct")
    if mode == "profile":
        pairs = [(row[3], peer) for row, peer in zip(segments, peer_joins, strict=True) if row[4]]
        emit("same_head_copy_layout", request_ids=request_ids, tp_rank=worker.tp_rank,
             raw_segments=len(segments), layout_boundaries=len(segments) - len(pairs),
             local_only_breaks=sum(not local and peer for local, peer in pairs),
             peer_only_breaks=sum(local and not peer for local, peer in pairs),
             both_breaks=sum(not local and not peer for local, peer in pairs),
             contiguous_both=sum(local and peer for local, peer in pairs))
    if verifying:
        observed = read_samples(samples)
        direct = original(backend, keys, addrs, sizes)
        if direct is None or len(direct) != len(keys) or any(direct):
            raise RuntimeError("Independent native GET for comparison failed")
        if observed != read_samples(samples):
            raise RuntimeError("Peer-distributed Attention differs from independent GET")
        emit("same_head_verify", request_ids=request_ids, tp_rank=worker.tp_rank,
             sample_bytes=sum(size for _, size in samples), samples=len(samples),
             method="packed" if packed else "direct")
    return [0] * len(keys)


def install(module, mode_file):
    if getattr(module, "_same_head_installed", False):
        return
    if module.__name__ == WORKER:
        worker_class = module.KVPoolWorker
        original = worker_class.start_load_kv

        @wraps(original)
        def load(self, metadata):
            ids = [req.req_id for req in metadata.requests if req.load_spec is not None and req.load_spec.can_load]
            if not ids:
                return original(self, metadata)
            mode = mode_file.read_text().strip() if mode_file.exists() else "baseline"
            if mode == "baseline":
                return original(self, metadata)
            if mode not in {"enabled", "verify", "profile", "packed", "verify-packed"}:
                raise RuntimeError(f"Unknown same-head mode {mode!r}")
            if not (self.kv_role == "kv_producer" and self.tp_size == 4 and self.num_kv_head == 2
                    and self.put_step == 2 and self.pcp_size == self.dcp_size == self.pp_size == 1
                    and not self.load_async and not self.use_layerwise and self.backend_name == "mooncake"):
                raise RuntimeError("Unexpected configuration for same-head KV experiment")
            if not hasattr(self, "_same_head_pair"):
                self._same_head_groups = attention_groups(self, module)
                self._same_head_pair = Pair(self.tp_rank)
            token = ACTIVE.set((self, ids, mode))
            try:
                return original(self, metadata)
            finally:
                ACTIVE.reset(token)

        worker_class.start_load_kv = load
    elif module.__name__ == BACKEND:
        backend_class = module.MooncakeBackend
        original = backend_class.get

        @wraps(original)
        def get(self, keys, addrs, sizes):
            context = ACTIVE.get()
            if context is None or not keys or context[0].m_store is not self:
                return original(self, keys, addrs, sizes)
            return deduplicated_get(original, self, keys, addrs, sizes, context)

        backend_class.get = get
    module._same_head_installed = True


class Loader:
    def __init__(self, downstream, mode_file):
        self.downstream = downstream
        self.mode_file = mode_file

    def create_module(self, spec):
        return self.downstream.create_module(spec)

    def exec_module(self, module):
        self.downstream.exec_module(module)
        install(module, self.mode_file)


class Finder:
    def __init__(self, mode_file):
        self.mode_file = mode_file

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (WORKER, BACKEND):
            return None
        start = sys.meta_path.index(self) + 1
        for finder in sys.meta_path[start:]:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader, self.mode_file)
                return spec
        return None


class InstallFilter(BaseFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode_file = Path(os.environ["ISSUE23_SAME_HEAD_MODE"])
        if not any(isinstance(finder, Finder) for finder in sys.meta_path):
            sys.meta_path.insert(0, Finder(mode_file))
            for name in (WORKER, BACKEND):
                if name in sys.modules:
                    install(sys.modules[name], mode_file)
