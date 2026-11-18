"""Physical KV allocation/copy lifecycle for a future serving simulator adapter.

No timing or model execution is predicted here. Each hybrid state group is an
object; a consumer pins a complete, matching bundle before execution. Allocation
identity is never reconstructed from a content hash when releasing a reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import itertools


@dataclass
class Allocation:
    id: str
    object_id: str
    version: int
    pool: str
    bytes: int
    visible: bool = False
    pins: int = 0
    inflight: bool = False
    freed: bool = False


class Registry:
    def __init__(self, capacities: dict[str, int]):
        if not capacities or any(not isinstance(value, int) or value < 0 for value in capacities.values()):
            raise ValueError("Nonnegative physical pool capacities required")
        self.capacities = dict(capacities)
        self.allocations: dict[str, Allocation] = {}
        self.copies: dict[str, tuple[str, str, str]] = {}
        self.leases: dict[str, tuple[str, ...]] = {}
        self.lease_ids = itertools.count()

    def used(self, pool: str) -> int:
        return sum(item.bytes for item in self.allocations.values() if item.pool == pool and not item.freed)

    def allocate(self, allocation_id: str, object_id: str, version: int, pool: str,
                 size: int) -> Allocation | None:
        if allocation_id in self.allocations:
            raise ValueError("Allocation IDs cannot be reused, including after free")
        if pool not in self.capacities or not isinstance(size, int) or size < 0 or not isinstance(version, int) or version < 0:
            raise ValueError("Invalid allocation")
        if self.used(pool) + size > self.capacities[pool]:
            return None
        item = Allocation(allocation_id, object_id, version, pool, size)
        self.allocations[allocation_id] = item
        self.audit()
        return item

    def live(self, allocation_id: str) -> Allocation:
        item = self.allocations[allocation_id]
        if item.freed:
            raise ValueError("Allocation is already freed")
        return item

    def publish(self, allocation_id: str) -> None:
        item = self.live(allocation_id)
        if item.inflight:
            raise ValueError("Copy completion must publish an in-flight destination")
        item.visible = True

    def start_copy(self, flow_id: str, source_id: str, destination_id: str,
                   pool: str) -> Allocation | None:
        if flow_id in self.copies:
            raise ValueError("Copy IDs cannot be reused")
        source = self.live(source_id)
        if not source.visible:
            raise ValueError("Source is not visible")
        destination = self.allocate(destination_id, source.object_id, source.version, pool, source.bytes)
        if destination is None:
            return None
        source.pins += 1
        destination.inflight = True
        self.copies[flow_id] = (source_id, destination_id, "active")
        self.audit()
        return destination

    def finish_copy(self, flow_id: str, success: bool = True) -> None:
        source_id, destination_id, state = self.copies[flow_id]
        if state != "active":
            raise ValueError("Copy already finished")
        source, destination = self.live(source_id), self.live(destination_id)
        source.pins -= 1
        destination.inflight = False
        if success:
            destination.visible = True
        else:
            destination.freed = True
        self.copies[flow_id] = (source_id, destination_id, "complete" if success else "aborted")
        self.audit()

    def acquire_bundle(self, expected: dict[str, tuple[str, int]]) -> str | None:
        """Map allocation ID -> expected (object ID, checkpoint version)."""
        if not expected:
            raise ValueError("A nonempty complete bundle is required")
        items = [self.live(key) for key in expected]
        if any((item.object_id, item.version) != expected[item.id] for item in items):
            raise ValueError("State object or checkpoint version mismatch")
        if any(not item.visible or item.inflight for item in items):
            return None
        for item in items:
            item.pins += 1
        lease_id = f"lease-{next(self.lease_ids)}"
        self.leases[lease_id] = tuple(expected)
        self.audit()
        return lease_id

    def release_bundle(self, lease_id: str) -> None:
        allocation_ids = self.leases[lease_id]
        items = [self.live(key) for key in allocation_ids]
        if any(item.pins < 1 for item in items):
            raise ValueError("Reference count underflow")
        for item in items:
            item.pins -= 1
        del self.leases[lease_id]
        self.audit()

    def free(self, allocation_id: str) -> None:
        item = self.live(allocation_id)
        if item.pins or item.inflight:
            raise ValueError("Pinned or in-flight allocation cannot be freed")
        item.freed = True
        self.audit()

    def audit(self) -> None:
        if any(self.used(pool) > capacity for pool, capacity in self.capacities.items()):
            raise AssertionError("Physical pool capacity exceeded")
        if any(item.pins < 0 or (item.freed and (item.pins or item.inflight)) for item in self.allocations.values()):
            raise AssertionError("Invalid allocation references")
        expected = Counter(source for source, _, state in self.copies.values() if state == "active")
        expected.update(key for keys in self.leases.values() for key in keys)
        if any(item.pins != expected[item.id] for item in self.allocations.values()):
            raise AssertionError("Reference owners do not match pin counts")
