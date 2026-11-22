"""Cache residency for the agent-trace replay."""

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from math import ceil

from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import ALIGN_STATE_OBJECT_BYTES
from pd_kv_transfer_opt.profiling.screen_restore_residency import ATTENTION_BLOCK_TOKENS, ATTENTION_BYTES


STATE_BYTES_PER_CHECKPOINT = 4 * ALIGN_STATE_OBJECT_BYTES
STORE_ATTENTION_BYTES = ATTENTION_BYTES // 2


def prefix_size(pages, has_attention, has_state):
    """Require contiguous Attention and an aligned recurrent checkpoint at the boundary."""
    best = 0
    for index, key in enumerate(pages, 1):
        if not has_attention(key):
            break
        if has_state(key):
            best = index
    return best


@dataclass(slots=True)
class Lease:
    keys: list[int] = field(default_factory=list)
    private: int = 0


class LocalCache:
    """Pin live HBM pages; evict only physical pages whose requests released them."""

    def __init__(self, capacity, *, release_suffix_first=False):
        self.capacity = capacity
        self.release_suffix_first = release_suffix_first
        self.idle = OrderedDict()
        self.live = Counter()
        self.states = set()
        self.reserved = self.evictions = self.peak = 0

    def prefix(self, pages):
        return prefix_size(pages, self.contains, self.states.__contains__)

    def contains(self, key):
        return key in self.live or key in self.idle

    def reserve(self, local, tokens):
        private = ceil(tokens / ATTENTION_BLOCK_TOKENS) - len(local)
        new_pins = sum(key not in self.live for key in local)
        used = len(self.live) + self.reserved + private + new_pins
        if used > self.capacity:
            return None
        keys = list(local)
        for key in keys:
            self.idle.pop(key, None)
            self.live[key] += 1
        self.reserved += private
        while used + len(self.idle) > self.capacity:
            key, _ = self.idle.popitem(last=False)
            self.states.discard(key)
            self.evictions += 1
        self.peak = max(self.peak, used + len(self.idle))
        return Lease(keys, private)

    def publish(self, lease, pages, states):
        if len(pages) > lease.private:
            raise RuntimeError("Publishing HBM pages without a reserved destination")
        for key in pages:
            self.idle.pop(key, None)
            self.live[key] += 1
            lease.keys.append(key)
        self.states.update(states)
        lease.private -= len(pages)
        self.reserved -= len(pages)

    def release(self, lease):
        self.reserved -= lease.private
        keys = reversed(lease.keys) if self.release_suffix_first else lease.keys
        for key in keys:
            self.live[key] -= 1
            if not self.live[key]:
                del self.live[key]
                self.idle[key] = None


class Store:
    """Size-aware LRU of unique head objects and independent aligned state objects."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.objects = OrderedDict()
        self.pins = Counter()
        self.ever_published = set()
        self.used = self.peak = self.evicted = self.failed = 0
        self.written = self.eviction_bytes = 0

    @staticmethod
    def size(key):
        return STORE_ATTENTION_BYTES if key > 0 else STATE_BYTES_PER_CHECKPOINT

    def lookup(self, pages, local):
        suffix = pages[local:]
        resident = local + prefix_size(suffix, self.objects.__contains__, lambda k: -k in self.objects)
        shadow = local + prefix_size(suffix, self.ever_published.__contains__,
                                     lambda k: -k in self.ever_published)
        return resident, shadow

    def acquire(self, attention, states):
        keys = [*attention, *(-key for key in states)]
        for key in keys:
            self.objects.move_to_end(key)
            self.pins[key] += 1
        return keys

    def release(self, keys):
        for key in keys:
            self.pins[key] -= 1
            if not self.pins[key]:
                del self.pins[key]

    def publish(self, keys):
        for key in keys:
            if key in self.objects:
                self.objects.move_to_end(key)
                continue
            size = self.size(key)
            self.ever_published.add(key)
            while self.used + size > self.capacity:
                victim = next((item for item in self.objects if item not in self.pins), None)
                if victim is None:
                    break
                del self.objects[victim]
                freed = self.size(victim)
                self.used -= freed
                self.eviction_bytes += freed
                self.evicted += 1
            if self.used + size > self.capacity:
                self.failed += 1
                continue
            self.objects[key] = None
            self.used += size
            self.written += size
            self.peak = max(self.peak, self.used)
