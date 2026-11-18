"""Measure pool lookup and backend GET around the native-store timing."""
import functools
import sys
import time

from pd_kv_transfer_opt.profiling.prefill_chunk_observability import InstallFilter as ChunkFilter
from pd_kv_transfer_opt.profiling.scoped_native_observability import emit, span


POOL = 'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store'
SCHEDULER = POOL + '.pool_scheduler'
BACKEND = POOL + '.backend.mooncake_backend'


def install(module):
    if getattr(module, '_issue23_method_observer', False):
        return
    if module.__name__ == SCHEDULER:
        for cls in tuple(vars(module).values()):
            if (
                isinstance(cls, type)
                and cls.__module__ == module.__name__
                and 'get_num_new_matched_tokens' in vars(cls)
            ):
                original = cls.get_num_new_matched_tokens

                @functools.wraps(original)
                def match(self, request, num_computed_tokens, _original=original):
                    started = time.monotonic_ns()
                    with span('pool_match_tokens'):
                        result = _original(self, request, num_computed_tokens)
                    emit('pool_match_tokens', request_id=request.request_id, local_tokens=num_computed_tokens,
                         external_tokens=result[0], load_async=result[1], elapsed_ns=time.monotonic_ns() - started)
                    return result

                cls.get_num_new_matched_tokens = match
    else:
        cls = module.MooncakeBackend
        original = cls.get

        @functools.wraps(original)
        def get(self, keys, addrs, sizes):
            started = time.monotonic_ns()
            with span('mooncake_backend_get'):
                result = original(self, keys, addrs, sizes)
            emit('mooncake_backend_get', keys=len(keys), elapsed_ns=time.monotonic_ns() - started)
            return result

        cls.get = get
    module._issue23_method_observer = True
    emit('rdma_method_observer_installed', module=module.__name__)


class ObserverLoader:
    def __init__(self, downstream):
        self.downstream = downstream

    def create_module(self, spec):
        return self.downstream.create_module(spec)

    def exec_module(self, module):
        self.downstream.exec_module(module)
        install(module)


class ObserverFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (SCHEDULER, BACKEND):
            return None
        start = sys.meta_path.index(self) + 1
        for finder in sys.meta_path[start:]:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = ObserverLoader(spec.loader)
                return spec
        return None


class InstallFilter(ChunkFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not any(isinstance(finder, ObserverFinder) for finder in sys.meta_path):
            sys.meta_path.insert(0, ObserverFinder())
            for name in (SCHEDULER, BACKEND):
                if name in sys.modules:
                    install(sys.modules[name])
