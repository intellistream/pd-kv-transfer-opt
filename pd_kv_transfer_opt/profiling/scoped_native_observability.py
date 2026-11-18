"""Observe native calls without enabling DEBUG in the model or acquiring logging locks."""
import contextlib,functools,importlib.abc,importlib.machinery,json,logging,os,sys,threading,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
_installed=False
_batch_context=threading.local()

def destinations_overlap(jobs):
    """Only distinct GET calls may run concurrently; shared destinations fall back."""
    ranges=sorted((int(addr),int(addr)+int(size),index)
        for index,(_,_,addrs,sizes) in enumerate(jobs)
        for addresses,lengths in zip(addrs,sizes,strict=True)
        for addr,size in zip(addresses,lengths,strict=True) if int(size)>0)
    farthest=-1;owner=-1
    for start,end,index in ranges:
        if start<farthest and index!=owner:return True
        if end>farthest:farthest=end;owner=index
    return False

def finish_batch(worker,jobs,original_get,concurrency):
    """All native reads finish successfully before start_load_kv returns."""
    if not jobs:return
    overlap=destinations_overlap(jobs) if len(jobs)>1 else False
    workers=min(concurrency,len(jobs)) if not overlap else 1
    def read(job):
        backend,keys,addrs,sizes=job
        backend.set_device()
        result=original_get(backend,keys,addrs,sizes)
        if result is None or len(result)!=len(keys) or any(int(v)!=0 for v in result):
            raise RuntimeError('Experimental batched KV read failed before model forward')
    start=time.monotonic_ns()
    with span('batch_get_gate'):
        if workers==1:
            for job in jobs:read(job)
        else:
            if not hasattr(worker,'_issue23_read_executor'):
                worker._issue23_read_executor=ThreadPoolExecutor(max_workers=concurrency,thread_name_prefix='issue23-pool-get')
            futures=[worker._issue23_read_executor.submit(read,job) for job in jobs]
            # Drain every read, including after an exception, before reporting failure.
            failure=None
            for future in futures:
                try:future.result()
                except BaseException as exc:failure=exc
            if failure is not None:raise failure
    emit('batch_get_gate',jobs=len(jobs),workers=workers,overlap_fallback=overlap,
         elapsed_ns=time.monotonic_ns()-start,
         payload_bytes=sum(sum(sum(int(n) for n in group) for group in sizes) for _,_,_,sizes in jobs))
def emit(kind,**fields):
    fields.update(type=kind,pid=os.getpid(),thread_id=threading.get_native_id(),unix_ns=time.time_ns(),monotonic_ns=time.monotonic_ns())
    os.write(2,('ISSUE23_OBSERVATION '+json.dumps(fields,separators=(',',':'))+'\n').encode())
def span(name):
    torch=sys.modules.get('torch')
    return torch.profiler.record_function('issue23::'+name) if torch is not None else contextlib.nullcontext()
class NativeLogger:
    def __init__(self,original):self.original=original
    def __getattr__(self,name):return getattr(self.original,name)
    def debug(self,msg,*args,**kwargs):
        if msg=='Mooncake transfer request=%s session id=%s src=%s dst=%s length=%s':
            rid,session,_,_,lengths=args;sizes=[int(n) for n in lengths]
            emit('native_transfer_submit',remote_request_id=rid,remote_session_id=session,
                 payload_bytes=sum(sizes),descriptor_count=len(sizes),descriptor_size_histogram=dict(Counter(sizes)))
        elif msg=='Adding request %s to the queue.Trans info:%s':
            rid,info=args
            emit('native_transfer_enqueue',request_id=rid,remote_request_id=info.get('remote_request_id'),
                 remote_host=info.get('remote_host'),remote_handshake_port=info.get('remote_handshake_port'),shard_idx=info.get('shard_idx'))
        elif msg in ['Starting to transfer KV cache for request %s.','Finished transferring KV cache for request %s.']:
            emit('native_handle_begin' if msg.startswith('Starting') else 'native_handle_end',remote_request_id=args[0])
        # All other DEBUG calls retain the original disabled behavior.
class ObservedStore:
    def __init__(self,native):self.native=native
    def __getattr__(self,name):return getattr(self.native,name)
    def call(self,method,keys,addrs,sizes,*extra):
        identity=f'{os.getpid()}:{threading.get_native_id()}:{time.monotonic_ns()}'
        payload=sum(sum(int(n) for n in group) for group in sizes)
        emit('store_begin',call_id=identity,operation=method,keys=len(keys),payload_bytes=payload)
        start=time.monotonic_ns()
        with span(method):result=getattr(self.native,method)(keys,addrs,sizes,*extra)
        emit('store_end',call_id=identity,operation=method,elapsed_ns=time.monotonic_ns()-start,
             negative_returns=sum(int(r)<0 for r in result),payload_bytes=payload,keys=len(keys))
        return result
    def batch_get_into_multi_buffers(self,keys,addrs,sizes):return self.call('batch_get_into_multi_buffers',keys,addrs,sizes)
    def batch_put_from_multi_buffers(self,keys,addrs,sizes,*extra):return self.call('batch_put_from_multi_buffers',keys,addrs,sizes,*extra)
class Loader:
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec) if hasattr(self.original,'create_module') else None
    def exec_module(self,module):
        self.original.exec_module(module)
        if module.__name__.endswith('mooncake_backend'):
            original=module.MooncakeBackend._setup_store
            @functools.wraps(original)
            def setup(self,*args,**kwargs):return ObservedStore(original(self,*args,**kwargs))
            module.MooncakeBackend._setup_store=setup
            original_get=module.MooncakeBackend.get
            module.MooncakeBackend._issue23_original_get=original_get
            @functools.wraps(original_get)
            def get(self,keys,addrs,sizes):
                jobs=getattr(_batch_context,'jobs',None)
                if jobs is None:return original_get(self,keys,addrs,sizes)
                # prepare_value only computes pointers and lengths. The stock caller
                # performs no model/cache computation between GET and method return.
                jobs.append((self,list(keys),[list(a) for a in addrs],[list(s) for s in sizes]))
                return [0]*len(keys)
            module.MooncakeBackend.get=get
        elif module.__name__.endswith('mooncake_connector'):
            module.logger=NativeLogger(module.logger)
        else:
            original=module.KVPoolWorker.start_load_kv
            @functools.wraps(original)
            def load(self,metadata):
                ids=[r.req_id for r in metadata.requests if r.load_spec is not None and r.load_spec.can_load]
                if not ids:return original(self,metadata)
                emit('worker_load_begin',request_ids=ids,load_async=self.load_async);start=time.monotonic_ns()
                concurrency=int(os.environ.get('ISSUE23_BATCH_GET_CONCURRENCY','1'))
                with span('worker_start_load_kv'):
                    if concurrency<=1 or self.load_async:
                        result=original(self,metadata)
                    else:
                        if hasattr(_batch_context,'jobs'):raise RuntimeError('Nested KV load batch')
                        jobs=[];_batch_context.jobs=jobs
                        try:result=original(self,metadata)
                        finally:del _batch_context.jobs
                        finish_batch(self,jobs,type(self.m_store)._issue23_original_get,concurrency)
                emit('worker_load_end',request_ids=ids,load_async=self.load_async,elapsed_ns=time.monotonic_ns()-start)
                return result
            module.KVPoolWorker.start_load_kv=load
        emit('observation_installed',module=module.__name__,model_root_log_level=logging.getLogger('vllm').getEffectiveLevel())
class Finder(importlib.abc.MetaPathFinder):
    targets={'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend',
             'vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector',
             'vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker'}
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in self.targets:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec and spec.loader:spec.loader=Loader(spec.loader)
        return spec
class InstallFilter(logging.Filter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        global _installed
        if not _installed:sys.meta_path.insert(0,Finder());_installed=True
    def filter(self,record):return True
