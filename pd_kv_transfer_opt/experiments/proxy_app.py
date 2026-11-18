"""Observe the installed proxy and optionally replay its recorded routes."""
import contextvars
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

run_dir=Path(os.environ['PD_KV_RUN_DIR'])
events=(run_dir/f'proxy-events-p{os.getpid()}.jsonl').open('x',buffering=1,encoding='utf-8')
client_id=contextvars.ContextVar('client_request_id',default=None)
def emit(kind,**fields):
    events.write(json.dumps({'type':kind,'monotonic_ns':time.monotonic_ns(),'unix_ns':time.time_ns(),**fields})+'\n')
path=Path(os.environ['PD_KV_PROXY_SOURCE'])
sys.path.insert(0,str(path.parent))
spec=importlib.util.spec_from_file_location('load_balance_proxy_server_example',path)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)

fixture_path = run_dir / 'fixed-routes.json'
if fixture_path.exists():
    fixed_routes = json.loads(fixture_path.read_text())['requests']
    preferred_server = contextvars.ContextVar('preferred_server', default=None)
    scheduler_class = module.SharedProxyScheduler
    original_pick = scheduler_class._pick_server

    def pick_server(self, role, load, *, active_tokens=False, kv_cache=False):
        key = preferred_server.get()
        if key is None:
            return original_pick(self, role, load, active_tokens=active_tokens, kv_cache=kv_cache)
        pool = self._pool(role)
        entry = pool.servers.get(key)
        if entry is None or key in pool.tainted:
            raise RuntimeError(f'Pinned {role.value} server is unavailable: {key}')
        if active_tokens:
            entry.active_tokens += load
        if kv_cache:
            entry.active_kv_cache += load
        self._push_heap(role, key)
        return {'key': key, 'host': entry.host, 'port': entry.port}

    def bind_selection(original):
        def select(self, load, preferred_key=None):
            token = preferred_server.set(preferred_key)
            try:
                return original(self, load)
            finally:
                preferred_server.reset(token)
        return select

    scheduler_class._pick_server = pick_server
    route_roles = {'begin_request': 'prefill', 'reserve_prefill_kv': 'prefill', 'pick_decoder': 'decode'}
    for name in route_roles:
        setattr(scheduler_class, name, bind_selection(getattr(scheduler_class, name)))

    original_schedule = module.WorkerRuntime.schedule

    async def schedule(self, method, /, *args, **kwargs):
        if method in route_roles:
            kwargs['preferred_key'] = fixed_routes[client_id.get()][route_roles[method]]
        return await original_schedule(self, method, *args, **kwargs)

    module.WorkerRuntime.schedule = schedule

config=os.environ.get('ISSUE23_PROXY_ARGS_JSON')
if config:
    import argparse
    module.global_args=argparse.Namespace(**json.loads(config))
else:module.global_args=module.parse_args()
module.setup_logging(module.global_args.log_level)
disable_keepalive=os.environ.get('ISSUE23_HTTP_NO_KEEPALIVE')=='1'
if disable_keepalive:
    original_async_client=module.httpx.AsyncClient
    class ExperimentAsyncClient(original_async_client):
        def __init__(self,*args,**kwargs):
            limits=kwargs.get('limits')
            if limits is not None:
                kwargs['limits']=module.httpx.Limits(max_connections=limits.max_connections,
                    max_keepalive_connections=0,keepalive_expiry=0)
            super().__init__(*args,**kwargs)
    module.httpx.AsyncClient=ExperimentAsyncClient
original_prefill=module.send_request_to_service
original_decode=module.stream_service_response_with_retry

def descriptors(blocks):
    if not isinstance(blocks,list):return None
    if blocks and isinstance(blocks[0],list):return [len(group) for group in blocks]
    return [len(blocks)]

async def prefill(client,endpoint,req_data,request_id,*args,**kwargs):
    cid=client_id.get()
    emit('prefill_begin',id=cid,proxy_request_id=request_id,url=str(client.base_url),
         input_tokens=len(req_data['prompt']) if isinstance(req_data.get('prompt'),list) else None)
    try:
        response=await original_prefill(client,endpoint,req_data,request_id,*args,**kwargs)
        if os.environ.get('ISSUE23_LIGHT_PROXY_OBSERVATIONS')=='1':
            emit('prefill_end',id=cid,proxy_request_id=request_id,status=response.status_code,
                 response_body_bytes=len(response.content),light_observation=True)
            return response
        data=response.json()
        kv=data.get('kv_transfer_params') or {}
        emit('prefill_end',id=cid,proxy_request_id=request_id,status=response.status_code,usage=data.get('usage'),
             handoff_present=bool(kv),remote_engine_id=kv.get('remote_engine_id'),remote_host=kv.get('remote_host'),
             remote_port=kv.get('remote_port'),remote_block_counts=descriptors(kv.get('remote_block_ids')),
             handoff_scalar_fields={k:v for k,v in kv.items() if isinstance(v,(str,int,float,bool)) or v is None})
        return response
    except Exception as exc:
        emit('prefill_error',id=cid,proxy_request_id=request_id,error_type=type(exc).__name__,error=str(exc)[:1000])
        raise

async def decode(client,endpoint,req_data,request_id,*args,**kwargs):
    cid=client_id.get()
    emit('decode_begin',id=cid,proxy_request_id=request_id,url=str(client.base_url),max_tokens=req_data.get('max_tokens'))
    count=0
    try:
        async for chunk in original_decode(client,endpoint,req_data,request_id,*args,**kwargs):
            if count==0:emit('decode_first_chunk',id=cid,proxy_request_id=request_id)
            count+=1
            yield chunk
    except Exception as exc:
        emit('decode_error',id=cid,proxy_request_id=request_id,error_type=type(exc).__name__,error=str(exc)[:1000])
        raise
    finally:
        emit('decode_end',id=cid,proxy_request_id=request_id,chunks=count)

module.send_request_to_service=prefill
module.stream_service_response_with_retry=decode

@module.app.middleware('http')
async def observe(request,call_next):
    cid=request.headers.get('X-Request-Id') or str(uuid.uuid4())
    token=client_id.set(cid)
    emit('proxy_arrival',id=cid,path=request.url.path)
    try:
        # Buffer the incoming body before the example's disconnect listener
        # starts consuming ASGI receive events concurrently with its handler.
        if request.method=='POST':
            await request.body()
        response=await call_next(request)
        response.headers['X-Request-Id']=cid
        return response
    finally:
        client_id.reset(token)

emit('proxy_start',pid=os.getpid(),upstream_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
     args=vars(module.global_args),observation_only=not disable_keepalive,
     http_connection_reuse_disabled=disable_keepalive,fixed_routes=fixture_path.exists())

app=module.app
