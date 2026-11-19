"""One-request cross-batch experiment for the fixed r20 video continuation."""

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from pathlib import Path
import os
import pickle
import time

from prefill_chunk_observability import batch_key, chunk_rows
from r21_stage import Stage, zero_overlap


TARGET_TOKENS = 67315
TARGET_RANK_BYTES = 351641600


def mode():
    path = Path(os.environ['ISSUE23_SAME_HEAD_MODE'])
    return path.read_text().strip() if path.is_file() else 'baseline'


def path():
    return Path(os.environ['ISSUE23_SAME_HEAD_MODE']).parent / 'r21-active-p0'


def is_p0():
    return os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == '0,1,2,3'


def save(filename, value):
    temp = filename.with_name(filename.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as stream:
        pickle.dump(value, stream, protocol=5)
    temp.replace(filename)


def restore(filename):
    try:
        with filename.open('rb') as stream:
            return pickle.load(stream)
    except FileNotFoundError:
        return None


def metadata_requests(meta):
    if hasattr(meta, 'requests'):
        yield meta
    for item in getattr(meta, 'metadata', ()) or ():
        yield from metadata_requests(item)


def loadable(meta):
    return [request for request in meta.requests if request.load_spec is not None and request.load_spec.can_load]


def signature(request):
    spec = request.load_spec
    return (
        request.req_id, request.token_len_chunk, spec.kvpool_cached_tokens, spec.vllm_cached_tokens,
        tuple(request.kv_cache_group_ids or ()), tuple(tuple(group) for group in request.block_ids_by_group),
        tuple(request.block_hashes),
    )


def attach_scheduler(scheduler, emit):
    if not is_p0():
        return
    original_add = scheduler.add_request
    schedule = scheduler.schedule
    previous = None
    watched = set()

    def add(request):
        result = original_add(request)
        if request.num_prompt_tokens == TARGET_TOKENS and mode() != 'baseline':
            watched.add(request.request_id)
            emit('r21_witness_arrival', request_id=request.request_id, prompt_tokens=request.num_prompt_tokens)
        return result

    def observe():
        nonlocal previous
        output = schedule()
        if not output.total_num_scheduled_tokens:
            return output
        batch = batch_key(chunk_rows(output))
        old, previous = previous, (batch, tuple(output.num_scheduled_tokens))
        targets = watched.intersection(output.num_scheduled_tokens)
        if not targets:
            return output
        run_mode = mode()
        if run_mode == 'baseline':
            return output
        candidates = [(meta, request, len(jobs)) for meta in metadata_requests(output.kv_connector_metadata)
                      if (jobs := loadable(meta)) for request in jobs if request.req_id in targets]
        if len(candidates) != 1:
            emit('r21_release_unavailable', batch=batch, mode=run_mode,
                 metadata_name=type(output.kv_connector_metadata).__name__, candidates=len(candidates))
            return output
        meta, request, restores = candidates[0]
        candidate = copy(meta)
        candidate.requests = [request]
        target_blocks = {b for group in request.block_ids_by_group for b in group if b}
        older_blocks = set()
        for rid in old[1] if old else ():
            blocks = scheduler.kv_cache_manager.get_blocks(rid)
            older_blocks.update(b for group in blocks.get_block_ids() for b in group if b)
        hints = {
            'batch': batch, 'predecessor_batch': old[0] if old else None,
            'meta': candidate, 'request_id': request.req_id, 'mode': run_mode,
            'dispatch_ns': time.monotonic_ns(),
            'zero_blocks': list(output.new_block_ids_to_zero or ()),
            'block_alias': len(target_blocks & older_blocks), 'only_restore': restores == 1,
        }
        path().mkdir(exist_ok=True, mode=0o700)
        save(path() / 'candidate', hints)
        watched.discard(request.req_id)
        emit('r21_release', **{key: hints[key] for key in
             ('batch', 'predecessor_batch', 'request_id', 'dispatch_ns', 'mode')},
             zero_block_count=len(hints['zero_blocks']), prior_used_block_overlap=hints['block_alias'],
             only_restore=hints['only_restore'])
        return output

    scheduler.add_request = add
    scheduler.schedule = observe
    emit('r21_scheduler_attached')


class Lookahead:
    def __init__(self, worker, collect, emit):
        self.worker = worker
        self.collect = collect
        self.emit = emit
        self.stage = Stage(worker)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='r21-prefetch')
        self.future = None
        self.active = None
        self.forward_markers = None

    def begin(self, runner, batch, torch):
        hint = restore(path() / 'candidate')
        if (not hint or hint['mode'] not in ('prepare-only', 'early-get') or hint['mode'] != mode()
                or hint['predecessor_batch'] != batch or not hint['only_restore']
                or self.active == hint['batch']):
            return
        if self.future is not None:
            if not self.future.done():
                raise RuntimeError('r21 staging buffer still has an active writer')
            self.future.result()
        self.active = hint['batch']
        first = torch.npu.Event(enable_timing=True)
        first.record()
        markers = {'first': first, 'last': None}
        self.forward_markers = markers
        self.future = self.executor.submit(self.prepare, hint, runner, markers)
        return markers

    def prepare(self, hint, runner, markers):
        worker = self.worker
        worker.m_store.set_device()
        request, = hint['meta'].requests
        snapshot = signature(request)
        begin = time.monotonic_ns()
        shadow = copy(worker)
        shadow.token_database = copy(worker.token_database)
        result, (job,), (plan,) = self.collect(shadow, hint['meta'])
        end = time.monotonic_ns()
        keep, digest, segments, _ = plan
        rid, backend, keys, addrs, sizes = job
        selected = tuple([values[i] for i in keep] for values in (keys, addrs, sizes))
        selected_keys, selected_addrs, selected_sizes = selected
        amount = sum(map(sum, selected_sizes))
        zeroer, overlap = zero_overlap(runner, hint['zero_blocks'], selected_addrs, selected_sizes)
        fields = {'batch': hint['batch'], 'predecessor_batch': hint['predecessor_batch'],
                  'request_id': rid, 'tp_rank': worker.tp_rank, 'mode': hint['mode']}
        self.emit('r21_plan', **fields, prepare_begin_ns=begin, prepare_end_ns=end,
                  total_bytes=amount, staging_bytes=(amount if hint['mode'] == 'early-get' else 0),
                  zero_overlap_bytes=overlap, zeroer_active=zeroer, zero_block_count=len(hint['zero_blocks']),
                  dispatch_ns=hint['dispatch_ns'], prior_used_block_overlap=hint['block_alias'],
                  reason='staging_preserves_successor_page_lifetime')
        ready = {'signature': snapshot, 'job': job, 'plan': plan, 'result': result,
                 'prepare_ns': end - begin, 'fields': fields, 'early': False}
        if hint['mode'] == 'prepare-only' or not segments or amount != TARGET_RANK_BYTES:
            return ready

        def ballot(rank):
            return path() / f'{hint["batch"]}-{rank}'

        # Rendezvous is separate from Gloo, which remains on the normal worker path.
        save(ballot(worker.tp_rank), digest)
        deadline = time.monotonic() + 0.100
        while time.monotonic() < deadline:
            votes = [restore(ballot(rank)) for rank in range(4)]
            if all(votes):
                break
            time.sleep(0.001)
        else:
            return ready
        if votes[0] != votes[1] or votes[2] != votes[3]:
            raise RuntimeError('r21 advance GET requires a matching packed pair plan')
        stage_addrs = self.stage.layout(selected_addrs, selected_sizes)
        first = markers['first']
        last_false = None
        while True:
            before_check = time.monotonic_ns()
            started = first.query()
            if started or time.monotonic() >= deadline:
                first_check = time.monotonic_ns()
                break
            last_false = before_check
            time.sleep(0.0005)
        last = markers['last']
        ended = last.query() if last is not None else None
        get_begin = time.monotonic_ns()
        response = type(backend)._r18_physical_get(backend, selected_keys, stage_addrs, selected_sizes)
        get_end = time.monotonic_ns()
        last = markers['last']
        ended_after = last.query() if last is not None else None
        success = response is not None and len(response) == len(selected_keys) and not any(response)
        self.emit('r21_early_get', **fields, get_begin_ns=get_begin, get_end_ns=get_end, payload_bytes=amount,
                  success=success, forward_started_before_get=started, forward_ended_before_get=ended,
                  forward_started_after_get=first.query(), forward_ended_after_get=ended_after,
                  forward_start_latest_ns=first_check if started else None, forward_start_earliest_ns=last_false)
        if not success:
            raise RuntimeError('r21 staged GET failed before state publication')
        ready['early'] = True
        return ready

    def take(self, batch, request):
        if self.active != batch or self.future is None:
            return None
        begin = time.monotonic_ns()
        result = self.future.result()
        wait = time.monotonic_ns() - begin
        self.future = None
        if result['signature'] != signature(request):
            if result['early']:
                raise RuntimeError('r21 live metadata differs from the issued read')
            return None
        result['wait_ns'] = wait
        markers = self.forward_markers
        ev = markers['last']
        result['old_forward_device_ms'] = markers['first'].elapsed_time(ev) if ev is not None and ev.query() else None
        return result
