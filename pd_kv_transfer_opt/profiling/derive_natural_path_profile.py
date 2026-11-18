"""Attribute TTFT and the final session drain using the runtime replay timeline."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path

from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import analyze, read_bundle
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import get_batches, normalized_proxy_id, pd_transfers, stats


def load_before_forward(p, batches):
    """Check synchronous pool GETs against the same rank's model-forward boundary."""
    get_by_thread = defaultdict(list)
    pending = {}
    for row in sorted(p['native'], key=lambda r: r['monotonic_ns']):
        kind = row['type']
        if kind == 'store_begin' and row['operation'] == 'batch_get_into_multi_buffers':
            pending[row['call_id']] = row['monotonic_ns']
        if kind == 'store_end' and row['operation'] == 'batch_get_into_multi_buffers':
            start = pending.pop(row['call_id'], None)
            if start is not None:
                get_by_thread[(row['pid'], row['thread_id'])].append((start, row['monotonic_ns'], row['elapsed_ns']))
    starts = {key: [start for start, _, _ in calls] for key, calls in get_by_thread.items()}
    timed_workers = {b['device']['pid'] for b in batches if 'device' in b}
    spans = []
    counts = 0
    for batch in batches:
        if 'device' not in batch:
            continue
        device = batch['device']
        key = device['pid'], device['thread_id']
        if key not in get_by_thread:
            continue
        begin, forward = device['worker_enter_ns'], device['forward_enter_ns']
        lo, hi = bisect_left(starts[key], begin), bisect_left(starts[key], forward)
        calls = [elapsed for _, end, elapsed in get_by_thread[key][lo:hi] if end <= forward]
        if calls:
            counts += len(calls)
            spans.append(sum(calls) / 1e6)
    return {'rank0_timed_native_GET_calls': sum(len(calls) for (pid, _), calls in get_by_thread.items() if pid in timed_workers),
            'rank0_GET_calls_inside_worker_before_forward': counts,
            'worker_steps_with_observed_sync_GET': len(spans), 'GET_wall_ms_in_those_steps': stats(spans)}


def timeline(p, d, general):
    manifest, = d['client_manifest']
    outcomes = {row['id']: row for row in d['client'] if row['type'] == 'request_end' and row['status'] == 'ok'}
    proxy = defaultdict(dict)
    for row in d['proxy']:
        if row['id'] in outcomes:
            proxy[row['id']][row['type']] = row
    arrivals, batches, errors = get_batches(p['native'])
    p_requests = {}
    for rid, event in arrivals.items():
        p_requests[normalized_proxy_id(rid)] = (event['arrival_ns'], event['instance'])
    steps = defaultdict(list)
    for batch in batches:
        for rid, _, _ in batch['rows']:
            steps[normalized_proxy_id(rid)].append(batch)
    transfers = pd_transfers(d)
    rows = []
    for spec in manifest['request_sources']:
        request = spec['id']
        if request not in outcomes:
            continue
        outcome = outcomes[request]
        events = proxy[request]
        pid = events['prefill_begin']['proxy_request_id']
        p_arrival, p_instance = p_requests[pid]
        queue = (steps[pid][0]['dispatch_ns'] - p_arrival) / 1e9

        def span(begin, end):
            return (events[end]['monotonic_ns'] - events[begin]['monotonic_ns']) / 1e9

        p_http = span('prefill_begin', 'prefill_end')
        decode_first = span('decode_begin', 'decode_first_chunk')
        client_ttft = outcome['first_s'] - outcome['ready_s']
        native = transfers['per_request'].get(request, {})
        get = general['P_GET']['per_request'].get(request, {})
        rows.append({'id': request, 'session_id': spec['session_id'], 'first_turn': spec['predecessor'] is None,
                     'P': p_instance, 'P_route': events['prefill_begin']['url'], 'D_route': events['decode_begin']['url'],
                     'prompt_tokens': spec['prompt_tokens'], 'output_tokens': outcome['completion_tokens'],
                     'ready_s': outcome['ready_s'], 'end_s': outcome['end_s'], 'client_TTFT_s': client_ttft,
                     'P_queue_s': queue, 'P_HTTP_s': p_http, 'P_outside_queue_s': p_http - queue,
                     'D_first_chunk_s': decode_first, 'D_later_stream_s': outcome['end_s'] - outcome['first_s'],
                     'PD_read_s': native.get('slowest_rank_s'), 'PD_bytes': native.get('payload_bytes'),
                     'PD_ranks': native.get('n_ranks', 0),
                     'P_GET_s': get.get('slowest_rank_get_ms', 0) / 1000, 'P_GET_bytes': get.get('payload_bytes', 0),
                     'P_GET_ranks': get.get('ranks', 0)})
    sessions = defaultdict(list)
    for row in rows:
        sessions[row['session_id']].append(row)
    fields = ('P_queue_s', 'P_HTTP_s', 'P_outside_queue_s', 'D_first_chunk_s', 'D_later_stream_s',
              'client_TTFT_s', 'PD_read_s', 'P_GET_s', 'output_tokens')
    chain = {key: {'first_ready_s': seq[0]['ready_s'], 'last_end_s': seq[-1]['end_s'], 'requests': len(seq),
                   **{field: sum(r[field] for r in seq if r[field] is not None) for field in fields}}
             for key, seq in sessions.items()}
    last_session = max(chain, key=lambda key: chain[key]['last_end_s'])
    groups = {'first_turn': [row for row in rows if row['first_turn']],
              'continuations': [row for row in rows if not row['first_turn']]}
    full_forward = [b['device']['forward_device_ms'] for b in batches if b['total'] == b['budget']
                    and b.get('device', {}).get('forward_device_ms') is not None]
    return {'run_id': manifest['run_id'], 'rows': rows, 'successful_requests': len(rows),
            'phases_s_by_cohort': {name: {f: stats(r[f] for r in seq if r[f] is not None) for f in fields[:-1]}
                                   for name, seq in groups.items()},
            'drain_session': last_session, 'all_session_chains': chain,
            'P_rank0_before_forward': load_before_forward(p, batches),
            'D_native_P_to_D': {k: v for k, v in transfers.items() if k != 'per_request'},
            'P_batches': {'n': len(batches), 'rank0_device_timings': sum('device' in b for b in batches),
                          'pairing_errors': errors,
                          'full_budget_rank0_forward_ms': stats(full_forward),
                          'runtime_modes': dict(Counter(r['runtime_mode'] for r in p['native'] if r['type'] == 'p_chunk_graph'))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefill', type=Path)
    parser.add_argument('decode', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    p, d = read_bundle(args.prefill), read_bundle(args.decode)
    general = analyze(args.prefill, args.decode)
    result = timeline(p, d, general)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'run_id': result['run_id'], 'success': result['successful_requests'],
                      'P_GET_timing': result['P_rank0_before_forward'],
                      'D_native_P_to_D': result['D_native_P_to_D'],
                      'drain_session': result['drain_session'],
                      'drain_chain': result['all_session_chains'][result['drain_session']]}, indent=2))


if __name__ == '__main__':
    main()
