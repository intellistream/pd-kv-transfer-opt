"""Join cold-turn scheduler and rank-0 NPU events to exact client requests."""
import argparse
from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
import math
from pathlib import Path

from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import distribution, interval_union, normalized_proxy_id, read_bundle


def stats(values):
    values = sorted(values)
    result = distribution(values)
    if values:
        result['p99'] = values[math.ceil(len(values) * 0.99) - 1]
    return result


def get_batches(data):
    batches = []
    pending = defaultdict(deque)
    devices = defaultdict(deque)
    arrived = {}
    errors = []
    for event in sorted(data, key=lambda row: row['monotonic_ns']):
        kind = event['type']
        if kind == 'p_chunk_arrival':
            arrived[event['request_id']] = event
        elif kind == 'p_chunk_dispatch':
            pending[event['instance'], event['pid'], event['batch']].append(event)
            batches.append(event)
        elif kind == 'p_chunk_ready':
            key = (event['instance'], event['pid'], event['batch'])
            if pending[key]:
                pending[key].popleft()['ready_ns'] = event['ready_ns']
            else:
                errors.append(('model_output_without_dispatch', event['batch']))
        elif kind == 'p_chunk_device':
            devices[event['instance'], event['batch']].append(event)
        elif kind == 'p_chunk_timing_unavailable':
            errors.append(('timing_unavailable', event['message']))
    for batch in batches:
        options = devices[batch['instance'], batch['batch']]
        if options and options[0]['worker_enter_ns'] >= batch['dispatch_ns']:
            batch['device'] = options.popleft()
    return arrived, batches, errors


def cold_chunk_summary(p, d):
    manifest, = d['client_manifest']
    sources = {row['id']: row for row in manifest['request_sources']}
    proxy = defaultdict(dict)
    names = {}
    for event in d['proxy']:
        if event['id'] in sources:
            proxy[event['id']][event['type']] = event
            if event['type'] == 'prefill_begin':
                names[event['proxy_request_id']] = event['id']
    ends = {r['id']: r for r in d['client'] if r['type'] == 'request_end' and r['status'] == 'ok'}
    arrivals, batches, errors = get_batches(p['native'])
    per_request = defaultdict(list)
    for batch in batches:
        for rid, offset, count in batch['rows']:
            if rid in arrivals and offset is not None and offset < arrivals[rid]['prompt_tokens']:
                per_request[rid].append((batch, offset, count))

    cold = []
    for rid, arrival in arrivals.items():
        client = names.get(normalized_proxy_id(rid))
        if client not in ends or sources[client]['predecessor'] is not None:
            continue
        chunks = per_request[rid]
        if not chunks or any('ready_ns' not in entry[0] for entry in chunks):
            errors.append(('incomplete_cold_chunks', client))
            continue
        outcome = ends[client]
        begin = chunks[0][0]['dispatch_ns']
        ready = max(batch['ready_ns'] for batch, _, _ in chunks)
        covered = interval_union((batch['dispatch_ns'], batch['ready_ns']) for batch, _, _ in chunks)
        device_ms = [batch.get('device', {}).get('forward_device_ms') for batch, _, _ in chunks]
        remaining = arrivals[rid]['prompt_tokens']
        total = sum(min(remaining - before, count) for _, before, count in chunks)
        unmatched_prompt_tail = remaining - chunks[0][1] - total
        consecutive = all(left[1] + left[2] == right[1] for left, right in zip(chunks, chunks[1:]))
        if unmatched_prompt_tail not in (0, 1) or not consecutive:
            errors.append(('cold_token_accounting', client))
        events = proxy[client]
        p_start = events['prefill_begin']['monotonic_ns']
        p_http = (events['prefill_end']['monotonic_ns'] - p_start) / 1e9
        instance = chunks[0][0]['instance']
        queue_ns = begin - arrival['arrival_ns']
        engine_s = (ready - arrival['arrival_ns']) / 1e9
        cold.append({'id': client, 'instance': instance, 'prompt_tokens': arrival['prompt_tokens'],
                     'first_cached_tokens': chunks[0][1], 'new_prompt_tokens': total,
                     'prompt_tail_tokens_outside_observed_chunks': unmatched_prompt_tail,
                     'n_chunks': len(chunks), 'timed_device_chunks': sum(x is not None for x in device_ms),
                     'ttft_s': outcome['first_s'] - outcome['ready_s'], 'P_HTTP_s': p_http,
                     'scheduler_queue_s': queue_ns / 1e9,
                     'chunk_service_wall_s': covered / 1e9, 'inter_chunk_gap_s': (ready - begin - covered) / 1e9,
                     'rank0_forward_s': sum(device_ms) / 1000 if None not in device_ms else None,
                     'P_http_outside_scheduler_to_ready_s': p_http - engine_s,
                     'arrival_ns': arrival['arrival_ns'], 'first_dispatch_ns': begin, 'last_ready_ns': ready,
                     'chunks': [{'batch': batch['batch'], 'offset': offset, 'tokens': count,
                                 'batch_tokens': batch['total'], 'batch_requests': len(batch['rows']),
                                 'dispatch_to_ready_ms': (batch['ready_ns'] - batch['dispatch_ns']) / 1e6,
                                 'rank0_forward_ms': ms} for (batch, offset, count), ms in zip(chunks, device_ms, strict=True)]})
    if not cold:
        raise ValueError(f'No cold requests matched. Arrivals: {len(arrivals)}, batches: {len(batches)}, proxy mappings: {len(names)}')
    # The time union across other requests' batches during each request's queue is
    # evidence of work ahead of it, even if the scheduler pipelines batches.
    by_instance = defaultdict(list)
    for step in batches:
        if 'ready_ns' in step:
            by_instance[step['instance']].append(step)
    occupancy = {}
    for instance, steps in by_instance.items():
        local = [r for r in cold if r['instance'] == instance]
        if not local:
            continue
        for row in local:
            q0, q1 = row['arrival_ns'], row['first_dispatch_ns']
            busy = interval_union((max(step['dispatch_ns'], q0), min(step['ready_ns'], q1))
                                  for step in steps if step['dispatch_ns'] < q1 and step['ready_ns'] > q0)
            row['other_batches_during_queue_s'] = busy / 1e9
        latest_first = max(r['first_dispatch_ns'] for r in local)
        relevant = [b for b in steps if b['dispatch_ns'] < latest_first and b['waiting'] > 0]
        occupancy[instance] = {'cold_requests': len(local), 'cold_prompt_tokens': sum(r['prompt_tokens'] for r in local),
                               'queued_dispatches': len(relevant),
                               'queued_dispatches_using_full_budget': sum(b['budget'] == b['total'] for b in relevant),
                               'queued_dispatches_single_request': sum(len(b['rows']) == 1 for b in relevant),
                               'queued_dispatches_tokens': stats(b['total'] for b in relevant),
                               'queued_dispatches_running': stats(b['running'] for b in relevant)}
    measurements = ('ttft_s', 'P_HTTP_s', 'scheduler_queue_s', 'other_batches_during_queue_s',
                    'chunk_service_wall_s', 'inter_chunk_gap_s', 'rank0_forward_s',
                    'P_http_outside_scheduler_to_ready_s', 'n_chunks')
    warm = [ends[c]['first_s'] - ends[c]['ready_s'] for c in ends if sources[c]['predecessor'] is not None]
    summary = {m: stats(row[m] for row in cold if row[m] is not None) for m in measurements}
    chunk_rows = [c for r in cold for c in r['chunks']]
    summary['chunk_wall_ms'] = stats(c['dispatch_to_ready_ms'] for c in chunk_rows)
    summary['chunk_rank0_forward_ms'] = stats(c['rank0_forward_ms'] for c in chunk_rows if c['rank0_forward_ms'] is not None)
    summary['first_cached_tokens'] = stats(r['first_cached_tokens'] for r in cold)
    summary['prompt_tail_tokens_outside_observed_chunks'] = dict(Counter(r['prompt_tail_tokens_outside_observed_chunks'] for r in cold))
    summary['continuation_ttft_s'] = stats(warm)
    summary['cold_paired_ttft_rank'] = sorted(cold, key=lambda r: r['ttft_s'], reverse=True)
    summary['quality'] = {'arrivals': len(arrivals), 'dispatches': len(batches),
                          'dispatches_with_model_output': sum('ready_ns' in b for b in batches),
                          'dispatches_with_rank0_device': sum(b.get('device', {}).get('forward_device_ms') is not None for b in batches),
                          'cold_prompt_chunks': len(chunk_rows),
                          'cold_prompt_chunks_with_rank0_device': sum(c['rank0_forward_ms'] is not None for c in chunk_rows),
                          'preemptions': sum(len(b['preempted']) for b in batches), 'errors': errors}
    summary['queue_pressure_by_P'] = occupancy
    return summary


def pd_transfers(d):
    """D's point-to-point READ durations from submit log through handler end."""
    pending = {}
    per_request = defaultdict(list)
    for row in sorted(d['native'], key=lambda r: r['monotonic_ns']):
        if row['type'] == 'native_transfer_submit':
            key = row['pid'], row['remote_request_id']
            pending[key] = (row['monotonic_ns'], row['payload_bytes'])
        elif row['type'] == 'native_handle_end':
            key = row['pid'], row['remote_request_id']
            if key in pending:
                start, size = pending.pop(key)
                per_request[normalized_proxy_id(key[1])].append((size, row['monotonic_ns'] - start))
    matched = {}
    for row in d['proxy']:
        if row['type'] != 'prefill_begin' or row['proxy_request_id'] not in per_request:
            continue
        ranks = per_request[row['proxy_request_id']]
        matched[row['id']] = {'n_ranks': len(ranks), 'payload_bytes': sum(size for size, _ in ranks),
                              'slowest_rank_s': max(elapsed for _, elapsed in ranks) / 1e9}
    return {'matched_ranks': sum(len(v) for v in per_request.values()),
            'matched_requests': len(matched), 'pending': len(pending),
            'per_request': matched,
            'slowest_rank_ms': stats(row['slowest_rank_s'] * 1000 for row in matched.values() if row['n_ranks'] == 4),
            'per_call_payload_GBps': stats(size / ns for ranks in per_request.values() for size, ns in ranks if ns > 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefill', type=Path)
    parser.add_argument('decode', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    p, d = read_bundle(args.prefill), read_bundle(args.decode)
    assert p['scope'][0]['run_id'] == d['scope'][0]['run_id']
    manifest, = d['client_manifest']
    client, = d['client_summary']
    summary = cold_chunk_summary(p, d)
    result = {'run_id': manifest['run_id'], 'workload_sha256': manifest['workload_sha256'],
              'source_arrival_sha256': sha256(json.dumps(manifest['request_sources'], sort_keys=True).encode()).hexdigest(),
              'P_batch_budget': manifest['p_batched_tokens'], 'background_traffic': manifest['background_traffic'],
              'client': {key: client[key] for key in ('counts', 'quality', 'elapsed_s', 'window', 'finite_cohort')},
              'cold': summary, 'P_to_D': pd_transfers(d),
              'P_sched_tokens': dict(Counter(t for row in p['native'] if row['type'] == 'p_chunk_dispatch' for _, _, t in row['rows'])),
              'rdma_pressure': d['rdma_pressure'],
              'percentiles': 'p95 and p99 are nearest rank, p50 is median for stage and chunk breakdown'}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'run_id': result['run_id'], 'counts': client['counts'], 'TTFT': client['finite_cohort']['ttft_from_ready'],
                      'tokens_s': client['window']['client_delivered_tokens_s'], 'quality': summary['quality'],
                      'cold_p50_s': {key: value.get('p50') for key, value in summary.items()
                                     if key.endswith('_s') and isinstance(value, dict)},
                      'queue_pressure_by_P': summary['queue_pressure_by_P']}, indent=2))


if __name__ == '__main__':
    main()
