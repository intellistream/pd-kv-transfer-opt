"""Join the natural 2P2D replay by request without logging prompt content."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from urllib.parse import urlsplit

NS_PER_S = 1_000_000_000
MIB = 1024**2
ATTENTION_OBJECT_BYTES = 10 * MIB
ALIGN_STATE_OBJECT_BYTES = 10 * (524288 + 12288)
REQUEST_ID = re.compile(r'^cmpl-([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})-0(?:-|$)')
METRIC = re.compile(r'^([^ {]+)(?:\{(.*)\})? (\S+)$')


def distribution(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    return {'n': len(values), 'mean': statistics.mean(values), 'p50': statistics.median(values),
            'p95': values[math.ceil(len(values) * 0.95) - 1], 'max': values[-1]}


def read_bundle(path):
    result = defaultdict(list)
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            result[row['kind']].append(row['data'])
    return result


def normalized_proxy_id(value):
    match = REQUEST_ID.match(value or '')
    return match[1] if match else None


def parse_metrics(lines):
    values = {}
    for line in lines:
        match = METRIC.fullmatch(line)
        if match:
            try:
                value = float(match[3])
            except ValueError:
                continue
            if math.isfinite(value):
                values[(match[1], match[2] or '')] = value
    return values


def metric_sum(values, name, labels=''):
    return sum(value for (key, label), value in values.items() if key == name and labels in label)


def engine_timing(bundle, role, start, end):
    instances = defaultdict(list)
    for row in bundle['metrics']:
        if row['instance'].startswith(role):
            instances[row['instance']].append((row['unix_ns'], parse_metrics(row['lines'])))
    sums = Counter()
    gauges = defaultdict(list)
    points_by_instance = {}
    for instance, samples in instances.items():
        samples.sort(key=lambda row: row[0])
        timestamp, latest = samples[-1]
        points_by_instance[instance] = {'first_unix_ns': samples[0][0], 'final_unix_ns': timestamp,
                                       'samples_during_case': sum(start <= t <= end for t, _ in samples)}
        # A fresh engine handles only this replay; the last cumulative snapshot includes it.
        for (key, labels), value in latest.items():
            sums[(key, labels.split('source=')[-1] if 'source=' in labels else '')] += value
        for stamp, values in samples:
            if start <= stamp <= end:
                for gauge in ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc'):
                    gauges[(instance, gauge)].append(metric_sum(values, 'vllm:' + gauge))
    stages = {}
    for stage in ('queue_time', 'prefill_time', 'decode_time'):
        base = 'vllm:request_' + stage + '_seconds_'
        n = sums[(base + 'count', '')]
        stages[stage] = {'count': n, 'total_s': sums[(base + 'sum', '')],
                         'mean_s': sums[(base + 'sum', '')] / n if n else None}
    return {'stages': stages, 'prompt_tokens_by_source': {source: sums[('vllm:prompt_tokens_by_source_total', f'"{source}"')]
                                                         for source in ('local_compute', 'local_cache_hit', 'external_kv_transfer')},
            'computed_prefill_tokens': sums[('vllm:request_prefill_kv_computed_tokens_sum', '')],
            'sample_bounds': points_by_instance,
            'gauges': {f'{name}:{metric}': distribution(series) for (name, metric), series in gauges.items()}}


def interval_union(intervals):
    covered = 0
    right = 0
    for start, end in sorted(intervals):
        covered += max(0, end - max(start, right))
        right = max(right, end)
    return covered


def classify_layout(row):
    for states in range(4):
        attention = row['keys'] - states
        attention_bytes = attention * ATTENTION_OBJECT_BYTES
        state_bytes = states * ALIGN_STATE_OBJECT_BYTES
        if attention >= 0 and attention_bytes + state_bytes == row['payload_bytes']:
            return attention_bytes, state_bytes
    return None


def summarize_p_workers(bundle, proxy_to_client, start, end):
    get_calls, puts, batches = [], [], []
    pending = {}
    batch_mismatch = Counter()
    request_times = defaultdict(lambda: defaultdict(float))
    request_payload = defaultdict(lambda: defaultdict(int))
    request_attention_payload = defaultdict(int)
    for row in sorted(bundle['native'], key=lambda r: r['monotonic_ns']):
        key = (row['pid'], row['thread_id'])
        if row['type'] == 'worker_load_begin':
            pending[key] = (row, [])
        elif row['type'] == 'store_end':
            if row['operation'] == 'batch_get_into_multi_buffers':
                get_calls.append(row)
                if key in pending:
                    pending[key][1].append(row)
            elif row['operation'] == 'batch_put_from_multi_buffers':
                puts.append(row)
        elif row['type'] == 'worker_load_end' and key in pending:
            batch, calls = pending.pop(key)
            reqs = row['request_ids']
            batches.append({'instance': row['instance'], 'pid': row['pid'], 'request_count': len(reqs),
                            'start_unix_ns': batch['unix_ns'], 'end_unix_ns': row['unix_ns'],
                            'worker_ms': row['elapsed_ns'] / 1e6, 'get_ms': sum(r['elapsed_ns'] for r in calls) / 1e6})
            if len(reqs) != len(calls):
                batch_mismatch['counts_do_not_match'] += 1
                continue
            for req, call in zip(reqs, calls):
                client = proxy_to_client.get(normalized_proxy_id(req))
                if client is None:
                    batch_mismatch['proxy_id_missing'] += 1
                    continue
                request_times[client][row['pid']] += call['elapsed_ns'] / 1e6
                request_payload[client][row['pid']] += call['payload_bytes']
                components = classify_layout(call)
                if components is not None:
                    request_attention_payload[client] += components[0]
    rank_intervals = defaultdict(list)
    layout_reconstructed = Counter()
    for row in get_calls:
        call_end = row['unix_ns']
        call_start = call_end - row['elapsed_ns']
        if start <= call_end <= end:
            rank_intervals[(row['instance'], row['pid'])].append((max(start, call_start), call_end))
        components = classify_layout(row)
        if components is not None:
            layout_reconstructed['calls'] += 1
            layout_reconstructed['attention_bytes'] += components[0]
            layout_reconstructed['align_state_bytes'] += components[1]
    fractions = {f'{instance}:{pid}': interval_union(intervals) / (end - start)
                 for (instance, pid), intervals in rank_intervals.items()}
    per_request = {client: {'ranks': len(rank_values), 'slowest_rank_get_ms': max(rank_values.values()),
                            'payload_bytes': sum(request_payload[client].values()),
                            'attention_payload_bytes': request_attention_payload[client]}
                   for client, rank_values in request_times.items()}
    return {'get_calls': len(get_calls), 'get_ms': distribution(r['elapsed_ns'] / 1e6 for r in get_calls),
            'get_payload_bytes': sum(r['payload_bytes'] for r in get_calls),
            'get_negative_returns': sum(r['negative_returns'] for r in get_calls),
            'get_wall_fraction_by_rank': fractions,
            'get_batch_ms': distribution(r['worker_ms'] for r in batches),
            'get_batch_size': distribution(r['request_count'] for r in batches), 'get_classified_layout': layout_reconstructed,
            'put': summarize_puts(puts), 'association_limitations': batch_mismatch,
            'restored_requests_four_ranks': sum(r['ranks'] == 4 for r in per_request.values()),
            'slowest_rank_per_restored_request_ms': distribution(r['slowest_rank_get_ms'] for r in per_request.values()
                                                                 if r['ranks'] == 4),
            'per_request': per_request}


def summarize_puts(rows):
    return {'calls': len(rows), 'keys': sum(r['keys'] for r in rows), 'negative_returns': sum(r['negative_returns'] for r in rows),
            'submitted_bytes': sum(r['payload_bytes'] for r in rows),
            'ms': distribution(r['elapsed_ns'] / 1e6 for r in rows)}


def summarize_d_workers(bundle):
    pending = {}
    transfers = []
    for row in sorted(bundle['native'], key=lambda r: r['monotonic_ns']):
        kind = row['type']
        if not kind.startswith('native_') or 'remote_request_id' not in row:
            continue
        pair = (row['pid'], row['remote_request_id'])
        if kind == 'native_transfer_enqueue':
            pending[pair] = {'enqueue': row['monotonic_ns']}
        elif kind == 'native_handle_begin' and pair in pending:
            pending[pair]['begin'] = row['monotonic_ns']
        elif kind == 'native_transfer_submit' and pair in pending:
            pending[pair]['submitted_bytes'] = row['payload_bytes']
            pending[pair]['descriptors'] = row['descriptor_count']
        elif kind == 'native_handle_end' and pair in pending:
            entry = pending.pop(pair)
            if 'begin' in entry and 'submitted_bytes' in entry:
                transfers.append(dict(entry, end=row['monotonic_ns']))
    puts = (r for r in bundle['native'] if r['type'] == 'store_end' and r['operation'] == 'batch_put_from_multi_buffers')
    return {'matched_submissions': len(transfers), 'submitted_bytes': sum(t['submitted_bytes'] for t in transfers),
            'enqueue_to_handler_ms': distribution((t['begin'] - t['enqueue']) / 1e6 for t in transfers),
            'handler_elapsed_ms': distribution((t['end'] - t['begin']) / 1e6 for t in transfers),
            'put': summarize_puts(list(puts))}


def master_usage(bundle, start, end):
    all_counters = []
    segments = defaultdict(list)
    for row in bundle['metrics']:
        if row['instance'] != 'master' or not start <= row['unix_ns'] <= end + 10 * NS_PER_S:
            continue
        metrics = parse_metrics(row['lines'])
        capacities = {label: val for (name, label), val in metrics.items() if name == 'segment_total_capacity_bytes'}
        used = {label: val for (name, label), val in metrics.items() if name == 'segment_allocated_bytes'}
        for label, capacity in capacities.items():
            if capacity:
                segments[label].append(used.get(label, 0) / capacity)
        all_counters.append({'unix_ns': row['unix_ns'], 'capacity_bytes': sum(capacities.values()),
                             'allocated_bytes': sum(used.values()),
                             'put_start_alloc_failures': metric_sum(metrics, 'master_put_start_alloc_failures_total'),
                             'successful_evictions': metric_sum(metrics, 'master_successful_evictions_total')})
    all_counters.sort(key=lambda r: r['unix_ns'])
    return {'last': all_counters[-1] if all_counters else None,
            'peak_pool_allocated_bytes': max((r['allocated_bytes'] for r in all_counters), default=None),
            'max_fraction_by_segment': {k: max(v) for k, v in segments.items()}, 'samples': all_counters}


def fabric_summary(bundle, start, end):
    counters = defaultdict(list)
    for row in bundle['fabric']:
        if start <= row['unix_ns'] <= end:
            counters[row['device']].append(row)
    fabric = {}
    for device, rows in counters.items():
        rows.sort(key=lambda r: r['unix_ns'])
        if len(rows) < 2:
            continue
        first, last = rows[0], rows[-1]
        seconds = (last['unix_ns'] - first['unix_ns']) / NS_PER_S
        rates = {}
        for direction in ('rx', 'tx'):
            name = f'mac_{direction}_total_oct_num'
            if name not in first['counters'] or name not in last['counters']:
                continue
            delta = last['counters'][name] - first['counters'][name]
            if delta >= 0:
                rates[direction + '_GBps'] = delta / seconds / 1e9
                intervals = [(y['counters'][name] - x['counters'][name]) / (y['unix_ns'] - x['unix_ns'])
                             for x, y in zip(rows, rows[1:]) if name in x['counters'] and name in y['counters']
                             and y['counters'][name] >= x['counters'][name] and y['unix_ns'] > x['unix_ns']]
                rates[direction + '_peak_sample_GBps'] = max(intervals, default=None)
        fabric[device] = {'measured_seconds': seconds, **rates}
    return {'devices': fabric, 'sum_port_mean_GBps': {direction: sum(r.get(direction + '_GBps', 0) for r in fabric.values())
                                                     for direction in ('rx', 'tx')}}


def npu_summary(bundle, start, end):
    devices = defaultdict(list)
    for row in bundle['npu']:
        if start <= row['unix_ns'] <= end:
            devices[row['device']].append(row)
    return {
        device: {
            'aicore_sample_percent': distribution(row['aicore_percent'] for row in rows),
            'hbm_used_mb': distribution(row['hbm_used_mb'] for row in rows if row['hbm_used_mb'] is not None),
        }
        for device, rows in devices.items()
    }


def runtime_signature(state):
    run = state['run_id']
    instances = []
    for instance in state['instances']:
        argv = [argument.replace(run, '<run>') for argument in instance['argv']
                if argument not in ('--enable-prefix-caching', '--no-enable-prefix-caching')]
        instances.append({'id': instance['id'], 'argv': argv, 'devices': instance['devices'], 'cpus': instance['cpus']})
    fields = {'instances': instances, 'execution_mode': state['execution_mode'], 'store_config': state['store_config']}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def analyze(prefill_path, decode_path):
    p, d = read_bundle(prefill_path), read_bundle(decode_path)
    assert p['scope'][0]['run_id'] == d['scope'][0]['run_id']
    manifest, = d['client_manifest']
    origin, = (row for row in d['client'] if row['type'] == 'client_origin')
    ends = {row['id']: row for row in d['client'] if row['type'] == 'request_end'}
    assert len(ends) == manifest['selected_requests']
    start = origin['anchor']['unix_ns']
    end = start + round(max(r['end_s'] for r in ends.values()) * NS_PER_S)
    proxies = defaultdict(dict)
    ids = {}
    for row in d['proxy']:
        if row['id'] in ends:
            proxies[row['id']][row['type']] = row
            pid = row.get('proxy_request_id')
            if pid:
                ids[pid] = row['id']
    pworkers = summarize_p_workers(p, ids, start, end)
    case_groups = defaultdict(lambda: defaultdict(list))
    restore_cover = defaultdict(Counter)
    possible_reuse = Counter()
    routing = defaultdict(Counter)
    tails = []
    for row in manifest['request_sources']:
        client = row['id']
        if ends[client]['status'] != 'ok':
            continue
        group = 'session_first_turn' if row['predecessor'] is None else 'session_continuation'
        outcome = ends[client]
        events = proxies[client]
        restore = pworkers['per_request'].get(client)
        ttft = outcome['first_s'] - outcome['ready_s']
        phases = {}
        for name, first, last in [('P_HTTP_s', 'prefill_begin', 'prefill_end'),
                                  ('P_to_D_proxy_gap_s', 'prefill_end', 'decode_begin'),
                                  ('D_first_chunk_s', 'decode_begin', 'decode_first_chunk'),
                                  ('D_HTTP_s', 'decode_begin', 'decode_end')]:
            if first in events and last in events:
                phases[name] = (events[last]['monotonic_ns'] - events[first]['monotonic_ns']) / NS_PER_S
        tails.append({'source_id': row['source_id'], 'session_first_turn': row['predecessor'] is None,
                      'input_tokens': row['prompt_tokens'], 'ready_s': outcome['ready_s'], 'TTFT_s': ttft,
                      'P_HTTP_s': phases.get('P_HTTP_s'),
                      'slowest_rank_GET_ms': restore['slowest_rank_get_ms'] if restore and restore['ranks'] == 4 else None})
        if group == 'session_continuation':
            shared = row['source_prompt_lcp_to_previous']
            aligned_shared = shared // 1024 * 1024
            possible_reuse['source_shared_tokens_at_1024_boundary'] += aligned_shared
            if (not restore or restore['ranks'] != 4) and shared >= 32768:
                possible_reuse['no_four_rank_get_with_common_prefix_ge_32k'] += 1
                possible_reuse['their_shared_tokens_at_1024_boundary'] += aligned_shared
            current_url = events.get('prefill_begin', {}).get('url')
            previous_url = proxies.get(row['predecessor'], {}).get('prefill_begin', {}).get('url')
            if current_url and previous_url:
                current = urlsplit(current_url).netloc
                previous = urlsplit(previous_url).netloc
                route = routing['same_P_as_previous_turn' if current == previous else 'different_P_from_previous_turn']
                route['requests'] += 1
                route['source_shared_tokens_at_1024_boundary'] += aligned_shared
                if restore and restore['ranks'] == 4:
                    route['observed_four_rank_GET'] += 1
        for key in ('all', group):
            gathered = case_groups[key]
            gathered['TTFT_s'].append(ttft)
            gathered['actual_output_tokens'].append(outcome['completion_tokens'])
            if outcome['completion_tokens'] > 1:
                gathered['client_TPOT_including_last_chunk_s'].append((outcome['end_s'] - outcome['first_s']) /
                                                                      (outcome['completion_tokens'] - 1))
            for name, seconds in phases.items():
                gathered[name].append(seconds)
            restore_cover[key]['successful_requests'] += 1
            if restore and restore['ranks'] == 4:
                restore_cover[key]['observed_four_rank_GET'] += 1
                gathered['slowest_rank_get_ms'].append(restore['slowest_rank_get_ms'])
    summ, = d['client_summary']
    flags = {role: ['--enable-prefix-caching' in instance['argv'] for instance in bundle['runtime_state'][0]['instances']]
             for role, bundle in [('P', p), ('D', d)]}
    runtime = {role: runtime_signature(bundle['runtime_state'][0]) for role, bundle in [('P', p), ('D', d)]}
    return {'run_id': p['scope'][0]['run_id'], 'workload_sha256': manifest['workload_sha256'],
            'timing_window_unix_ns': [start, end], 'source_arrival_sha256': hashlib.sha256(json.dumps(manifest['request_sources'],
                                                                                                     sort_keys=True).encode()).hexdigest(),
            'requests': summ['counts'], 'drain_s': summ['finite_cohort']['drain_included_s'],
            'output_token_count': summ['window']['client_delivered_token_ids'], 'output_tokens_s': summ['window']['client_delivered_tokens_s'],
            'completed_requests_s': summ['window']['completed_requests_s'], 'TTFT': summ['finite_cohort']['ttft_from_ready'],
            'P_GET': pworkers, 'D_native': summarize_d_workers(d), 'pool_master': master_usage(d, start, end),
            'server': {role: engine_timing(bundle, role, start, end) for role, bundle in [('P', p), ('D', d)]},
            'fabric': {role: fabric_summary(bundle, start, end) for role, bundle in [('P', p), ('D', d)]},
            'npu': {role: npu_summary(bundle, start, end) for role, bundle in [('P', p), ('D', d)]},
            'request_phases': {key: {metric: distribution(rows) for metric, rows in values.items()} for key, values in case_groups.items()},
            'slowest_TTFT_requests': sorted(tails, key=lambda request: request['TTFT_s'], reverse=True)[:10],
            'prefix_caching_flags_by_role': flags,
            'runtime_signature_excluding_prefix_caching': runtime,
            'restore_match_coverage': restore_cover, 'possible_reuse_from_source_prefix': possible_reuse,
            'continuation_routing': routing, 'output_length_and_finish_reasons': {
                'tokens': summ['completion_tokens'], 'finish_reasons': summ['finish_reasons'],
            },
            'scope': 'Finite trace-driven replay. Timings use same-host clocks and successful request cohorts. Port counters cover all traffic; API submitted bytes are separate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefill', type=Path)
    parser.add_argument('decode', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = analyze(args.prefill, args.decode)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('run_id','requests','drain_s','output_token_count','output_tokens_s','TTFT',
                                                 'restore_match_coverage')}, indent=2))


if __name__ == '__main__':
    main()
