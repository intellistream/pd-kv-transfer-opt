"""Join each request across the client, proxy, P worker, and D transfer handler."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import interval_union, normalized_proxy_id, read_bundle
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import get_batches, stats
from pd_kv_transfer_opt.profiling.analyze_rdma_hard_caps import analyze_case
from pd_kv_transfer_opt.experiments.rdma_full_protocol import CASES, ORDERS, PROMPT_LENGTHS, port_comparison


def associate_loads(events):
    pending = {}
    requests = defaultdict(list)
    batches = []
    mismatches = []
    for row in events:
        thread = (row['pid'], row['thread_id'])
        kind = row['type']
        if kind == 'worker_load_begin':
            pending[thread] = {'start': row, 'GET': [], 'backend': []}
        elif thread in pending:
            item = pending[thread]
            if kind == 'store_end' and row['operation'] == 'batch_get_into_multi_buffers':
                item['GET'].append(row)
            elif kind == 'mooncake_backend_get':
                item['backend'].append(row)
            elif kind == 'worker_load_end':
                ids = [normalized_proxy_id(value) for value in row['request_ids']]
                gets = item['GET']
                backend = item['backend']
                if len(gets) != len(ids):
                    mismatches.append({'pid': row['pid'], 'requests': ids, 'gets': len(gets)})
                else:
                    worker_ns = row['elapsed_ns']
                    get_ns = sum(value['elapsed_ns'] for value in gets)
                    backend_ns = sum(value['elapsed_ns'] for value in backend)
                    batch = {'instance': row['instance'], 'pid': row['pid'], 'thread_id': row['thread_id'],
                             'proxy_ids': ids, 'begin_ns': item['start']['monotonic_ns'],
                             'end_ns': row['monotonic_ns'], 'worker_ms': worker_ns / 1e6,
                             'native_GET_sum_ms': get_ns / 1e6,
                             'backend_ms': backend_ns / 1e6 if len(backend) == len(ids) else None,
                             'worker_other_ms': (worker_ns - get_ns) / 1e6}
                    batches.append(batch)
                    have_backend = len(backend) == len(ids)
                    for i, (pid, native) in enumerate(zip(ids, gets, strict=True)):
                        native_ns = native['elapsed_ns']
                        backend_call_ns = backend[i]['elapsed_ns'] if have_backend else None
                        requests[pid].append({
                            'instance': row['instance'], 'pid': row['pid'], 'thread_id': row['thread_id'],
                            'position': i, 'requests_in_load': len(ids), 'keys': native['keys'],
                            'payload_bytes': native['payload_bytes'], 'negative_returns': native['negative_returns'],
                            'begin_ns': native['monotonic_ns'] - native_ns, 'end_ns': native['monotonic_ns'],
                            'begin_unix_ns': native['unix_ns'] - native_ns, 'end_unix_ns': native['unix_ns'],
                            'native_GET_ms': native_ns / 1e6, 'native_GET_Gbps': 8 * native['payload_bytes'] / native_ns,
                            'backend_ms': backend_call_ns / 1e6 if have_backend else None,
                            'backend_other_ms': (backend_call_ns - native_ns) / 1e6 if have_backend else None,
                            'other_GETs_in_same_load_ms': (get_ns - native_ns) / 1e6,
                        })
                del pending[thread]
    return requests, batches, mismatches


def associate_transfers(events):
    transactions = defaultdict(dict)
    for row in events:
        kind = row['type']
        if kind in {'native_transfer_enqueue', 'native_handle_begin', 'native_transfer_submit', 'native_handle_end'}:
            identifier = normalized_proxy_id(row['remote_request_id'])
            if identifier:
                transactions[(identifier, row['pid'])][kind] = row
    transfers = defaultdict(list)
    for (identifier, pid), records in transactions.items():
        enqueue = records.get('native_transfer_enqueue')
        begin = records.get('native_handle_begin')
        submit = records.get('native_transfer_submit')
        end = records.get('native_handle_end')
        if all((enqueue, begin, submit, end)):
            enqueue_ns, begin_ns, submit_ns, end_ns = (
                r['monotonic_ns'] for r in (enqueue, begin, submit, end)
            )
            transfers[identifier].append({
                'instance': submit['instance'], 'pid': pid, 'payload_bytes': submit['payload_bytes'],
                'enqueue_ns': enqueue_ns, 'handler_begin_ns': begin_ns, 'submit_ns': submit_ns, 'end_ns': end_ns,
                'submit_unix_ns': submit['unix_ns'], 'end_unix_ns': end['unix_ns'],
                'queue_ms': (begin_ns - enqueue_ns) / 1e6, 'pre_submit_ms': (submit_ns - begin_ns) / 1e6,
                'submit_to_end_ms': (end_ns - submit_ns) / 1e6,
            })
    return transfers


def find_methods_and_batches(p_events):
    arrivals, batches, errors = get_batches(p_events)
    arr = {normalized_proxy_id(rid): row for rid, row in arrivals.items()}
    steps = defaultdict(list)
    for batch in batches:
        for rid, _, _ in batch['rows']:
            steps[normalized_proxy_id(rid)].append(batch)
    lookups = defaultdict(list)
    for row in p_events:
        if row['type'] == 'pool_match_tokens':
            lookups[normalized_proxy_id(row['request_id'])].append(row)
    return arr, batches, steps, lookups, errors


def stats_fields(rows, columns):
    return {key: stats(row[key] for row in rows if row.get(key) is not None) for key in columns}


def detail_case(summary, decode, case, arrivals, steps, lookups, loads, load_batches, transfers):
    clients = [row for row in decode['client'] if row['case'] == case]
    anchor, = [row for row in clients if row['type'] == 'client_origin']
    origin_ns = round(anchor['monotonic_s'] * 1e9)
    by_client = {row['id']: row for row in clients if row['type'] == 'request_end'}
    proxy = defaultdict(dict)
    for row in decode['proxy']:
        if row['id'] in by_client:
            proxy[row['id']][row['type']] = row

    identifiers = {events['prefill_begin']['proxy_request_id'] for events in proxy.values()}
    rank0_by_instance = {step['instance']: step['device']['pid'] for pid in identifiers for step in steps[pid]
                         if 'device' in step}
    rank0_intervals = defaultdict(list)
    for pid in sorted(identifiers):
        for call in loads[pid]:
            instance = call['instance']
            if call['pid'] == rank0_by_instance.get(instance):
                rank0_intervals[instance].append((call['begin_unix_ns'], call['end_unix_ns']))
    for row in summary['requests_joined']:
        rid = row['id']
        c = by_client[rid]
        events = proxy[rid]
        pid = events['prefill_begin']['proxy_request_id']
        times = {key: value['monotonic_ns'] for key, value in events.items()}
        dispatch = origin_ns + round(c['dispatch_s'] * 1e9)
        first = origin_ns + round(c['first_s'] * 1e9)
        request_steps = steps[pid]
        first_step = request_steps[0]
        first_arrival = arrivals[pid]
        arrival_ns = first_arrival['arrival_ns']
        arrival_unix_ns = first_arrival['unix_ns'] + arrival_ns - first_arrival['monotonic_ns']
        ingress_begin_ns = events['prefill_begin']['unix_ns']
        first_dispatch = first_step['dispatch_ns']
        last_step = request_steps[-1]
        last_ready = last_step.get('ready_ns')
        engine_s = (last_ready - first_dispatch) / 1e9 if last_ready is not None else None
        result_unix_ns = last_step['unix_ns'] + last_ready - last_step['monotonic_ns'] if last_ready is not None else None
        ingress_overlap = interval_union((max(a, ingress_begin_ns), min(b, arrival_unix_ns))
                                         for a, b in rank0_intervals[first_step['instance']]
                                         if a < arrival_unix_ns and b > ingress_begin_ns)
        method_calls = lookups[pid]
        queue_lookup_ms = sum(call['elapsed_ns'] / 1e6 for call in method_calls
                              if arrival_ns <= call['monotonic_ns'] <= first_dispatch)
        device = first_step.get('device')
        load_calls = loads[pid]
        d_ranks = transfers[pid]
        row.update({
            'proxy_request_id': pid, 'P_instance': first_step['instance'],
            'P_prefill_batches': [{'batch': b['batch'], 'dispatch_ns': b['dispatch_ns'],
                                   'ready_ns': b.get('ready_ns'), 'batch_tokens': b['total'],
                                   'rank0_device_timing': b.get('device')} for b in request_steps],
            'P_GET_rank_calls': sorted(load_calls, key=lambda item: item['pid']),
            'D_native_rank_calls': sorted(d_ranks, key=lambda item: item['pid']),
            'lookup_calls': [{'elapsed_ms': r['elapsed_ns'] / 1e6, 'external_tokens': r['external_tokens'],
                              'local_tokens': r['local_tokens'], 'load_async': r['load_async']} for r in method_calls],
            'TTFT_s': c['first_s'] - c['ready_s'],
            'client_queue_s': c['dispatch_s'] - c['ready_s'],
            'client_dispatch_to_proxy_s': (times['proxy_arrival'] - dispatch) / 1e9,
            'proxy_arrival_to_P_s': (times['prefill_begin'] - times['proxy_arrival']) / 1e9,
            'P_HTTP_begin_to_engine_arrival_s': (arrival_unix_ns - ingress_begin_ns) / 1e9,
            'P_ingress_overlaps_same_instance_rank0_GET_s': ingress_overlap / 1e9,
            'P_queue_lookup_ms': queue_lookup_ms if method_calls else None,
            'P_dispatch_to_result_s': engine_s,
            'P_HTTP_outside_engine_s': row['P_HTTP_s'] - row['P_queue_s'] - engine_s if engine_s is not None else None,
            'P_engine_result_to_HTTP_reply_s': (events['prefill_end']['unix_ns'] - result_unix_ns) / 1e9
            if result_unix_ns is not None else None,
            'P_first_dispatch_to_worker_ms': (device['worker_enter_ns'] - first_dispatch) / 1e6 if device else None,
            'P_first_worker_to_forward_ms': (device['forward_enter_ns'] - device['worker_enter_ns']) / 1e6 if device else None,
            'P_first_forward_device_ms': device.get('forward_device_ms') if device else None,
            'P_remote_tokens': max((r['external_tokens'] or 0 for r in method_calls), default=None),
            'proxy_after_P_s': (times['decode_begin'] - times['prefill_end']) / 1e9,
            'D_first_chunk_to_client_token_s': (first - times['decode_first_chunk']) / 1e9,
            'D_first_enqueue_after_HTTP_ms': (min(r['enqueue_ns'] for r in d_ranks) - times['decode_begin']) / 1e6 if d_ranks else None,
            'D_all_ranks_enqueue_to_end_ms': (max(r['end_ns'] for r in d_ranks) - min(r['enqueue_ns'] for r in d_ranks)) / 1e6 if d_ranks else None,
            'D_last_rank_end_to_first_chunk_ms': (times['decode_first_chunk'] - max(r['end_ns'] for r in d_ranks)) / 1e6 if d_ranks else None,
        })
    rows = summary['requests_joined']
    load_calls = [event for row in rows for event in row['P_GET_rank_calls']]
    d_calls = [event for row in rows for event in row['D_native_rank_calls']]
    batch_loads = [b for b in load_batches if set(b['proxy_ids']) & identifiers]
    assert all(set(b['proxy_ids']) <= identifiers for b in batch_loads)
    for row in rows:
        assert len(row['P_GET_rank_calls']) == 4 and len(row['D_native_rank_calls']) == 4, row['id']
        assert all(call['negative_returns'] == 0 for call in row['P_GET_rank_calls']), row['id']
        assert sum(value['payload_bytes'] for value in row['P_GET_rank_calls']) == row['P_GET']['payload_bytes']
        inclusive = ('client_queue_s', 'client_dispatch_to_proxy_s', 'proxy_arrival_to_P_s', 'P_HTTP_s',
                     'proxy_after_P_s', 'D_first_chunk_s', 'D_first_chunk_to_client_token_s')
        assert abs(sum(row[key] for key in inclusive) - row['TTFT_s']) < 1e-6, row['id']
    columns = ('TTFT_s', 'client_queue_s', 'client_dispatch_to_proxy_s', 'proxy_arrival_to_P_s', 'P_queue_s',
               'P_queue_lookup_ms', 'P_dispatch_to_result_s', 'P_HTTP_outside_engine_s',
               'P_HTTP_begin_to_engine_arrival_s', 'P_ingress_overlaps_same_instance_rank0_GET_s',
               'P_engine_result_to_HTTP_reply_s',
               'P_first_dispatch_to_worker_ms', 'P_first_worker_to_forward_ms', 'P_first_forward_device_ms',
               'P_HTTP_s', 'proxy_after_P_s', 'D_first_chunk_s', 'D_first_chunk_to_client_token_s',
               'D_first_enqueue_after_HTTP_ms', 'D_all_ranks_enqueue_to_end_ms', 'D_last_rank_end_to_first_chunk_ms',
               'D_stream_after_first_s')
    summary['detailed_request_phases'] = stats_fields(rows, columns)
    summary['P_ingress_time_overlap_fraction'] = (
        sum(r['P_ingress_overlaps_same_instance_rank0_GET_s'] for r in rows)
        / sum(r['P_HTTP_begin_to_engine_arrival_s'] for r in rows)
    )
    summary['load_method_stats'] = stats_fields(load_calls, ('native_GET_ms', 'native_GET_Gbps', 'backend_ms',
                                                             'backend_other_ms', 'other_GETs_in_same_load_ms'))
    summary['load_batches'] = batch_loads
    summary['load_batch_stats'] = stats_fields(batch_loads, ('worker_ms', 'native_GET_sum_ms', 'worker_other_ms'))
    summary['D_handler_stats'] = stats_fields(d_calls, ('queue_ms', 'pre_submit_ms', 'submit_to_end_ms'))
    return summary


def repeated_estimates(summaries):
    estimates = {}
    methods = ('P_queue_s', 'P_dispatch_to_result_s', 'P_HTTP_outside_engine_s', 'P_first_forward_device_ms',
               'P_HTTP_begin_to_engine_arrival_s', 'P_engine_result_to_HTTP_reply_s',
               'P_queue_lookup_ms', 'D_first_chunk_s', 'D_stream_after_first_s')
    for cap in ORDERS[0]:
        runs = [summaries[f'b{block}-cap{cap}'] for block in range(1, len(ORDERS) + 1)]
        estimates[cap] = {
            'requests_s_by_block': [r['requests_s'] for r in runs],
            'output_tokens_s_by_block': [r['output_tokens_s'] for r in runs],
            'drain_s_by_block': [r['drain_s'] for r in runs],
            'TTFT_p99_s_by_block': [r['TTFT']['p99_s'] for r in runs],
            'TX_peak_Gbps_by_block': [r['port_samples']['D_tx_Gbps']['max'] for r in runs],
            'GET_wall_fraction_by_block': [r['P_GET_wall_rank_fraction']['mean'] for r in runs],
            'slowest_rank_GET_p50_ms_by_block': [r['P_GET']['slowest_rank_per_restored_request_ms']['p50'] for r in runs],
            'output_tokens_s_mean': mean(r['output_tokens_s'] for r in runs),
            'TTFT_p99_s_median': median(r['TTFT']['p99_s'] for r in runs),
            'request_phase_p50_median': {
                method: median(r['detailed_request_phases'][method]['p50'] for r in runs)
                for method in methods if all(r['detailed_request_phases'][method]['n'] for r in runs)
            },
        }
    throughput_200_vs_100 = [
        estimates[200]['output_tokens_s_by_block'][i] / estimates[100]['output_tokens_s_by_block'][i] - 1
        for i in range(len(ORDERS))
    ]
    breakdown = ('client_queue_s', 'client_dispatch_to_proxy_s', 'proxy_arrival_to_P_s', 'P_queue_s',
                 'P_dispatch_to_result_s', 'P_HTTP_outside_engine_s', 'proxy_after_P_s',
                 'D_first_chunk_s', 'D_first_chunk_to_client_token_s')
    breakdown_deltas = {}
    for method in ('TTFT_s', *breakdown):
        low, high = ([summaries[f'b{block}-cap{cap}']['detailed_request_phases'][method]
                      for block in range(1, len(ORDERS) + 1)] for cap in (100, 200))
        if all(record['n'] == 96 for record in low + high):
            breakdown_deltas[method] = mean(a['mean'] - b['mean'] for a, b in zip(low, high, strict=True))
    return {
        'caps': estimates,
        'output_tokens_s_200_vs_100_fraction_by_block': throughput_200_vs_100,
        'mean_TTFT_100_minus_200_breakdown_s': breakdown_deltas,
    }


def summarize_stack_snapshots(prefill):
    calls = {row['call_id']: row for row in prefill['native']
             if row['type'] == 'store_end' and row['operation'] == 'batch_get_into_multi_buffers'}
    results = []
    for sample in prefill['cpu_stack']:
        trigger = sample['trigger']
        end = calls.get(trigger['call_id'])
        if end is None:
            continue
        snapshots = []
        for snap in sample['snapshots']:
            text = snap.get('stack', '')
            snapshots.append({
                'process': snap['process'], 'pid': snap['pid'], 'exit_code': snap['exit_code'],
                'entire_dump_within_GET': trigger['unix_ns'] <= snap['begin_unix_ns'] <= snap['end_unix_ns'] <= end['unix_ns'],
                'step_with_batch_queue': 'step_with_batch_queue' in text,
                'awaits_worker_response': '_wait_for_response' in text,
                'native_store_wrapper': 'scoped_native_observability.py' in text,
                'pool_start_load_kv': 'start_load_kv' in text,
                'frames': [line.strip() for line in text.splitlines() if line.strip()],
                'error': snap.get('error') or snap.get('stderr') or None,
            })
        results.append({'case': sample['case'], 'call_id': trigger['call_id'],
                        'GET_elapsed_ms': end['elapsed_ns'] / 1e6, 'snapshots': snapshots})
    return results


def comparison(prefill_path, decode_path, *, partial=False):
    p, d = read_bundle(prefill_path), read_bundle(decode_path)
    run_id = p['scope'][0]['run_id']
    assert run_id == d['scope'][0]['run_id']
    workload_hashes = {row['workload_sha256'] for row in d['client_manifest']}
    assert len(workload_hashes) == 1, workload_hashes
    workload_sha256, = workload_hashes
    control = {row['name']: row['data'] for row in d['cap_control']}
    if not partial:
        assert control['progress.json']['stage'] == 'complete' and control['shaping-restore.json']['restored']
    p_events = sorted(p['native'], key=lambda row: row['monotonic_ns'])
    d_events = sorted(d['native'], key=lambda row: row['monotonic_ns'])
    arrivals, batches, steps, lookups, errors = find_methods_and_batches(p_events)
    arrival_monotonic_ns = {pid: row['arrival_ns'] for pid, row in arrivals.items()}
    loads, load_batches, mismatches = associate_loads(p_events)
    transfers = associate_transfers(d_events)
    summaries = {}
    port_comparison_config = control.get('progress.json', {}).get('port_rate_comparison')
    cases, weighted_cases = port_comparison(port_comparison_config['port_Mbps']) if port_comparison_config else (CASES, {})
    if weighted_cases and port_comparison_config['weighted_case'] not in weighted_cases:
        raise ValueError("The comparison did not apply weighted rates in the planned case")
    completed = {row['case'] for row in d['client_summary']}
    profile_cases = {case for case, _, _ in CASES if case.startswith('profile-')}
    if not partial:
        assert (completed & profile_cases) in (set(), profile_cases)
    for case, cap, count in cases:
        if cap is None or (case not in completed and (partial or case in profile_cases)):
            continue
        shaping = control[f'{case}-shaping.json']
        assert len(shaping['readback']) == 8
        assert sum(r['current_Mbps'] for r in shaping['readback'].values()) == cap * 1000
        if weighted_cases:
            expected = list(weighted_cases.get(case, [cap * 125] * 8))
            if (shaping['requested_port_Mbps'] != expected or
                    any(shaping['readback'][str(port)]['configured_Mbps'] != rate
                        or shaping['readback'][str(port)]['current_Mbps'] != rate for port, rate in enumerate(expected))):
                raise ValueError(f"Unexpected per-port shaping readback for {case}")
        summary = analyze_case(p, d, p_events, d_events, arrival_monotonic_ns, batches, case, cap, count)
        assert sum(row['prompt_tokens'] for row in summary['requests_joined']) == count // 2 * sum(PROMPT_LENGTHS)
        summary = detail_case(summary, d, case, arrivals, steps, lookups, loads, load_batches, transfers)
        summaries[case] = summary
    result = {'run_id': run_id, 'workload_sha256': workload_sha256, 'control': control, 'cases': summaries,
              'clock_readbacks': {'P': p['clock'], 'D': d['clock']},
              'P0_CPU_STACK_EVIDENCE': summarize_stack_snapshots(p),
              'P_scheduler_pairing_errors': errors, 'P_load_pairing_issues': mismatches,
              'P_store_errors': p['store_error'], 'D_store_errors': d['store_error'], 'D_master_events': d['master_event']}
    if not partial and weighted_cases:
        balanced_before, weighted, balanced_after = [summaries[case]['output_tokens_s'] for case, cap, _ in cases if cap]
        result['port_rate_comparison'] = {
            'equal_before_tokens_s': balanced_before, 'weighted_tokens_s': weighted, 'equal_after_tokens_s': balanced_after,
            'weighted_vs_equal_mean_pct': (weighted / mean((balanced_before, balanced_after)) - 1) * 100,
        }
    elif not partial:
        result['repeated_estimates'] = repeated_estimates(summaries)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefill', type=Path)
    parser.add_argument('decode', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partial', action='store_true', help='Analyze completed cases while the run is still active.')
    args = parser.parse_args()
    result = comparison(args.prefill, args.decode, partial=args.partial)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
