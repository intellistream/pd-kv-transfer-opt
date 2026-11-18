"""Compare fixed cohorts and join physical port counters to the synchronous GET path."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import normalized_proxy_id, read_bundle, summarize_p_workers
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import get_batches, pd_transfers, stats
from pd_kv_transfer_opt.profiling.derive_natural_path_profile import load_before_forward
from pd_kv_transfer_opt.experiments.rdma_hard_cap_protocol import CASES, OUTPUT_TOKENS, RUN_ID, WORKLOAD_SHA256


def case_gets(rows, proxy_ids):
    selected = []
    active = set()
    for row in rows:
        kind = row['type']
        key = (row['pid'], row['thread_id'])
        if kind == 'worker_load_begin':
            request_ids = {normalized_proxy_id(rid) for rid in row['request_ids']}
            if request_ids & proxy_ids.keys():
                assert request_ids <= proxy_ids.keys(), row
                active.add(key)
                selected.append(row)
        elif key in active:
            if kind == 'worker_load_end':
                selected.append(row)
                active.remove(key)
            elif kind in {'store_begin', 'store_end'} and row['operation'] == 'batch_get_into_multi_buffers':
                selected.append(row)
    assert not active, active
    return {'native': selected}


def port_rates(samples):
    segments = []
    counters_delta = defaultdict(int)
    for before, after in zip(samples, samples[1:]):
        rates = []
        for a, b in zip(before['ports'], after['ports'], strict=True):
            assert a['port'] == b['port']
            elapsed = b['monotonic_ns'] - a['monotonic_ns']
            for key in sorted(a['counters'].keys() & b['counters'].keys()):
                delta = b['counters'][key] - a['counters'][key]
                assert delta >= 0, (a, b, key)
                counters_delta[key] += delta
            rates.append(8 * (b['counters']['mac_tx_total_oct_num'] - a['counters']['mac_tx_total_oct_num']) / elapsed)
        segments.append({'start_unix_ns': before['unix_ns'], 'end_unix_ns': after['unix_ns'],
                         'D_tx_Gbps_by_port': rates, 'D_tx_Gbps': sum(rates)})
    return {'D_tx_Gbps': stats(segment['D_tx_Gbps'] for segment in segments),
            'counters_delta': dict(counters_delta), 'segments': segments}


def analyze_case(prefill, decode, p_events, d_events, prefill_arrivals, prefill_batches, case, limit, requests):
    manifest, = [r for r in decode['client_manifest'] if r['case'] == case]
    summary, = [r for r in decode['client_summary'] if r['case'] == case]
    assert manifest['workload_sha256'] and manifest['aggregate_D_egress_cap_Gbps'] == limit
    assert summary['counts'] == {'ok': requests} and not summary['fixed_output_deviations']
    assert not summary['quality']['errors']
    rows = [row for row in decode['client'] if row['case'] == case]
    ends = {r['id']: r for r in rows if r['type'] == 'request_end'}
    origin, = [row for row in rows if row['type'] == 'client_origin']
    start = origin['anchor']['unix_ns']
    finite = summary['finite_cohort']
    elapsed = finite['drain_included_s']
    end = start + round(elapsed * 1e9)
    proxies = [row for row in decode['proxy'] if row['id'] in ends]
    proxy_ids = {row['proxy_request_id']: row['id'] for row in proxies if row['type'] == 'prefill_begin'}
    assert len(proxy_ids) == requests
    selected = case_gets(p_events, proxy_ids)
    get = summarize_p_workers(selected, proxy_ids, start, end)
    request_get = get.pop('per_request')
    batches = []
    first_dispatch = {}
    scheduled = defaultdict(int)
    for batch in prefill_batches:
        ids = [(normalized_proxy_id(rid), tokens) for rid, _, tokens in batch['rows']]
        if any(pid in proxy_ids for pid, _ in ids):
            assert all(pid in proxy_ids for pid, _ in ids), ids
            batches.append(batch)
            for pid, tokens in ids:
                first_dispatch.setdefault(pid, batch['dispatch_ns'])
                scheduled[pid] += tokens
    critical = load_before_forward(selected, batches)
    prefill_devices = [batch['device'] for batch in batches if batch.get('device', {}).get('forward_device_ms') is not None]
    pd = pd_transfers({'proxy': proxies, 'native': [r for r in d_events
                      if normalized_proxy_id(r.get('remote_request_id')) in proxy_ids]})
    pd_per_request = pd.pop('per_request')
    phases = defaultdict(dict)
    for row in proxies:
        phases[row['id']][row['type']] = row['monotonic_ns']

    def span(rid, begin, end):
        return (phases[rid][end] - phases[rid][begin]) / 1e9

    joined = []
    for pid, rid in proxy_ids.items():
        entry = {'id': rid, 'prompt_tokens': ends[rid]['prompt_tokens'],
                 'P_queue_s': (first_dispatch[pid] - prefill_arrivals[pid]) / 1e9,
                 'P_HTTP_s': span(rid, 'prefill_begin', 'prefill_end'),
                 'D_first_chunk_s': span(rid, 'decode_begin', 'decode_first_chunk'),
                 'D_stream_after_first_s': ends[rid]['end_s'] - ends[rid]['first_s'],
                 'scheduled_P_tokens': scheduled[pid], 'P_GET': request_get.get(rid), 'P_to_D': pd_per_request.get(rid)}
        joined.append(entry)
    return {'hardware_cap_Gbps': limit, 'requests': requests, 'output_tokens': requests * OUTPUT_TOKENS,
            'start_unix_ns': start, 'end_unix_ns': end, 'drain_s': elapsed,
            'requests_s': requests / elapsed, 'output_tokens_s': requests * OUTPUT_TOKENS / elapsed,
            'TTFT': finite['ttft_from_ready'],
            'P_GET_payload_Gbps': 8 * get['get_payload_bytes'] / (1e9 * elapsed),
            'P_GET': get, 'P_GET_wall_rank_fraction': stats(get['get_wall_fraction_by_rank'].values()),
            'P_GET_before_forward': critical,
            'P_forward_device_ms': stats(device['forward_device_ms'] for device in prefill_devices),
            'P_worker_until_forward_ms': stats((r['forward_enter_ns'] - r['worker_enter_ns']) / 1e6
                                               for r in prefill_devices),
            'P_vendor_AICore_percent': stats(r['aicore_percent'] for r in prefill['npu'] if start <= r['unix_ns'] <= end),
            'P_scheduled_tokens': sum(scheduled.values()),
            'D_PD_reads': pd, 'phase_s': {k: stats(r[k] for r in joined) for k in
                                         ('P_queue_s', 'P_HTTP_s', 'D_first_chunk_s', 'D_stream_after_first_s')},
            'port_samples': port_rates([sample for sample in decode['cap_port_sample'] if sample['case'] == case]),
            'requests_joined': sorted(joined, key=lambda r: r['id'])}


def analyze(prefill_path, decode_path):
    p, d = read_bundle(prefill_path), read_bundle(decode_path)
    assert p['scope'][0]['run_id'] == d['scope'][0]['run_id'] == RUN_ID
    control = {row['name']: row['data'] for row in d['cap_control']}
    assert control['progress.json']['stage'] == 'complete' and control['shaping-restore.json']['restored']
    p_events = sorted(p['native'], key=lambda r: r['monotonic_ns'])
    d_events = sorted(d['native'], key=lambda r: r['monotonic_ns'])
    arrivals, batches, errors = get_batches(p_events)
    prefill_arrivals = {normalized_proxy_id(row['request_id']): row['arrival_ns'] for row in arrivals.values()}
    summaries = {}
    for case, limit, requests in CASES:
        if limit is None:
            continue
        observed = control[f'{case}-shaping.json']
        assert len(observed['readback']) == 8
        assert sum(row['current_Mbps'] for row in observed['readback'].values()) == limit * 1000
        summaries[case] = analyze_case(p, d, p_events, d_events, prefill_arrivals, batches, case, limit, requests)
    return {'run_id': RUN_ID, 'workload_sha256': WORKLOAD_SHA256, 'cap_control': control,
            'P_scheduler_pairing_errors': errors, 'P_store_errors': p['store_error'],
            'D_store_errors': d['store_error'], 'D_master_events': d['master_event'], 'cases': summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefill', type=Path)
    parser.add_argument('decode', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.prefill, args.decode)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
