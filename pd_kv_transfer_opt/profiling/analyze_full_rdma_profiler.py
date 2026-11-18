"""Align CPU method intervals to the same rank's CANN kernels and RDMA READs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from pd_kv_transfer_opt.profiling.analyze_natural_reproduction import read_bundle
from pd_kv_transfer_opt.profiling.analyze_prefill_chunks import stats


def merge(intervals):
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(high, merged[-1][1])
        elif low < high:
            merged.append((low, high))
    return merged


def milliseconds(intervals):
    return sum(high - low for low, high in merge(intervals)) / 1e6


def overlap_ms(left, right):
    left, right = merge(left), merge(right)
    i = j = duration = 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        duration += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] < b[1]:
            i += 1
        else:
            j += 1
    return duration / 1e6


def interval(row):
    return row['start_ns'], row['start_ns'] + row['duration_ns']


def kind(row):
    name = row['name'].lower().replace('_', '')
    core = row['core'].upper()
    if 'batchget' in name:
        return 'device_batch_get'
    if 'batchput' in name:
        return 'device_batch_put'
    if core == 'COMMUNICATION':
        return 'other_device_communication'
    if 'CORE' in core or core.startswith('MIX_'):
        return 'device_compute'
    return 'other_device'


def associate_captures(bundle, windows):
    matches = {}
    collections = defaultdict(list)
    for row in bundle['kernel']:
        collections[row['capture']].append(interval(row))
    for capture in bundle['capture']:
        intervals = collections[capture['capture']]
        scores = sorted((overlap_ms(intervals, [window]), case) for case, window in windows.items())
        best, case = scores[-1]
        assert best > 0 and scores[-2][0] == 0, (capture, scores)
        assert case not in matches, (capture, case)
        matches[case] = capture
    assert set(matches) == set(windows), matches
    return matches


def analyze_capture(bundle, capture, summary):
    capture_name = capture['capture']
    role = capture['host_role']
    pid = capture['pid']
    start, end = summary['start_unix_ns'], summary['end_unix_ns']
    groups = defaultdict(list)
    names = defaultdict(list)
    for row in bundle['kernel']:
        if row['capture'] != capture_name:
            continue
        low, high = interval(row)
        if low < end and high > start:
            clipped = max(start, low), min(end, high)
            groups[kind(row)].append(clipped)
            names[row['name']].append(clipped)
    methods = defaultdict(list)
    for row in bundle['CPU_scope']:
        if row['capture'] == capture_name:
            low, high = interval(row)
            if low < end and high > start:
                methods[row['name']].append((max(start, low), min(end, high)))
    gets = []
    associated = []
    for request in summary['requests_joined']:
        field = 'P_GET_rank_calls' if role == 'P' else 'D_native_rank_calls'
        for call in request[field]:
            if call['pid'] != pid:
                continue
            begin = call['begin_unix_ns'] if role == 'P' else call['submit_unix_ns']
            bounds = begin, call['end_unix_ns']
            gets.append(bounds)
            associated.append({
                'id': request['id'], 'proxy_request_id': request['proxy_request_id'],
                'GET_start_ns': bounds[0], 'GET_end_ns': bounds[1],
                'get_bytes': call['payload_bytes'],
                'host_ms': milliseconds([bounds]),
                'device_batch_get_overlap_ms': overlap_ms([bounds], groups['device_batch_get']),
                'device_compute_overlap_ms': overlap_ms([bounds], groups['device_compute']),
                'other_device_communication_overlap_ms': overlap_ms([bounds], groups['other_device_communication']),
            })
    assert len(associated) == summary['requests'] // 2, (role, pid, len(associated))
    total = milliseconds(gets)
    api_rows = [row for row in bundle['API_stat'] if row['capture'] == capture_name]
    return {
        'capture': capture,
        'diagnostic_client_window_unix_ns': [start, end],
        'host_native_get_union_ms': total,
        'device_activity_union_ms': milliseconds(x for group in groups.values() for x in group),
        'device_activity_ms_by_kind': {key: milliseconds(group) for key, group in groups.items()},
        'host_GET_overlap_ms': {key: overlap_ms(gets, group) for key, group in groups.items()},
        'device_batch_get_fraction_of_host_GET': overlap_ms(gets, groups['device_batch_get']) / total,
        'compute_fraction_of_host_GET': overlap_ms(gets, groups['device_compute']) / total,
        'host_calls_ms': stats(row['host_ms'] for row in associated),
        'device_batch_get_overlap_per_call_ms': stats(row['device_batch_get_overlap_ms'] for row in associated),
        'host_methods': {name: {'calls': len(spans), 'union_ms': milliseconds(spans),
                                'native_GET_overlap_ms': overlap_ms(gets, spans)}
                         for name, spans in sorted(methods.items())},
        'top_cann_host_apis': sorted(api_rows, key=lambda row: float(row['Time(us)']), reverse=True)[:12],
        'top_device_kernel_names': [{'name': name, 'calls': len(spans), 'union_ms': milliseconds(spans)}
                                    for name, spans in sorted(names.items(), key=lambda pair: milliseconds(pair[1]),
                                                              reverse=True)[:12]],
        'request_calls': associated,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request_comparison', type=Path)
    parser.add_argument('prefill_profiler', type=Path)
    parser.add_argument('decode_profiler', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.request_comparison.read_text(encoding='utf-8'))
    summaries = {case: value for case, value in comparison['cases'].items() if case.startswith('profile-')}
    windows = {case: (summary['start_unix_ns'], summary['end_unix_ns']) for case, summary in summaries.items()}
    reports = {}
    for role, path in (('P', args.prefill_profiler), ('D', args.decode_profiler)):
        bundle = read_bundle(path)
        captures = associate_captures(bundle, windows)
        reports[role] = {case: analyze_capture(bundle, capture, summaries[case]) for case, capture in captures.items()}
    out = {'run_id': comparison['run_id'], 'P': reports['P'], 'D': reports['D']}
    args.output.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
