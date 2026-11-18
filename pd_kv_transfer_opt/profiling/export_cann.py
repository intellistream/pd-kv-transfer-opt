"""Export rank-0 Ascend profiler intervals, keeping full captures on the hosts."""
import argparse
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import re


TIME_KEYS = ('Start Time(us)', 'Duration(us)')
RANK0 = re.compile(r'_rank0_(\d+)_.*_ascend_pt$')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def time_ns(value):
    return int(Decimal(str(value)) * 1000)


def export_capture(emit, root, capture, rank, role):
    output = capture / 'ASCEND_PROFILER_OUTPUT'
    kernels = output / 'kernel_details.csv'
    if not kernels.is_file():
        from torch_npu.profiler.profiler import analyse

        analyse(str(capture), max_process_number=2)
    trace = output / 'trace_view.json'
    assert kernels.is_file(), capture
    key = capture.name
    emit('capture', {
        'capture': key, 'host_role': role, 'pid': rank, 'path': str(capture.relative_to(root)),
        'kernel_csv_sha256': sha256(kernels),
        'kernel_csv_bytes': kernels.stat().st_size,
        'trace_json_bytes': trace.stat().st_size if trace.is_file() else None,
    })
    n_kernels = 0
    with kernels.open(newline='', encoding='utf-8-sig') as stream:
        rows = csv.DictReader(stream)
        assert set(TIME_KEYS) <= set(rows.fieldnames) and 'Accelerator Core' in rows.fieldnames, rows.fieldnames
        for row in rows:
            if not all(row[name] and row[name] != 'N/A' for name in TIME_KEYS):
                continue
            start, duration = (time_ns(row[name]) for name in TIME_KEYS)
            emit('kernel', {'capture': key, 'start_ns': start, 'duration_ns': duration,
                            'name': row['Name'], 'core': row['Accelerator Core']})
            n_kernels += 1
    n_cpu = 0
    if trace.is_file():
        with trace.open(encoding='utf-8') as stream:
            data = json.load(stream, parse_float=Decimal)
        events = data['traceEvents'] if isinstance(data, dict) else data
        for event in events:
            name = event.get('name', '')
            if event.get('ph') != 'X' or not (name.startswith('issue23::') or name.startswith('aclrtSynchronize')):
                continue
            emit('CPU_scope', {'capture': key, 'name': name, 'category': event.get('cat'),
                               'tid': event.get('tid'), 'start_ns': time_ns(event['ts']),
                               'duration_ns': time_ns(event['dur'])})
            n_cpu += 1
    api = output / 'api_statistic.csv'
    if api.is_file():
        with api.open(newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                if row.get('Time(us)'):
                    emit('API_stat', {'capture': key, **row})
    return {'capture': key, 'pid': rank, 'kernels': n_kernels, 'CPU_scopes': n_cpu}


def export():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--captures', type=int, default=3)
    args = parser.parse_args()
    result = args.run_dir.resolve()
    state = json.loads((result / 'state.json').read_text(encoding='utf-8'))
    role = state['role']
    profile_root = result / 'native-process-profile' / f'{role}0'
    profiles = [(capture, int(match[1])) for capture in sorted(profile_root.glob('*_ascend_pt'))
                if (match := RANK0.search(capture.name))]
    assert len(profiles) == args.captures, [capture.name for capture, _ in profiles]
    export_path = result / f'full-rdma-profiler-{role}.jsonl.gz'
    temporary = export_path.with_suffix('.tmp')
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=4) as stream:
        def emit(kind, data):
            stream.write(json.dumps({'kind': kind, 'data': data}, separators=(',', ':')) + '\n')

        inventory = [export_capture(emit, result, capture, rank, role) for capture, rank in profiles]
    temporary.replace(export_path)
    receipt = {'role': role, 'compressed_export': str(export_path), 'bytes': export_path.stat().st_size,
               'sha256': sha256(export_path), 'raw_profile_root': str(profile_root),
               'all_raw_capture_dirs': len(list(profile_root.glob('*_ascend_pt'))),
               'profile_inventory': inventory}
    (result / f'full-rdma-profiler-{role}-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    export()
