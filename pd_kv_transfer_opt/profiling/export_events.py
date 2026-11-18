"""Export compact inference timings and counters, leaving prompt bodies on D."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    result = args.run_dir.resolve()
    active = json.loads((result / 'state.json').read_text(encoding='utf-8'))
    role = active['role']
    run = active['run_id']
    logs = result / 'logs'
    export = result / f'natural-repro-{role}.jsonl.gz'
    temporary = export.with_suffix('.tmp')
    counts = Counter()
    sha256 = {}


    def jsonl(path, compressed=False):
        open_stream = gzip.open if compressed else Path.open
        try:
            with open_stream(path, 'rt') as stream:
                for line in stream:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        counts['incomplete_rows'] += 1
        except EOFError:
            # The collector flushes each row while its gzip member is still live.
            counts['open_gzip_streams'] += 1


    def digest(path):
        result = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                result.update(chunk)
        return result.hexdigest()


    def retained_metric(name, instance):
        if name.endswith('_bucket') or name.startswith('#'):
            return False
        if instance == 'master':
            return bool(re.search(r'capacity|used|usage|alloc|evict|object|replica|put|ss[dD]|lease', name))
        return name.startswith(('vllm:request_queue_time_', 'vllm:request_prefill_time_', 'vllm:request_decode_time_',
                                'vllm:request_prefill_kv_computed_tokens_', 'vllm:prompt_tokens_',
                                'vllm:generation_tokens_', 'vllm:external_prefix_cache_', 'vllm:num_requests_',
                                'vllm:kv_cache_usage_', 'vllm:num_preemptions_', 'vllm:request_success_',
                                'vllm:request_time_per_output_token_'))


    def device_samples(snapshot):
        if snapshot['format'] == 'json':
            for device in snapshot['data'].get('devices', []):
                if device.get('aicore_percent') is not None:
                    yield {'device': device['npu'], 'aicore_percent': device['aicore_percent'],
                           'hbm_used_mb': device.get('hbm_used_mb'), 'hbm_total_mb': device.get('hbm_total_mb')}
            return
        current = None
        for line in snapshot['command'].get('stdout', '').splitlines():
            card = re.match(r'^\|\s*(\d+)\s+910B2\s*\|', line)
            if card:
                current = int(card[1])
                continue
            reading = re.match(r'^\|\s*\d+\s*\|\s*[0-9a-fA-F:.]+\s*\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)', line)
            if current is not None and reading:
                yield {'device': current, 'aicore_percent': int(reading[1]),
                       'hbm_used_mb': int(reading[2]), 'hbm_total_mb': int(reading[3])}
                current = None


    with gzip.open(temporary, 'wt', compresslevel=4) as stream:
        def emit(kind, data):
            counts[kind] += 1
            stream.write(json.dumps({'kind': kind, 'data': data}, separators=(',', ':')) + '\n')

        emit('scope', {'run_id': run, 'role': role, 'active': active})
        emit('runtime_state', json.loads((result / 'state.json').read_text()))
        try:
            clock = subprocess.run(['chronyc', '-n', 'tracking'], capture_output=True, text=True, timeout=3)
            emit('clock', {'text': clock.stdout, 'returncode': clock.returncode})
        except (OSError, subprocess.TimeoutExpired) as error:
            emit('clock', {'error': repr(error)})
        for name in [f'{role}0.log', f'{role}1.log']:
            path = logs / name
            with path.open(errors='replace') as source:
                worker_ranks = set()
                for line in source:
                    if 'ISSUE23_OBSERVATION ' in line:
                        try:
                            data = json.loads(line.split('ISSUE23_OBSERVATION ', 1)[1])
                        except ValueError:
                            counts['incomplete_observations'] += 1
                            continue
                        emit('native', dict(data, instance=name.removesuffix('.log')))
                    else:
                        match = re.search(r'Worker_TP(\d+)(?:_EP\d+)? pid=(\d+)', line)
                        if match and match.groups() not in worker_ranks:
                            worker_ranks.add(match.groups())
                            emit('worker_rank', {'instance': name.removesuffix('.log'), 'rank': int(match[1]), 'pid': int(match[2])})
                        if '[mooncake_backend.py:' in line and 'Failed to ' in line:
                            emit('store_error', {'instance': name.removesuffix('.log'), 'line': line.rstrip()})
        master_log = logs / 'master.log'
        if master_log.exists():
            with master_log.open(errors='replace') as source:
                for line in source:
                    if any(x in line for x in ('[EVICT-', 'no_available_handle', 'insufficient space', 'NO_AVAILABLE_HANDLE')):
                        emit('master_event', {'line': line.rstrip()})
        host_path = result / 'host-telemetry/host-samples.jsonl.gz'
        for row in jsonl(host_path, compressed=True) if host_path.is_file() else ():
            if snapshot := row.get('npu'):
                anchor = row['anchor']
                command = snapshot['command']
                when = anchor['unix_ns'] + (command['start_monotonic_ns'] + command['end_monotonic_ns']) // 2 - anchor['monotonic_ns']
                for device in device_samples(snapshot):
                    emit('npu', {'unix_ns': when, **device})
            for instance, value in row.get('metrics', {}).items():
                if value.get('status') != 200:
                    continue
                lines = []
                for line in value['text'].splitlines():
                    name = re.split(r'[ {]', line, maxsplit=1)[0]
                    if retained_metric(name, instance):
                        lines.append(line)
                anchor = row['anchor']
                when = anchor['unix_ns'] + (value['start_monotonic_ns'] + value['end_monotonic_ns']) // 2 - anchor['monotonic_ns']
                emit('metrics', {'instance': instance, 'unix_ns': when, 'lines': lines})
        fabric_path = result / 'fabric-telemetry/host-samples.jsonl.gz'
        for row in jsonl(fabric_path, compressed=True) if fabric_path.is_file() else ():
            for device, value in row.get('hccn_stats', {}).items():
                if not value.get('raw_counters'):
                    continue
                anchor, command = row['anchor'], value['command']
                when = anchor['unix_ns'] + (command['start_monotonic_ns'] + command['end_monotonic_ns']) // 2 - anchor['monotonic_ns']
                emit('fabric', {'device': device, 'unix_ns': when, 'counters': value['raw_counters']})
        if role == 'P':
            for file in sorted((result / 'cpu-stack-diagnostics').glob('*.json')):
                emit('cpu_stack', json.loads(file.read_text()))
        if role == 'D':
            cap_suite = result / 'hard-cap-suite'
            for file in sorted(cap_suite.glob('*.json')):
                emit('cap_control', {'name': file.name, 'data': json.loads(file.read_text())})
            for file in sorted(cap_suite.glob('*-ports.jsonl')):
                for data in jsonl(file):
                    emit('cap_port_sample', {'case': file.name.removesuffix('-ports.jsonl'), **data})
            pressure = result / 'rdma-pressure-receipt.json'
            if pressure.is_file():
                emit('rdma_pressure', json.loads(pressure.read_text()))
            for file in sorted(result.glob('proxy-events*.jsonl')):
                for data in jsonl(file):
                    if data.get('id'):
                        emit('proxy', data)
            for path in sorted(result.glob('natural-sessions*/manifest.json')):
                manifest = json.loads(path.read_text())
                case = manifest.get('case', path.parent.name)
                emit('client_manifest', {'case': case, **manifest})
                client = path.with_name('client.jsonl')
                if client.exists():
                    for data in jsonl(client):
                        # Only request-level data is needed here. The exact chunk stream stays on D.
                        if data['type'] != 'stream_chunk':
                            emit('client', {'case': case, **data})
                summary = path.with_name('analysis.json')
                if summary.exists():
                    emit('client_summary', {'case': case, **json.loads(summary.read_text())})
                    sha256[str(client.relative_to(result))] = digest(client)
    temporary.replace(export)
    print(json.dumps({'path': str(export), 'sha256': digest(export), 'bytes': export.stat().st_size,
                      'role': role, 'rows': counts, 'remote_client_trace_sha256': sha256}, indent=2))


if __name__ == '__main__':
    main()
