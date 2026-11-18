"""Capture brief Python wall-stack snapshots during the separate P0 trace cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import time

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    result = args.run_dir.resolve()
    state = json.loads((result / 'state.json').read_text(encoding='utf-8'))
    if state['role'] != 'P' or state['stage'] != 'ready':
        raise ValueError('Use the ready P-side run directory.')
    logfile = result / 'logs/P0.log'
    out = result / 'cpu-stack-diagnostics'
    out.mkdir(exist_ok=False)
    profile_cases = iter(('profile-cap100', 'profile-cap400', 'profile-cap1600'))
    initial = logfile.read_text(errors='replace')
    engine_pid = int(re.search(r'EngineCore pid=(\d+)', initial)[1])
    worker_pid = int(re.search(r'Worker_TP0(?:_EP0)? pid=(\d+)', initial)[1])

    def snapshot(label, pid):
        assert (Path('/proc') / str(pid)).exists(), (label, pid)
        start = time.time_ns()
        try:
            proc = subprocess.run(['py-spy', 'dump', '--pid', str(pid)], capture_output=True,
                                  text=True, timeout=6)
            return {'process': label, 'pid': pid, 'begin_unix_ns': start, 'end_unix_ns': time.time_ns(),
                    'exit_code': proc.returncode, 'stack': proc.stdout, 'stderr': proc.stderr}
        except subprocess.TimeoutExpired:
            return {'process': label, 'pid': pid, 'begin_unix_ns': start, 'end_unix_ns': time.time_ns(),
                    'exit_code': 124, 'error': 'py-spy dump exceeded 6 s'}

    start_time = time.monotonic()
    next_state_check = start_time
    case = None
    taken = 0
    with logfile.open(errors='replace') as stream:
        stream.seek(0, 2)
        while time.monotonic() - start_time < 1050:
            line = stream.readline()
            if not line:
                if time.monotonic() >= next_state_check:
                    if json.loads((out.parent / 'state.json').read_text())['stage'] == 'finished':
                        return
                    next_state_check = time.monotonic() + 1
                time.sleep(0.06)
                continue
            if 'Profiler started.' in line:
                case = next(profile_cases, None)
                taken = 0
                continue
            if case is None or taken >= 2 or 'ISSUE23_OBSERVATION ' not in line:
                continue
            try:
                event = json.loads(line.split('ISSUE23_OBSERVATION ', 1)[1])
            except ValueError:
                continue
            if (
                event['type'] != 'store_begin'
                or event.get('operation') != 'batch_get_into_multi_buffers'
                or event['pid'] != worker_pid
                or not 0 <= time.time_ns() - event['unix_ns'] < 300_000_000
            ):
                continue
            with ThreadPoolExecutor(max_workers=2) as executor:
                engine = executor.submit(snapshot, 'EngineCore', engine_pid)
                worker = executor.submit(snapshot, 'Worker_TP0', worker_pid)
                record = {'case': case, 'trigger': event, 'snapshots': [engine.result(), worker.result()]}
            taken += 1
            (out / f'{case}-{taken}.json').write_text(json.dumps(record, indent=2) + '\n')
            if case == 'profile-cap1600' and taken >= 2:
                return


if __name__ == '__main__':
    run()
