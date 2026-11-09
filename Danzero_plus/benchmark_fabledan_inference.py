"""Compare local CPU and shared GPU inference using real baseline matches.

Runs alternating CPU/GPU trials with independent game processes and disjoint
ports. Trials score hands rather than complete matches. The game server does
not expose a fixed deal seed, so results include variation between deals.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Danzero_plus.evaluate_paper_baselines import (
    check_inference_server, compile_port_library, inference_worker_args,
    shared_inference, stop_processes,
)


def run_trial(args, tables, device, trial_index, port_library, mode='shared'):
    label = f'tables-{tables}-{device.replace(":", "-")}-{mode}-trial-{trial_index}'
    directory = args.output.parent / label
    directory.mkdir(parents=True, exist_ok=False)
    backend = SimpleNamespace(
        policy='fabledan', checkpoint=args.checkpoint, device=device,
        inference_mode=mode, parallel=tables, single_match=False, episode_worker=False,
        infer_precision='fp32', infer_batch_size=args.batch_size,
        infer_batch_wait_ms=args.batch_wait_ms, inference_endpoint=None,
        output=directory / 'results.json')
    started = time.perf_counter()
    workers = []
    outputs = []
    with shared_inference(backend), ExitStack() as stack:
        try:
            for table in range(tables):
                output = directory / f'table-{table:03d}.json'
                command = [
                    sys.executable, '-u', str(REPO / 'Danzero_plus/evaluate_paper_baselines.py'),
                    '--baseline', str(args.baseline), '--policy', 'fabledan',
                    '--checkpoint', str(args.checkpoint), '--games', '1',
                    '--parallel', '1', '--per-episode', '--episode-worker',
                    '--episodes', str(args.episodes_per_table),
                    '--seat-team', str(table % 2),
                    '--server-offset', str(args.server_offset + table),
                    '--port-library', str(port_library),
                    '--timeout', str(args.timeout), '--stall-timeout', '120',
                    '--output', str(output), *inference_worker_args(backend),
                ]
                log = stack.enter_context((directory / f'table-{table:03d}.log').open('w'))
                workers.append(subprocess.Popen(command, cwd=REPO, stdout=log,
                                                stderr=subprocess.STDOUT,
                                                start_new_session=True))
                outputs.append(output)
            while any(worker.poll() is None for worker in workers):
                check_inference_server(backend)
                if time.perf_counter() - started > args.timeout + 150:
                    raise TimeoutError(f'benchmark trial timed out: {directory}')
                time.sleep(0.05)
            failures = [(worker.pid, worker.returncode) for worker in workers
                        if worker.returncode != 0]
            if failures:
                raise RuntimeError(f'benchmark workers failed: {failures}; inspect {directory}')
        finally:
            stop_processes(workers, process_group=True)
    elapsed = time.perf_counter() - started
    hands = 0
    policy_decisions = 0
    for output in outputs:
        report = json.loads(output.read_text())
        hands += report['episodes_observed']
        log = Path(report['run_directory']) / 'match/server.err'
        policy_decisions += sum(
            int(seat) in report['policy_seats']
            for seat in re.findall(r'client([0-3]) send ', log.read_text()))
    result = {
        'tables': tables, 'device': device, 'mode': mode, 'trial': trial_index,
        'hands': hands, 'elapsed_seconds': elapsed,
        'hands_per_second': hands / elapsed,
        'policy_decisions': policy_decisions,
        'policy_decisions_per_second': policy_decisions / elapsed,
        'inference': backend.inference_metadata, 'artifacts': str(directory),
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-modes', nargs='+', choices=('shared', 'local'), default=['shared'])
    parser.add_argument('--skip-cpu', action='store_true')
    parser.add_argument('--baseline', type=int, default=2, choices=(1, 2, 4, 5, 6))
    parser.add_argument('--tables', nargs='+', type=int, default=[4])
    parser.add_argument('--episodes-per-table', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-wait-ms', type=float, default=2)
    parser.add_argument('--server-offset', type=int, default=400)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.tables, args.episodes_per_table, args.repeats) < 1:
        parser.error('tables, episodes and repeats must be positive')
    if not args.device.startswith('cuda'):
        parser.error('--device must specify a CUDA device')
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    port_library = args.output.parent / 'server_port.so'
    compile_port_library(port_library)
    report = {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(args.checkpoint), 'baseline': args.baseline,
        'note': 'Real hands, random deals, includes model/service startup, FP32. '
                'Other GPU jobs are not stopped. Not a full-match benchmark.',
        'gpu_snapshot': subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,name,memory.free,utilization.gpu',
            '--format=csv'], text=True),
        'trials': [], 'summary': [],
    }
    for tables in args.tables:
        for trial_index in range(args.repeats):
            devices = [] if args.skip_cpu else [('cpu', 'local')]
            devices += [(args.device, mode) for mode in args.gpu_modes]
            if trial_index % 2:
                devices.reverse()
            for device, mode in devices:
                report['trials'].append(run_trial(
                    args, tables, device, trial_index, port_library, mode))
                args.output.write_text(json.dumps(report, indent=2) + '\n')
        medians = {}
        for device, mode in devices:
            samples = [trial['hands_per_second'] for trial in report['trials']
                       if trial['tables'] == tables and trial['device'] == device
                       and trial['mode'] == mode]
            medians[f'{device}/{mode}'] = statistics.median(samples)
        summary = {'tables': tables, 'median_hands_per_second': medians,
                   'relative_to_first': {mode: value / next(iter(medians.values()))
                                         for mode, value in medians.items()}}
        report['summary'].append(summary)
        print(json.dumps(summary), flush=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
