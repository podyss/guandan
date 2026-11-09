import argparse
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--tables', type=int, nargs='+', default=[1, 32, 64])
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    status_path = root / 'status.json'

    def status(stage, **details):
        record = {'stage': stage, 'time': datetime.datetime.now().astimezone().isoformat(), **details}
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(status_path)
        print(json.dumps(record), flush=True)

    benchmarks = []
    for tables in args.tables:
        if tables <= 0:
            raise ValueError('Table counts must be positive')
        path = root / f'speed-{tables}' / 'learner' / 'summary.json'
        if path.exists():
            continue
        status('benchmarking', tables=tables)
        benchmark_environment = {**os.environ, 'TABLES': str(tables), 'MAX_SECONDS': '120',
                                 'MAX_UPDATES': '0', 'OMP_NUM_THREADS': '1',
                                 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '0',
                                 'POOL_SIZE': '32768', 'SEED': '20261105'}
        benchmark_environment.pop('INIT_WEIGHTS', None)
        with (root / f'speed-{tables}-runner.log').open('w') as output:
            subprocess.run(['bash', str(repo / 'Danzero_plus/train_local.sh'), str(root / f'speed-{tables}')],
                           cwd=repo, env=benchmark_environment, stdout=output,
                           stderr=subprocess.STDOUT, check=True)
    for tables in args.tables:
        path = root / f'speed-{tables}' / 'learner' / 'summary.json'
        summary = json.loads(path.read_text())
        if summary['updates'] < 1 or summary['optimizer_steps'] < 1:
            raise RuntimeError(f'No valid PPO learning in benchmark {path}')
        benchmarks.append({'tables': tables, **summary})
    best = max(benchmarks, key=lambda item: (item['optimizer_steps'] / item['elapsed_s'],
                                            item['received_steps_per_second']))
    training = root / 'training'
    environment = {**os.environ, 'TABLES': str(best['tables']),
                   'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'CUDA_VISIBLE_DEVICES': '0', 'MAX_UPDATES': str(args.updates),
                   'MAX_SECONDS': str(args.seconds), 'SEED': '20261105',
                   'POOL_SIZE': '2048'}
    environment.pop('INIT_WEIGHTS', None)
    try:
        (repo / 'Danzero_plus/runs/latest-run.txt').write_text(str(training) + '\n')
        status('training', tables=best['tables'], max_seconds=args.seconds,
               max_updates=args.updates, pool_size=2048, benchmarks=benchmarks,
               selection='actual optimizer steps per second')
        with (root / 'training-runner.log').open('w') as output:
            subprocess.run(['bash', str(repo / 'Danzero_plus/train_local.sh'), str(training)],
                           cwd=repo, env=environment, stdout=output,
                           stderr=subprocess.STDOUT, check=True)
        checkpoint = training / 'learner/final.pth'
        summary = json.loads((training / 'learner/summary.json').read_text())
        if summary['optimizer_steps'] < 1:
            raise RuntimeError('Training completed without optimizer steps')
        status('evaluating', tables=best['tables'], training=summary, games=1000)
        evaluation_path = root / 'head-to-head.json'
        command = [sys.executable, '-u', '-m', 'fabledan.evaluate',
                   '--a', 'ckpts/dmc-top2-selector/final.pt',
                   '--b', f'danzero-ppo:{checkpoint}',
                   '--filter-ckpt', 'Danzero_plus/actor_torch/q_network.ckpt',
                   '--candidate-k', '2', '--games', '1000', '--seed', str(args.seed),
                   '--workers', str(args.workers), '--log-every', '100',
                   '--output', str(evaluation_path)]
        with (root / 'evaluation.log').open('w') as output:
            subprocess.run(command, cwd=repo, env=environment, stdout=output,
                           stderr=subprocess.STDOUT, check=True)
        evaluation = json.loads(evaluation_path.read_text())
        result = {'selected_tables': best['tables'], 'benchmarks': benchmarks,
                  'training': summary, 'evaluation': evaluation,
                  'hashes': {str(path): digest(path) for path in (
                      checkpoint, repo / 'ckpts/dmc-top2-selector/final.pt',
                      repo / 'Danzero_plus/actor_torch/q_network.ckpt')},
                  'limitations': [
                      'PPO is a fresh short-budget pilot, not demonstrated convergence.',
                      'FableDan previously trained for 12 hours and 31,018,859 decisions.',
                      'Single-round evaluation approximates original match context.',
                      'PPO trains in the original multi-round executable; FableDan trained in the local single-round engine.',
                      'Opaque original game executable is not seeded.',
                      'Upstream replay sampling and per-match policy refresh are retained.']}
        (root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
        lines = ['# Parallel PPO Pilot Comparison', '',
                 '| Tables | Decisions/s | Updates/min | Optimizer steps |',
                 '| ---: | ---: | ---: | ---: |']
        for benchmark in benchmarks:
            lines.append('| {tables} | {received_steps_per_second:.1f} | '
                         '{updates_per_minute:.2f} | {optimizer_steps} |'.format(**benchmark))
        baseline = min(benchmarks, key=lambda item: item['tables'])['received_steps_per_second']
        lines.extend(['', f"Selected {best['tables']} tables by actual optimizer steps per second (not empty learner updates): "
                      f"{best['received_steps_per_second'] / baseline:.2f}x lowest-table-count collection throughput.",
                      '', 'Speed trials use the previous 32,768-entry replay pool. Training uses the upstream default 2,048-entry pool to reduce policy lag.',
                      '', f"Fresh PPO training: {summary['elapsed_s'] / 60:.1f} minutes, {summary['received_steps']:,} received decisions, "
                      f"{summary['updates']:,} learner updates, {summary['optimizer_steps']:,} optimizer steps.",
                      '', f"FableDan vs PPO: {evaluation['wins']}/1000 wins ({100 * evaluation['win_rate']:.1f}%), "
                      f"mean reward {evaluation['mean_reward']:+.3f}; Wilson 95% interval "
                      f"[{100 * evaluation['win_rate_ci95'][0]:.1f}%, {100 * evaluation['win_rate_ci95'][1]:.1f}%].", '',
                      '## Limitations', '', *('- ' + limitation for limitation in result['limitations'])])
        (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
        exported = repo / 'ckpts/danzero-ppo-fast'
        exported.mkdir(parents=True, exist_ok=True)
        for source in (checkpoint, training / 'learner/training_state.pt',
                       training / 'learner/summary.json', evaluation_path, root / 'REPORT.md'):
            shutil.copy2(source, exported / source.name)
        status('complete', report=str(root / 'REPORT.md'),
               win_rate=evaluation['win_rate'], checkpoint=str(checkpoint),
               exported_checkpoint=str(exported / 'final.pth'))
    except BaseException as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
