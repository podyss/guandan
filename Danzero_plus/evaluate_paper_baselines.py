import argparse
from collections import deque
from contextlib import contextmanager
import fcntl
import json
import os
import random
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    from actor_torch.process_supervisor import stop_processes
except ImportError:
    from Danzero_plus.actor_torch.process_supervisor import stop_processes


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
BASELINES = {1, 2, 4, 5, 6}


def interrupt_evaluation(signum, frame):
    raise KeyboardInterrupt


@contextmanager
def reserve_match_ports(preferred_offset=0, needs_actions=False):
    lock_root = Path(tempfile.gettempdir()) / f'danzero-ports-{os.getuid()}'
    lock_root.mkdir(mode=0o700, exist_ok=True)
    offsets = list(range(1, 8001))
    random.SystemRandom().shuffle(offsets)
    offsets.insert(0, preferred_offset)
    for offset in offsets:
        server_port = 23456 + offset
        action_port = 16000 + offset * 4
        ports = [server_port]
        if needs_actions:
            ports += list(range(action_port, action_port + 4))
        if any(port < 1024 or port > 65535 for port in ports):
            continue
        locks = []
        probes = []
        try:
            for port in sorted(set(ports)):
                lock = (lock_root / f'{port}.lock').open('a')
                locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                probes.append(probe)
                probe.bind(('0.0.0.0', port))
                if socket.has_ipv6:
                    probe6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
                    probes.append(probe6)
                    probe6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                    probe6.bind(('::', port))
        except OSError:
            for probe in probes:
                probe.close()
            for lock in locks:
                lock.close()
            continue
        for probe in probes:
            probe.close()
        try:
            yield server_port, action_port
        finally:
            for lock in locks:
                lock.close()
        return
    raise RuntimeError('no free game-server ports available')


def inference_worker_args(args):
    if args.policy != 'fabledan':
        return []
    command = ['--device', args.device, '--infer-precision', args.infer_precision,
               '--infer-batch-size', str(args.infer_batch_size),
               '--inference-mode', getattr(args, 'inference_mode', 'shared')]
    if args.inference_endpoint:
        command += ['--inference-endpoint', args.inference_endpoint]
    return command


def check_inference_server(args):
    process = getattr(args, 'inference_process', None)
    if process is not None and process.poll() is not None:
        raise RuntimeError(f'shared inference server exited: {process.returncode}; '
                           f'inspect {args.output}.inference.log')


def check_local_gpu_memory(device, parallel):
    query = 'import torch; import sys; print(torch.cuda.mem_get_info(sys.argv[1])[0])'
    free = int(subprocess.check_output(
        [sys.executable, '-c', query, device], text=True).strip())
    clients = 2 * parallel
    minimum = (clients + 1) * 256 * 1024 ** 2
    if free < minimum:
        raise RuntimeError(
            f'local GPU inference needs {clients} independent CUDA clients; '
            f'estimated minimum {minimum / 1024 ** 3:.2f} GiB, '
            f'available {free / 1024 ** 3:.2f} GiB on {device}. '
            'Use --inference-mode shared or reduce --parallel. '
            'The estimate is only a lower bound, not a memory guarantee.')


@contextmanager
def shared_inference(args):
    args.inference_metadata = None
    if args.policy != 'fabledan':
        yield
        return
    if not args.inference_endpoint and args.device != 'cpu':
        import torch
        if args.device == 'auto':
            args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device = torch.device(args.device)
        if device.type != 'cuda' and device.type != 'cpu':
            raise ValueError('--device must be auto, cpu, cuda, or cuda:N')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; use --device cpu')
        if (device.type == 'cuda' and getattr(args, 'inference_mode', 'shared') == 'local'
                and not getattr(args, 'single_match', False)
                and not getattr(args, 'episode_worker', False)):
            check_local_gpu_memory(args.device, args.parallel)
            print(f'Local GPU inference: up to {args.parallel * 2} independent CUDA clients; '
                  'each needs its own model and CUDA memory. Start with --parallel 1 or 2.',
                  file=sys.stderr, flush=True)
    args.inference_metadata = {
        'device': args.device,
        'backend': ('shared' if args.inference_endpoint or (
            args.device != 'cpu' and getattr(args, 'inference_mode', 'shared') == 'shared')
                    else 'local'),
        'precision': args.infer_precision, 'batch_size': args.infer_batch_size,
    }
    if (args.inference_endpoint or args.device == 'cpu' or
            getattr(args, 'inference_mode', 'shared') == 'local'):
        if args.device == 'cpu' and args.infer_precision != 'fp32':
            raise ValueError('CPU evaluation requires --infer-precision fp32')
        yield
        return
    with tempfile.TemporaryDirectory(prefix='fabledan-infer-') as directory:
        root = Path(directory)
        args.inference_endpoint = f'ipc://{root / "inference.sock"}'
        ready_file = root / 'ready.json'
        command = [
            sys.executable, '-u', '-m', 'fabledan.eval_inference',
            '--checkpoint', str(args.checkpoint),
            '--dmc-checkpoint', str(ROOT / 'actor_torch/q_network.ckpt'),
            '--device', args.device, '--precision', args.infer_precision,
            '--batch-size', str(args.infer_batch_size),
            '--batch-wait-ms', str(args.infer_batch_wait_ms),
            '--endpoint', args.inference_endpoint, '--ready-file', str(ready_file),
        ]
        previous_sigterm = signal.signal(
            signal.SIGTERM, interrupt_evaluation)
        process = None
        try:
            with Path(f'{args.output}.inference.log').open('w') as log:
                process = subprocess.Popen(
                    command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                    start_new_session=True)
                args.inference_process = process
                deadline = time.monotonic() + 120
                while not ready_file.exists():
                    check_inference_server(args)
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'inference startup timed out; inspect {args.output}.inference.log')
                    time.sleep(0.05)
                args.inference_metadata.update(json.loads(ready_file.read_text()))
                print(f'Shared FableDan inference: {args.inference_metadata}; '
                      f'log: {args.output}.inference.log', flush=True)
                yield
        finally:
            if process is not None:
                stop_processes([process], process_group=True)
            args.inference_process = None
            args.inference_endpoint = None
            signal.signal(signal.SIGTERM, previous_sigterm)


def prepare_baseline_client(baseline, seat, phase_dir, record_stats=False):
    client_number = seat + 1 if baseline in (4, 5, 6) else seat
    client_path = ROOT / 'wintest' / f'ai{baseline}' / f'client{client_number}.py'
    working_dir = client_path.parent
    if baseline == 6:
        working_dir = phase_dir / 'baseline6'
        working_dir.mkdir(exist_ok=True)
        data_name = f'data{client_number}.txt'
        (working_dir / data_name).write_bytes(
            (client_path.parent / data_name).read_bytes())
    if record_stats:
        source_dir = ROOT / 'wintest' / f'ai{baseline}'
        working_dir = phase_dir / f'baseline{baseline}-seat{seat}'
        working_dir.mkdir(exist_ok=True)
        for dependency in source_dir.glob('*.py'):
            shutil.copy2(dependency, working_dir / dependency.name)
        client_path = working_dir / client_path.name
        source = (source_dir / f'client{client_number}.py').read_text()
        marker = 'message = json.loads(str(message))'
        stats_code = (
            "\n        record_stage = ('episodeOver' if os.environ.get("
            "'DANZERO_EVAL_RECORD_EPISODES') else 'gameResult')\n"
            "        if message.get('stage') == record_stage:\n"
            "            stats_path = os.environ.get('DANZERO_EVAL_STATS')\n"
            "            if stats_path:\n"
            "                with open(stats_path, 'a') as stats_file:\n"
            "                    stats_file.write(json.dumps(message) + '\\n')\n"
        )
        source = source.replace('import json\n', 'import json\nimport os\n', 1)
        source = source.replace(marker, marker + stats_code, 1)
        client_path.write_text(source)
        if baseline == 6:
            data_name = f'data{client_number}.txt'
            (working_dir / data_name).write_bytes(
                (ROOT / 'wintest' / f'ai{baseline}' / data_name).read_bytes())
    return client_path, working_dir


def wait_for_connection(server_log, endpoint, process, timeout=30):
    marker = f'GET /game/client{endpoint}'
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker in server_log.read_text(errors='ignore'):
            return
        if process.poll() is not None:
            raise RuntimeError(
                f'client process exited before connecting: endpoint={endpoint}, '
                f'returncode={process.returncode}')
        time.sleep(0.05)
    raise TimeoutError(f'client endpoint {endpoint} did not connect within {timeout}s')


def wait_for_server_ready(server_log, process, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        contents = server_log.read_text(errors='ignore')
        if 'Address already in use' in contents:
            raise RuntimeError(f'game server port is occupied; inspect {server_log}')
        if process.poll() is not None:
            raise RuntimeError(
                f'game server exited during startup; returncode={process.returncode}; '
                f'inspect {server_log}')
        if 'Ready for connect.' in contents:
            return
        time.sleep(0.05)
    raise TimeoutError(f'game server did not become ready within {timeout}s')


def compile_port_library(output_path):
    subprocess.run([
        'gcc', '-O2', '-shared', '-fPIC',
        str(ROOT / 'actor_torch/server_port.c'), '-o', str(output_path), '-ldl',
    ], check=True)


def episode_policy_win(order, policy_seats):
    pattern = ''.join('1' if seat in policy_seats else '0' for seat in order)
    return pattern in {'1100', '1010', '1001'}


def run_match(args):
    with reserve_match_ports(args.server_offset, args.policy == 'ppo') as ports:
        return run_match_on_ports(args, *ports)


def run_match_on_ports(args, server_port, action_port):
    root = Path(tempfile.mkdtemp(prefix=f'danzero-match-{args.baseline}-', dir=args.output.parent))
    weights = root / 'weights'
    weights.mkdir()
    if args.policy == 'ppo':
        (weights / '1.pth').write_bytes(args.checkpoint.read_bytes())
    phase_dir = root / 'match'
    phase_dir.mkdir()
    (phase_dir / 'actor').mkdir()
    stats_file = root / ('episode-results.jsonl' if args.per_episode
                         else 'game-results.jsonl')
    progress_file = root / 'episode-progress.jsonl'
    policy_seats = [0, 2] if args.seat_team == 0 else [1, 3]
    env_base = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'CUDA_VISIBLE_DEVICES': '-1', 'PYTHONPATH': str(ROOT / 'actor_torch'),
                'LOG_FORMAT': 'stdout,csv'}
    port_library = getattr(args, 'port_library', None)
    if port_library is None:
        port_library = phase_dir / 'server_port.so'
        compile_port_library(port_library)
    server_env = {**env_base, 'LD_PRELOAD': str(port_library),
                  'DANZERO_SERVER_PORT': str(server_port),
                  'TMPDIR': str(phase_dir), 'DANZERO_EVAL_STATS': str(stats_file),
                  'DANZERO_EVAL_PROGRESS': str(progress_file)}
    if args.per_episode:
        server_env['DANZERO_EVAL_RECORD_EPISODES'] = '1'
    processes = []
    process_roles = []
    previous_sigterm = signal.signal(
        signal.SIGTERM, interrupt_evaluation)
    try:
        server = subprocess.Popen([
            str(ROOT / 'actor_torch/danserver'),
            str(args.episodes if args.per_episode else 1),
        ], env=server_env,
            stdout=(phase_dir / 'server.log').open('w'),
            stderr=(phase_dir / 'server.err').open('w'),
            start_new_session=True)
        processes.append(server)
        process_roles.append('game server')
        wait_for_server_ready(phase_dir / 'server.err', server)

        if args.policy == 'ppo':
            actor = subprocess.Popen([
                'uv', '--project', str(REPO), 'run', 'python', '-u',
                str(ROOT / 'actor_torch/actor.py'), '--weights-dir', str(weights),
                '--dmc-ckpt', str(ROOT / 'actor_torch/q_network.ckpt'),
                '--action_port', str(action_port), '--seats', ','.join(map(str, policy_seats)),
                '--evaluation', '--exp_path', str(phase_dir / 'actor'),
            ], env=env_base, stdout=(phase_dir / 'actor.log').open('w'),
               stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(actor)
            process_roles.append('PPO actor')
        connection_order = []
        for seat in range(4):
            is_policy = seat in policy_seats
            if is_policy and args.policy in ('ppo', 'fabledan'):
                client_command = [
                    sys.executable, '-u',
                    str(ROOT / 'actor_torch/game.py'), '--server_port', str(server_port),
                    '--action_port', str(action_port), '--seats', str(seat),
                ]
                if args.policy == 'fabledan':
                    client_command += ['--fabledan-checkpoint', str(args.checkpoint),
                                       '--fabledan-device', args.device,
                                       '--fabledan-precision', args.infer_precision]
                    if args.inference_endpoint:
                        client_command += ['--fabledan-inference-endpoint',
                                           args.inference_endpoint]
                client_env = {**server_env,
                              'DANZERO_EVAL_RECORD_SEAT': str(min(policy_seats))}
                if (args.policy == 'fabledan' and not args.inference_endpoint
                        and args.device.startswith('cuda')):
                    client_env.pop('CUDA_VISIBLE_DEVICES')
                    if 'CUDA_VISIBLE_DEVICES' in os.environ:
                        client_env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
                process = subprocess.Popen(
                    client_command, env=client_env,
                    stdout=(phase_dir / f'{args.policy}-client-{seat}.log').open('w'),
                    stderr=subprocess.STDOUT, start_new_session=True)
                role = f'{args.policy} websocket client seat {seat}'
            else:
                baseline = args.policy_baseline if is_policy else args.baseline
                client_path, client_cwd = prepare_baseline_client(
                    baseline, seat, phase_dir,
                    record_stats=(is_policy and seat == min(policy_seats)))
                client_env = {**env_base, 'PYTHONPATH': str(client_path.parent),
                              'LD_PRELOAD': str(port_library),
                              'DANZERO_SERVER_PORT': str(server_port),
                              'DANZERO_EVAL_STATS': str(stats_file),
                              'DANZERO_EVAL_PROGRESS': str(progress_file),
                              **({'DANZERO_EVAL_RECORD_EPISODES': '1'}
                                 if args.per_episode else {})}
                process = subprocess.Popen([
                    sys.executable, '-u', str(client_path)
                ], cwd=client_cwd, env=client_env,
                   stdout=(phase_dir / f'{"policy-" if is_policy else ""}baseline-seat-{seat}.log').open('w'),
                   stderr=subprocess.STDOUT, start_new_session=True)
                role = f'{"policy " if is_policy else ""}baseline seat {seat} client'
            processes.append(process)
            process_roles.append(role)
            wait_for_connection(phase_dir / 'server.err', seat, process,
                                timeout=120 if args.policy == 'fabledan' else 30)
            connection_order.append(seat)

        deadline = time.monotonic() + args.timeout
        last_progress_size = 0
        last_progress_time = time.monotonic()
        while time.monotonic() < deadline:
            if stats_file.exists() and stats_file.stat().st_size:
                messages = [json.loads(line) for line in stats_file.read_text().splitlines() if line.strip()]
                if args.per_episode and len(messages) >= args.episodes:
                    episode_wins = [episode_policy_win(message['order'], policy_seats)
                                    for message in messages[:args.episodes]]
                    policy_wins = sum(episode_wins)
                    report = {
                        'baseline': args.baseline, 'policy': args.policy,
                        'inference': args.inference_metadata,
                        'policy_baseline': getattr(args, 'policy_baseline', None),
                        'checkpoint': str(args.checkpoint),
                        'episodes_requested': args.episodes,
                        'episodes_observed': len(episode_wins),
                        'policy_seats': policy_seats,
                        'policy_episode_wins': policy_wins,
                        'opponent_episode_wins': len(episode_wins) - policy_wins,
                        'policy_win_rate': policy_wins / len(episode_wins),
                        'episode_results': messages[:args.episodes],
                        'run_directory': str(root),
                        'connection_order': connection_order,
                    }
                    args.output.write_text(json.dumps(report, indent=2) + '\n')
                    return report
                if messages and not args.per_episode:
                    result = messages[-1]
                    seat_wins = sum(result['victoryNum'][seat] for seat in policy_seats)
                    if seat_wins not in (0, 2):
                        raise RuntimeError(f'unexpected complete-match win count {seat_wins}: {result}')
                    report = {
                        'baseline': args.baseline, 'policy': args.policy,
                        'inference': args.inference_metadata,
                        'policy_baseline': getattr(args, 'policy_baseline', None),
                        'checkpoint': str(args.checkpoint),
                        'games_requested': 1, 'games_observed': 1,
                        'policy_seats': policy_seats, 'policy_team_wins': seat_wins // 2,
                        'opponent_team_wins': 1 - seat_wins // 2,
                        'ppo_seats': policy_seats, 'ppo_team_wins': seat_wins // 2,
                        'baseline_team_wins': 1 - seat_wins // 2,
                        'game_result': result, 'run_directory': str(root),
                        'connection_order': connection_order,
                    }
                    args.output.write_text(json.dumps(report, indent=2) + '\n')
                    return report
            if server.poll() is not None:
                stage = 'episodeOver' if args.per_episode else 'gameResult'
                raise RuntimeError(f'game server stopped before {stage}; inspect {phase_dir}')
            exited = [(role, process.pid, process.poll())
                      for role, process in zip(process_roles[1:], processes[1:])
                      if process.poll() is not None]
            if exited:
                raise RuntimeError(f'evaluation components exited before gameResult: {exited}; inspect {phase_dir}')
            check_inference_server(args)
            progress_size = progress_file.stat().st_size if progress_file.exists() else 0
            if progress_size > last_progress_size:
                last_progress_size = progress_size
                last_progress_time = time.monotonic()
            elif time.monotonic() - last_progress_time > args.stall_timeout:
                raise TimeoutError(f'no completed hand for {args.stall_timeout}s; inspect {phase_dir}')
            time.sleep(0.25)
        unit = 'episodes' if args.per_episode else 'full match'
        raise TimeoutError(f'{unit} timed out after {args.timeout}s; inspect {phase_dir}')
    except KeyboardInterrupt:
        raise
    except Exception:
        print(f'Failed match artifacts: {root}', file=sys.stderr)
        raise
    finally:
        stop_processes(processes, process_group=True)
        signal.signal(signal.SIGTERM, previous_sigterm)


def run_parallel(args):
    shard_root = Path(tempfile.mkdtemp(prefix=f'danzero-eval-{args.baseline}-', dir=args.output.parent))
    port_library = shard_root / 'server_port.so'
    compile_port_library(port_library)
    started_at = time.monotonic()
    output_by_match = {
        match_index: shard_root / f'match-{match_index:04d}.json'
        for match_index in range(args.games)
    }
    pending = deque(range(args.games))
    attempts = {match_index: 0 for match_index in range(args.games)}
    children = {}
    reports_by_match = {}
    last_display = -1

    def start_worker(match_index):
        match_output = output_by_match[match_index]
        worker_command = [
            sys.executable, str(Path(__file__).resolve()),
            '--baseline', str(args.baseline), '--policy', args.policy, '--games', '1',
            *(('--policy-baseline', str(args.policy_baseline))
              if args.policy == 'baseline' else ()),
            '--output', str(match_output),
            '--parallel', '1', '--timeout', str(args.timeout),
            '--stall-timeout', str(args.stall_timeout),
            '--server-offset', str(match_index + 1), '--single-match',
            '--seat-team', str(match_index % 2),
            '--port-library', str(port_library),
        ]
        if args.checkpoint is not None:
            worker_command += ['--checkpoint', str(args.checkpoint)]
        worker_command += inference_worker_args(args)
        attempts[match_index] += 1
        log_path = shard_root / f'match-{match_index:04d}-attempt-{attempts[match_index]}.log'
        with log_path.open('w') as log:
            return subprocess.Popen(worker_command, cwd=REPO, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)

    try:
        while pending or children:
            check_inference_server(args)
            while pending and len(children) < args.parallel:
                match_index = pending.popleft()
                children[start_worker(match_index)] = match_index

            for child, match_index in list(children.items()):
                returncode = child.poll()
                if returncode is None:
                    continue
                child.wait()
                del children[child]
                match_output = output_by_match[match_index]
                if returncode == 0 and match_output.is_file():
                    reports_by_match[match_index] = json.loads(match_output.read_text())
                    continue
                if attempts[match_index] < 2:
                    print(f'\nRetrying failed match {match_index + 1}/{args.games} '
                          f'(exit code {returncode}).', flush=True)
                    pending.appendleft(match_index)
                    continue
                raise RuntimeError(
                    f'match worker failed twice: match {match_index}; '
                    f'exit code {returncode}; inspect '
                    f'{shard_root / f"match-{match_index:04d}-attempt-{attempts[match_index]}.log"}')

            completed = len(reports_by_match)
            if completed != last_display:
                now = time.monotonic()
                elapsed = max(now - started_at, 0.001)
                rate = completed / elapsed
                eta = (args.games - completed) / rate if rate else 0
                width = 30
                filled = min(width, width * completed // args.games)
                print(f"\r[{'=' * filled}{' ' * (width - filled)}] "
                      f'{completed}/{args.games} complete matches | {rate:.2f}/s | ETA {eta:.0f}s',
                      end='', flush=True)
                last_display = completed
            if children:
                time.sleep(0.05)
    finally:
        stop_processes(children, process_group=True)

    reports = [reports_by_match[match_index] for match_index in range(args.games)]

    print()
    wins = sum(report['policy_team_wins'] for report in reports)
    elapsed = time.monotonic() - started_at
    combined = {
        'baseline': args.baseline, 'policy': args.policy,
        'inference': args.inference_metadata,
        'policy_baseline': getattr(args, 'policy_baseline', None),
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'games_requested': args.games, 'games_observed': len(reports),
        'policy_team_wins': wins, 'opponent_team_wins': args.games - wins,
        'policy_win_rate': wins / args.games,
        'ppo_team_wins': wins, 'baseline_team_wins': args.games - wins,
        'ppo_win_rate': wins / args.games,
        'seat_balance': {'policy_seats_0_2': args.games // 2,
                         'policy_seats_1_3': args.games - args.games // 2},
        'parallel': args.parallel, 'elapsed_seconds': round(elapsed, 2),
        'matches_per_second': round(args.games / max(elapsed, 0.001), 4),
        'match_reports': reports,
        'note': 'Each worker runs one complete match; win is read from that match gameResult.'
    }
    args.output.write_text(json.dumps(combined, indent=2) + '\n')
    print(json.dumps({key: value for key, value in combined.items() if key != 'match_reports'}, indent=2))


def run_parallel_episodes(args):
    shard_root = Path(tempfile.mkdtemp(
        prefix=f'danzero-episodes-{args.baseline}-', dir=args.output.parent))
    port_library = shard_root / 'server_port.so'
    compile_port_library(port_library)
    base, remainder = divmod(args.episodes, args.parallel)
    shards = []
    next_episode = 0
    for shard_index in range(args.parallel):
        shard_episodes = base + (1 if shard_index < remainder else 0)
        if not shard_episodes:
            continue
        shards.append((shard_index, shard_episodes, next_episode))
        next_episode += shard_episodes

    children = {}
    outputs = {}
    started_at = time.monotonic()
    try:
        for shard_index, shard_episodes, first_episode in shards:
            output = shard_root / f'shard-{shard_index:03d}.json'
            command = [
                sys.executable, str(Path(__file__).resolve()),
                '--baseline', str(args.baseline), '--policy', args.policy,
                '--games', '1', '--parallel', '1', '--per-episode',
                '--episodes', str(shard_episodes), '--seat-team', str(shard_index % 2),
                '--timeout', str(args.timeout), '--stall-timeout', str(args.stall_timeout),
                '--server-offset', str(shard_index + 1), '--episode-worker',
                '--port-library', str(port_library),
                '--output', str(output),
            ]
            if args.policy_baseline is not None:
                command += ['--policy-baseline', str(args.policy_baseline)]
            if args.checkpoint is not None:
                command += ['--checkpoint', str(args.checkpoint)]
            command += inference_worker_args(args)
            log = (shard_root / f'shard-{shard_index:03d}.log').open('w')
            children[shard_index] = subprocess.Popen(
                command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            outputs[shard_index] = output

        last_display = -1
        while any(child.poll() is None for child in children.values()):
            check_inference_server(args)
            finished = sum(
                len(json.loads(path.read_text()).get('episode_results', []))
                for path in outputs.values() if path.exists())
            if finished != last_display:
                elapsed = max(time.monotonic() - started_at, 0.001)
                rate = finished / elapsed
                eta = (args.episodes - finished) / rate if rate else 0
                width = 30
                filled = min(width, width * finished // args.episodes)
                print(f"\r[{'=' * filled}{' ' * (width - filled)}] "
                      f'{finished}/{args.episodes} episodes | {rate:.2f}/s | ETA {eta:.0f}s',
                      end='', flush=True)
                last_display = finished
            time.sleep(0.5)

        failures = [(index, child.returncode) for index, child in children.items()
                    if child.returncode != 0 or not outputs[index].exists()]
        if failures:
            raise RuntimeError(
                f'episode workers failed: {failures}; logs: {shard_root}')
    finally:
        stop_processes(children.values(), process_group=True)

    reports = [json.loads(outputs[index].read_text()) for index, _, _ in shards]
    episode_results = [
        result for report in reports for result in report['episode_results']]
    policy_wins = sum(report['policy_episode_wins'] for report in reports)
    seat_0_2 = sum(report['episodes_observed'] for report in reports
                   if report['policy_seats'] == [0, 2])
    seat_1_3 = sum(report['episodes_observed'] for report in reports
                   if report['policy_seats'] == [1, 3])
    elapsed = time.monotonic() - started_at
    combined = {
        'baseline': args.baseline, 'policy': args.policy,
        'inference': args.inference_metadata,
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'episodes_requested': args.episodes,
        'episodes_observed': len(episode_results),
        'policy_episode_wins': policy_wins,
        'opponent_episode_wins': len(episode_results) - policy_wins,
        'policy_win_rate': policy_wins / max(1, len(episode_results)),
        'seat_balance': {'policy_seats_0_2': seat_0_2,
                         'policy_seats_1_3': seat_1_3},
        'parallel': len(shards), 'elapsed_seconds': round(elapsed, 2),
        'episodes_per_second': round(len(episode_results) / max(elapsed, 0.001), 4),
        'worker_reports': reports, 'episode_results': episode_results,
        'note': 'Each worker runs an independent Danzero+ server session; wins are counted per episodeOver order.'
    }
    args.output.write_text(json.dumps(combined, indent=2) + '\n')
    print('\n' + json.dumps({key: value for key, value in combined.items()
                            if key not in ('worker_reports', 'episode_results')}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=int, choices=sorted(BASELINES), required=True)
    parser.add_argument('--policy', choices=('ppo', 'fabledan', 'baseline'), default='ppo')
    parser.add_argument('--policy-baseline', type=int, choices=sorted(BASELINES),
                        help='baseline used by the policy team when --policy baseline')
    parser.add_argument('--games', type=int, default=1000, help='number of complete GuanDan matches')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='auto',
                        help='FableDan inference: auto (CUDA when available), cpu, or cuda:N')
    parser.add_argument('--inference-mode', choices=('shared', 'local'), default='shared',
                        help='shared GPU service or independent models in each client')
    parser.add_argument('--infer-batch-size', type=int, default=32,
                        help='maximum requests per shared inference batch')
    parser.add_argument('--infer-batch-wait-ms', type=float, default=2,
                        help='maximum time to collect an inference batch')
    parser.add_argument('--infer-precision', choices=('fp32', 'bf16'), default='fp32',
                        help='selector precision; DMC filtering always uses FP32')
    parser.add_argument('--inference-endpoint', help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--parallel', type=int, help='number of independent complete-match workers')
    parser.add_argument('--timeout', type=int, default=7200, help='per-match timeout in seconds')
    parser.add_argument('--stall-timeout', type=int, default=180,
                        help='seconds without a completed hand before retrying a match')
    parser.add_argument('--per-episode', action='store_true',
                        help='run one server session and score individual episodeOver rounds')
    parser.add_argument('--episodes', type=int, default=100,
                        help='number of episodeOver rounds for --per-episode')
    parser.add_argument('--episode-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--single-match', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--seat-team', type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument('--server-offset', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--port-library', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.infer_batch_size < 1 or args.infer_batch_wait_ms < 0:
        parser.error('inference batch size must be positive and batch wait must be nonnegative')
    if args.policy != 'fabledan' and args.device != 'auto':
        parser.error('--device is only supported for --policy fabledan')
    if args.games < 1:
        parser.error('--games must be positive')
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    if args.policy == 'baseline' and args.policy_baseline is None:
        parser.error('--policy-baseline is required when --policy baseline')
    if args.policy != 'baseline' and args.policy_baseline is not None:
        parser.error('--policy-baseline is only valid with --policy baseline')
    if args.parallel is None:
        args.parallel = min(50, args.games)
    if args.parallel < 1 or args.parallel > args.games:
        parser.error('--parallel must be between 1 and games')
    if args.checkpoint is None and args.policy in ('ppo', 'fabledan'):
        args.checkpoint = (REPO / 'ckpts/danzero-ppo-fast/final.pth'
                           if args.policy == 'ppo' else
                           REPO / 'ckpts/dmc-top2-selector/final.pt')
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f'checkpoint not found: {args.checkpoint}')
    if args.stall_timeout < 1:
        parser.error('--stall-timeout must be positive')
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    if args.port_library is not None and not args.port_library.is_file():
        parser.error(f'port library not found: {args.port_library}')
    if args.single_match and (args.games != 1 or args.seat_team is None):
        parser.error('single-match workers require --games 1 and --seat-team')
    if args.per_episode and args.single_match:
        parser.error('--per-episode cannot be used by complete-match workers')
    if args.per_episode:
        args.output = args.output or REPO / (
            f'ckpts/danzero-ppo-fast/{args.policy}-vs-baseline-'
            f'{args.baseline}-{args.episodes}-episodes.json')
    else:
        default_name = f'{args.policy}-vs-baseline-{args.baseline}.json'
        args.output = args.output or REPO / f'ckpts/danzero-ppo-fast/{default_name}'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with shared_inference(args):
        run_evaluation(args)


def run_evaluation(args):
    if args.per_episode:
        if not args.episode_worker and args.parallel > 1:
            run_parallel_episodes(args)
            return
        if args.seat_team is None:
            args.seat_team = 0
        run_match(args)
        return
    if args.single_match:
        run_match(args)
    else:
        run_parallel(args)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupt_evaluation)
    try:
        main()
    except KeyboardInterrupt:
        print('Evaluation interrupted; child processes stopped.', file=sys.stderr)
        sys.exit(130)
