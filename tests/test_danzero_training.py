import os
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[1]


def test_shared_process_supervisor_stops_subprocess_and_worker():
    sys.path.insert(0, str(REPO / 'Danzero_plus/actor_torch'))
    from process_supervisor import stop_process

    child = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(60)'],
        start_new_session=True)
    stop_process(child, process_group=True, timeout=1)
    assert child.poll() is not None

    worker = multiprocessing.Process(target=time.sleep, args=(60,))
    worker.start()
    stop_process(worker, timeout=1)
    assert not worker.is_alive()


def test_baseline6_client_seats_and_isolated_state(tmp_path):
    import runpy
    evaluator = runpy.run_path(str(REPO / 'Danzero_plus/evaluate_paper_baselines.py'))
    prepare_client = evaluator['prepare_baseline_client']
    assert 6 in evaluator['BASELINES']
    first_match = tmp_path / 'first'
    second_match = tmp_path / 'second'
    first_match.mkdir()
    second_match.mkdir()
    for seat in range(4):
        client_path, working_dir = prepare_client(6, seat, first_match)
        assert client_path.name == f'client{seat + 1}.py'
        assert f'/game/client{seat}' in client_path.read_text()
        state_path = working_dir / f'data{seat + 1}.txt'
        state_path.write_text('isolated')
        _, other_working_dir = prepare_client(6, seat, second_match)
        assert other_working_dir != working_dir
        assert state_path.read_text() == 'isolated'
        assert (other_working_dir / state_path.name).read_text() != 'isolated'
    for baseline in (1, 2, 4, 5):
        client_path, working_dir = prepare_client(baseline, 1, first_match)
        assert client_path.exists()
        assert working_dir == client_path.parent


def test_baseline_evaluation_alternates_seat_team():
    source = (REPO / 'Danzero_plus/evaluate_paper_baselines.py').read_text()
    assert 'seat_team = match_index % 2' in source
    assert 'match_index < args.games // 2' not in source


def test_fabledan_native_candidate_k_from_checkpoint(monkeypatch):
    from unittest.mock import Mock
    sys.path.insert(0, str(REPO / 'Danzero_plus/actor_torch'))
    import fabledan_policy

    monkeypatch.setattr(fabledan_policy.DanZeroFilter, 'from_checkpoint',
                        lambda path: Mock())
    for meta, expected in [({}, 2), ({'candidate_k': 2}, 2),
                           ({'candidate_k': 10}, 10)]:
        monkeypatch.setattr(fabledan_policy, 'load_ckpt',
                            lambda path, device: (Mock(), {'meta': meta}))
        policy = fabledan_policy.FableDanPolicy('model.pt', 'dmc.ckpt')
        assert policy.candidate_k == expected


def test_fabledan_move_mapping_cache_is_bounded_and_keyed():
    from unittest.mock import patch
    sys.path.insert(0, str(REPO / 'Danzero_plus/actor_torch'))
    import fabledan_policy

    policy = fabledan_policy.FableDanPolicy.__new__(
        fabledan_policy.FableDanPolicy)
    policy._move_cache_key = None
    policy._move_cache = None
    calls = []

    def fake_gen_moves(hand, level):
        calls.append((tuple(sorted(hand)), level))
        return []

    with patch.object(fabledan_policy, 'gen_moves', fake_gen_moves):
        first = policy._moves_by_signature([9, 2], 3)
        assert policy._moves_by_signature([2, 9], 3) is first
        assert len(calls) == 1
        policy._moves_by_signature([2, 9], 4)
        policy._moves_by_signature([2, 10], 4)
        assert len(calls) == 3
        assert policy._move_cache_key == (4, (2, 10))


def test_fabledan_native_level_mapping():
    sys.path.insert(0, str(REPO / 'Danzero_plus/actor_torch'))
    from fabledan_policy import history_move, level_id
    from fabledan.combos import STRAIGHT

    expected = {'2': 1, '3': 2, '9': 8, 'T': 9, '10': 9,
                'J': 10, 'K': 12, 'A': 0}
    assert {value: level_id(value) for value in expected} == expected
    wild_straight = history_move(
        ['Straight', '6', ['H2', 'H3', 'H4', 'H5', 'H6']], level_id('6'))
    assert wild_straight.type == STRAIGHT
    assert wild_straight.claim_ranks == [0, 1, 2, 3, 4]


def test_actor_initial_weights_and_buffer_reset():
    program = '''
import argparse
import pathlib
import pickle
import tempfile
import numpy as np
import torch
from actor import MLPActorCritic, Player

with tempfile.TemporaryDirectory() as temporary:
    root = pathlib.Path(temporary)
    weights_dir = root / 'weights'
    weights_dir.mkdir()
    expected = MLPActorCritic((2, 624), 2)
    with (weights_dir / '3.pth').open('wb') as stream:
        pickle.dump(expected.state_dict(), stream)
    (weights_dir / '999.tmp').write_bytes(b'incomplete atomic export')
    args = argparse.Namespace(client_index=0, ip='127.0.0.1', data_port=29999,
                              exp_path=str(root), weights_dir=str(weights_dir),
                              dmc_ckpt='Danzero_plus/actor_torch/q_network.ckpt')
    player = Player(args)
    assert player.model_id == 3
    for name, value in expected.state_dict().items():
        assert torch.equal(value, player.model.state_dict()[name])
    state = {'x_batch': np.zeros((2, 567), dtype=np.float32),
             'x_no_action': np.zeros(516, dtype=np.float32)}
    for episode in range(2):
        for decision in range(2):
            assert player.sample(state) in (0, 1)
        player.save_data(1)
        player.prepare_training_data('y1')
        payload = pickle.loads(player.data)
        assert payload['obs'].shape == (2, 624)
        assert payload['obs_cut'].shape == (2, 516)
        assert not player.all_mb_states
        assert not player.all_mb_state_cut
        assert not player.all_mb_legal_indexs
print('actor initial weights and trajectory reset OK')
'''
    environment = {**os.environ, 'PYTHONPATH': str(REPO / 'Danzero_plus/actor_torch'),
                   'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'LOG_FORMAT': 'stdout,csv'}
    subprocess.run([sys.executable, '-c', program], cwd=REPO, env=environment,
                   check=True, timeout=60)


def test_server_port_remapping():
    with tempfile.TemporaryDirectory() as temporary:
        library = Path(temporary) / 'ports.so'
        subprocess.run(['gcc', '-O2', '-Wall', '-Wextra', '-shared', '-fPIC',
                        str(REPO / 'Danzero_plus/actor_torch/server_port.c'),
                        '-o', str(library), '-ldl'], check=True)
        program = '''
import socket
for family, address in ((socket.AF_INET, '127.0.0.1'), (socket.AF_INET6, '::1')):
    with socket.socket(family) as probe:
        probe.bind((address, 0))
        chosen = probe.getsockname()[1]
    import os
    os.environ['DANZERO_SERVER_PORT'] = str(chosen)
    with socket.socket(family) as server:
        server.bind((address, 23456))
        assert server.getsockname()[1] == chosen
    with socket.socket(family) as unchanged:
        unchanged.bind((address, 0))
        assert unchanged.getsockname()[1] != 23456
print('IPv4/IPv6 server port remapping OK')
'''
        environment = {**os.environ, 'LD_PRELOAD': str(library)}
        subprocess.run([sys.executable, '-c', program], env=environment,
                       check=True, timeout=30)


def test_ppo_tensor_update():
    program = '''
import numpy as np
import torch
from model import MLPActorCritic
from ppo import PPOAgent

torch.set_num_threads(1)
torch.manual_seed(123)
model = MLPActorCritic((2, 624), 2)
agent = PPOAgent(model, train_iters=2, target_kl=100)
obs = torch.randn(8, 624)
legal = torch.ones(8, 2)
actions = torch.zeros(8)
with torch.no_grad():
    _, logp, values = model(obs, actions, legal)
before = next(model.parameters()).detach().clone()
data = dict(obs=obs.numpy(), legal=legal.numpy(), act=actions.numpy(),
            logp=logp.numpy(), ret=(values + 1).numpy(),
            adv=np.ones(8, dtype=np.float32))
stats = agent.update(data)
assert stats['optimizer_steps'] == 2
assert abs(stats['initial_kl']) < 1e-6
assert stats['initial_logp_error'] < 1e-6
assert all(np.isfinite(value) for value in stats.values())
assert not torch.equal(before, next(model.parameters()).detach())
print('PPO tensor update and optimizer-step counting OK')
'''
    environment = {**os.environ, 'PYTHONPATH': str(REPO / 'Danzero_plus/learner_torch'),
                   'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'CUDA_VISIBLE_DEVICES': ''}
    subprocess.run([sys.executable, '-c', program], cwd=REPO, env=environment,
                   check=True, timeout=60)


def test_receiver_sample_alignment():
    program = '''
import multiprocessing
import pickle
import socket
import threading
import numpy as np
import zmq
from learner import recv_data
from mem_pool import MemPoolManager

with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
manager = MemPoolManager()
manager.start()
pool = manager.MemPool(capacity=128)
condition = multiprocessing.Condition()
receptions = multiprocessing.Value('i', 0)
steps = multiprocessing.Value('q', 0)
trajectories = multiprocessing.Value('q', 0)
receiver = multiprocessing.Process(target=recv_data,
    args=(port, pool, condition, receptions, False, steps, trajectories))
receiver.start()
errors = []
def send():
    context = zmq.Context()
    connection = context.socket(zmq.REQ)
    connection.setsockopt(zmq.RCVTIMEO, 10000)
    connection.connect(f'tcp://127.0.0.1:{port}')
    try:
        for index in range(120):
            rows = np.arange(index * 64, (index + 1) * 64)
            data = {key: rows.copy() for key in ('obs', 'act', 'legal', 'ret', 'adv', 'logp')}
            connection.send(pickle.dumps(data))
            connection.recv()
    except BaseException as error:
        errors.append(error)
    finally:
        connection.close(linger=0)
        context.term()
sender = threading.Thread(target=send)
sender.start()
checked = 0
try:
    while sender.is_alive() or checked == 0:
        with condition:
            if len(pool) < 64:
                condition.wait(timeout=0.05)
                continue
            batch = pool.sample(size=64)
            for key in batch:
                assert np.array_equal(batch['obs'], batch[key]), key
            checked += 1
    sender.join(timeout=10)
    assert not errors, errors
    assert steps.value == 120 * 64
    assert trajectories.value == 120
finally:
    receiver.terminate()
    receiver.join(timeout=5)
    manager.shutdown()
print('concurrent receive/sample field alignment OK', checked)
'''
    environment = {**os.environ, 'PYTHONPATH': str(REPO / 'Danzero_plus/learner_torch'),
                   'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                   'CUDA_VISIBLE_DEVICES': ''}
    subprocess.run([sys.executable, '-c', program], cwd=REPO, env=environment,
                   check=True, timeout=60)


if __name__ == '__main__':
    test_shared_process_supervisor_stops_subprocess_and_worker()
    test_fabledan_move_mapping_cache_is_bounded_and_keyed()
    test_actor_initial_weights_and_buffer_reset()
    test_server_port_remapping()
    test_ppo_tensor_update()
    test_receiver_sample_alignment()
