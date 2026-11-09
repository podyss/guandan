import multiprocessing
import os
import io
import pickle
import json
import random
import time
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from multiprocessing import Process
from pathlib import Path

import torch
import numpy as np
import zmq
from ppo import PPOAgent
from common import create_experiment_dir, load_yaml_config, save_yaml_config
from mem_pool import MemPoolManager, MultiprocessingMemPool
from model import MLPActorCritic, MLPQNetwork
# from torch.utils.tensorboard import SummaryWriter
from utils import logger
from utils.cmdline import parse_cmdline_kwargs

warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument('--env', type=str, default='GuanDan', help='The game environment')
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to receive training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server to publish model parameters')
# parser.add_argument('--pool_size', type=int, default=256, help='The max length of data pool')
# parser.add_argument('--batch_size', type=int, default=256, help='The batch size for training')
parser.add_argument('--pool_size', type=int, default=2048, help='The max length of data pool')
parser.add_argument('--batch_size', type=int, default=2048, help='The batch size for training')
parser.add_argument('--training_freq', type=int, default=13,
                    help='How many receptions of new data are between each training, '
                         'which can be fractional to represent more than one training per reception')
parser.add_argument('--keep_training', type=bool, default=False,
                    help="No matter whether new data is received recently, keep training as long as the data is enough "
                         "and ignore `--training_freq`")
parser.add_argument('--config', type=str, default=None, help='Directory to config file')
parser.add_argument('--exp_path', type=str, default=None, help='Directory to save logging data and config file')
parser.add_argument('--record_throughput_interval', type=int, default=20,
                    help='The time interval between each throughput record')
parser.add_argument('--ckpt_save_freq', type=int, default=500, help='The number of updates between each weights saving')
parser.add_argument('--ckpt_save_type', type=str, default='weight', help='Type of checkpoint file will be recorded : weight(smaller) or checkpoint(bigger')
parser.add_argument('--dmc_ckpt_dir', type=str, default=None, help='Directory shared with actors for the latest PPO weights')
parser.add_argument('--init_weights', type=str, default=None, help='Optional PPO state_dict to resume from')
parser.add_argument('--max_updates', type=int, default=0)
parser.add_argument('--max_seconds', type=float, default=0)
parser.add_argument('--seed', type=int, default=20261105)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def save_weights(model, destination):
    weights = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    temporary = destination.with_suffix('.tmp')
    with open(temporary, 'wb') as stream:
        pickle.dump(weights, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)

def main():
    ActionNumber = 2
    # Parse input parameters
    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Load config file
    load_yaml_config(args, 'learner')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = MLPActorCritic((2, 516 + ActionNumber * 54), ActionNumber).to(device)
    if args.init_weights:
        with open(args.init_weights, 'rb') as f:
            model.set_weights(CPU_Unpickler(f).load())
    agent = PPOAgent(model)
    # Configure experiment directory
    create_experiment_dir(args, 'LEARNER-')    
    save_yaml_config(args.exp_path / 'config.yaml', args, 'learner', agent)
    args.log_path = args.exp_path / 'log'
    args.ckpt_path = args.exp_path / 'ckpt'
    args.ckpt_path.mkdir()
    args.dmc_ckpt_dir = args.dmc_ckpt_dir or str(args.exp_path / 'actor_ckpt')
    actor_ckpt_path = Path(args.dmc_ckpt_dir)
    actor_ckpt_path.mkdir(parents=True, exist_ok=True)
    args.log_path.mkdir()
    model_update_times = 0

    logger.configure(str(args.log_path))

    receiving_condition = multiprocessing.Condition()
    num_receptions = multiprocessing.Value('i', 0)
    received_steps = multiprocessing.Value('q', 0)
    received_trajectories = multiprocessing.Value('q', 0)

    # Start memory pool in another process
    manager = MemPoolManager()
    manager.start()
    mem_pool = manager.MemPool(capacity=args.pool_size)
    receiver = Process(target=recv_data,
                       args=(args.data_port, mem_pool, receiving_condition, num_receptions,
                             args.keep_training, received_steps, received_trajectories))
    receiver.start()

    # Print throughput statistics
    throughput = Process(target=MultiprocessingMemPool.record_throughput,
                         args=(mem_pool, args.record_throughput_interval))
    throughput.start()
    save_weights(model, actor_ckpt_path / '0.pth')
    started = time.monotonic()
    update_seconds = 0.0
    optimizer_steps = 0
    stop_reason = 'interrupted'

    def time_expired():
        return args.max_seconds > 0 and time.monotonic() - started >= args.max_seconds

    try:
        while not time_expired():
            if not receiver.is_alive():
                raise RuntimeError('Trajectory receiver exited unexpectedly')
            if len(mem_pool) < args.batch_size:
                time.sleep(0.02)
                continue
            with receiving_condition:
                while not args.keep_training and num_receptions.value < args.training_freq and not time_expired():
                    receiving_condition.wait(timeout=1)
                if time_expired():
                    break
                if not args.keep_training:
                    num_receptions.value -= args.training_freq
                data = mem_pool.sample(size=args.batch_size)
            update_started = time.monotonic()
            stat = agent.update(data)
            if model_update_times == 0 and stat['initial_logp_error'] > 0.001:
                raise RuntimeError('Initial actor/learner log probabilities disagree; check trajectory alignment')
            optimizer_steps += stat['optimizer_steps']
            update_seconds += time.monotonic() - update_started
            model_update_times += 1
            save_weights(model, actor_ckpt_path / f'{model_update_times}.pth')
            if model_update_times > 4:
                (actor_ckpt_path / f'{model_update_times - 5}.pth').unlink(missing_ok=True)
            if model_update_times % args.ckpt_save_freq == 0:
                save_weights(model, args.ckpt_path / f'ppo{model_update_times}.pth')
            for name, value in stat.items():
                logger.record_tabular(name, value)
            logger.record_tabular('updates', model_update_times)
            logger.record_tabular('elapsed_s', round(time.monotonic() - started, 2))
            logger.record_tabular('update_s', round(time.monotonic() - update_started, 3))
            logger.dump_tabular()
            if args.max_updates > 0 and model_update_times >= args.max_updates:
                stop_reason = 'max_updates'
                break
        else:
            stop_reason = 'max_seconds'
        if time_expired():
            stop_reason = 'max_seconds'
    finally:
        save_weights(model, args.exp_path / 'final.pth')
        torch.save({'model': model.state_dict(), 'optimizer': agent.optimizer.state_dict(),
                    'updates': model_update_times}, args.exp_path / 'training_state.pt')
        elapsed = time.monotonic() - started
        summary = {'updates': model_update_times, 'elapsed_s': elapsed,
                   'updates_per_minute': model_update_times * 60 / elapsed,
                   'learner_update_s': update_seconds, 'stop_reason': stop_reason,
                   'received_steps': received_steps.value,
                   'received_trajectories': received_trajectories.value,
                   'received_steps_per_second': received_steps.value / elapsed,
                   'optimizer_steps': optimizer_steps,
                   'batch_size': args.batch_size, 'training_freq': args.training_freq,
                   'pool_size': args.pool_size,
                   'seed': args.seed, 'init_weights': args.init_weights}
        (args.exp_path / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('Training summary:', json.dumps(summary), flush=True)
        for process in (receiver, throughput):
            process.terminate()
            process.join(timeout=5)
        manager.shutdown()


def recv_data(data_port, mem_pool, receiving_condition, num_receptions, keep_training,
              received_steps, received_trajectories):
    context = zmq.Context()
    data_socket = context.socket(zmq.REP)
    data_socket.bind(f'tcp://*:{data_port}')

    while True:
        data: dict = pickle.loads(data_socket.recv())

        with receiving_condition:
            mem_pool.push(data)
            if not keep_training:
                num_receptions.value += 1
                receiving_condition.notify()
        with received_steps.get_lock():
            received_steps.value += len(data['obs'])
        with received_trajectories.get_lock():
            received_trajectories.value += 1
        data_socket.send(b'200')


if __name__ == '__main__':
    main()
