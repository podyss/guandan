# -*- coding: utf-8 -*-
"""Head-to-head evaluation: team A (seats 0,2) vs team B (seats 1,3),
seat-swapped halfway for fairness."""

import argparse
import random
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor

from .agents import RandomAgent, RuleAgent, NumpyAgent
from .engine import play_round

_WORKER_AGENT_CACHE = {}


def make_agent(spec, seed=None, filter_model=None, candidate_k=2,
               filter_only=False):
    if spec == "dmc":
        if filter_model is None:
            raise ValueError("dmc agent requires --filter-ckpt")
        from .agents import DanZeroDMCOnlyAgent
        return lambda: DanZeroDMCOnlyAgent(filter_model)
    if spec.startswith("danzero-ppo:"):
        if filter_model is None:
            raise ValueError("danzero-ppo requires --filter-ckpt")
        from .agents import DanZeroPPOAgent
        checkpoint = spec.split(":", 1)[1]
        model = DanZeroPPOAgent(checkpoint, filter_model).model
        return lambda: DanZeroPPOAgent(checkpoint, filter_model, model=model)
    if spec == "random":
        return lambda: RandomAgent(seed)
    if spec == "rule":
        return lambda: RuleAgent()
    if spec.endswith(".npz"):
        from .model_np import NumpyModel
        m = NumpyModel(spec)
        return lambda: NumpyAgent(m)
    if spec.endswith(".pt"):
        from .model_torch import load_ckpt
        from .agents import TorchAgent
        model, _ = load_ckpt(spec)
        def factory():
            agent = TorchAgent(model)
            if filter_model is not None:
                agent.with_danzero_filter(filter_model, candidate_k,
                                          filter_only)
            return agent
        return factory
    raise ValueError(spec)


def evaluate(make_a, make_b, games=200, seed=42, log_every=0):
    rng = random.Random(seed)
    wins = 0
    total_reward = 0
    for g in range(games):
        a_seats = (0, 2) if g % 2 == 0 else (1, 3)
        agents = [make_a() if p in a_seats else make_b() for p in range(4)]
        rewards, ranking, _ = play_round(agents, rng=random.Random(rng.getrandbits(48)))
        ra = rewards[a_seats[0]]
        total_reward += ra
        if ra > 0:
            wins += 1
        if log_every and (g + 1) % log_every == 0:
            print("  %d/%d  win %.1f%%  avg reward %.3f"
                  % (g + 1, games, 100.0 * wins / (g + 1),
                     total_reward / (g + 1)))
    return wins / games, total_reward / games


def evaluate_worker(payload):
    import torch
    torch.set_num_threads(1)
    spec_a, spec_b, filter_ckpt, candidate_k, filter_only, seeds = payload
    cache_key = (spec_a, spec_b, filter_ckpt, candidate_k, filter_only)
    factories = _WORKER_AGENT_CACHE.get(cache_key)
    if factories is None:
        filter_model = None
        if filter_ckpt:
            from .danzero_filter import DanZeroFilter
            filter_model = DanZeroFilter.from_checkpoint(filter_ckpt)
        factories = (
            make_agent(spec_a, filter_model=filter_model,
                       candidate_k=candidate_k, filter_only=filter_only),
            make_agent(spec_b, filter_model=filter_model,
                       candidate_k=candidate_k),
        )
        _WORKER_AGENT_CACHE[cache_key] = factories
    make_a, make_b = factories
    results = []
    for game_index, game_seed in seeds:
        a_seats = (0, 2) if game_index % 2 == 0 else (1, 3)
        agents = [make_a() if seat in a_seats else make_b() for seat in range(4)]
        rewards, _, _ = play_round(agents, rng=random.Random(game_seed))
        results.append(rewards[a_seats[0]])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True,
                    help="random|rule|dmc|<ckpt.npz>|<ckpt.pt>")
    ap.add_argument("--b", default="random")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--filter-ckpt", default="",
                    help="DanZero+ actor_torch/q_network.ckpt")
    ap.add_argument("--candidate-k", type=int, default=2)
    ap.add_argument("--filter-only", action="store_true",
                    help="evaluate the frozen DMC top-1 baseline")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    if args.games <= 0 or args.workers <= 0:
        ap.error('--games and --workers must be positive')
    started = time.monotonic()
    if args.workers > 1 or args.output:
        rng = random.Random(args.seed)
        seeds = [(index, rng.getrandbits(48)) for index in range(args.games)]
        chunks = [seeds[index:index + 20] for index in range(0, args.games, 20)]
        payloads = [(args.a, args.b, args.filter_ckpt, args.candidate_k,
                     args.filter_only, chunk) for chunk in chunks]
        rewards = []
        if args.workers == 1:
            batches = map(evaluate_worker, payloads)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            batches = executor.map(evaluate_worker, payloads)
        try:
            for batch in batches:
                rewards.extend(batch)
                if args.log_every and len(rewards) % args.log_every < len(batch):
                    print('%d/%d win %.1f%%' % (len(rewards), args.games,
                          100 * sum(reward > 0 for reward in rewards) / len(rewards)), flush=True)
        finally:
            if executor:
                executor.shutdown()
        wins = sum(reward > 0 for reward in rewards)
        wr = wins / args.games
        denominator = 1 + 1.96 ** 2 / args.games
        center = (wr + 1.96 ** 2 / (2 * args.games)) / denominator
        radius = 1.96 * math.sqrt(wr * (1 - wr) / args.games +
                                 1.96 ** 2 / (4 * args.games ** 2)) / denominator
        result = {'a': args.a, 'b': args.b, 'games': args.games, 'seed': args.seed,
                  'wins': wins, 'win_rate': wr, 'mean_reward': sum(rewards) / args.games,
                  'win_rate_ci95': [center - radius, center + radius],
                  'filter_ckpt': args.filter_ckpt, 'candidate_k': args.candidate_k,
                  'elapsed_s': time.monotonic() - started, 'rewards': rewards}
        if args.output:
            from pathlib import Path
            Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        print('A=%s vs B=%s : win rate %.1f%%, avg reward %.3f, 95%% CI [%.1f%%, %.1f%%]'
              % (args.a, args.b, wr * 100, result['mean_reward'],
                 100 * (center - radius), 100 * (center + radius)))
        return
    filter_model = None
    if args.filter_ckpt:
        from .danzero_filter import DanZeroFilter
        filter_model = DanZeroFilter.from_checkpoint(args.filter_ckpt)
    wr, avg = evaluate(make_agent(args.a, filter_model=filter_model,
                                  candidate_k=args.candidate_k,
                                  filter_only=args.filter_only),
                       make_agent(args.b,
                                  filter_model=filter_model,
                                  candidate_k=args.candidate_k),
                       args.games, args.seed, args.log_every)
    print("A=%s vs B=%s : win rate %.1f%%, avg reward %.3f"
          % (args.a, args.b, wr * 100, avg))


if __name__ == "__main__":
    main()
