# -*- coding: utf-8 -*-
"""Smoke tests: python tests/test_all.py  (or pytest)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fabledan.combos import gen_moves, classify_claim, claim_ids, TYPE_NAMES
from fabledan.encode import encode_decision, VOCAB
from fabledan.engine import play_round
from fabledan.agents import RandomAgent, RuleAgent


def test_claim_roundtrip():
    rng = random.Random(7)
    deck = list(range(108))
    for _ in range(300):
        rng.shuffle(deck)
        hand, lv = deck[:27], rng.randrange(13)
        for m in gen_moves(hand, lv, None):
            ci = claim_ids(m)
            m2 = classify_claim(m.cards, ci, lv)
            assert m2.type == m.type and m2.key == m.key, \
                (TYPE_NAMES[m.type], TYPE_NAMES[m2.type])


def test_engine_invariants():
    for g in range(100):
        agents = [RandomAgent(g * 4 + i) for i in range(4)]
        rewards, ranking, rnd = play_round(agents, rng=random.Random(g))
        assert sorted(rewards) in ([-3, -3, 3, 3], [-2, -2, 2, 2], [-1, -1, 1, 1])
        played = sum(len(e[2].cards) for e in rnd.events if e[0] == 'play')
        assert played + sum(len(h) for h in rnd.hands) == 108


def test_encode():
    def cb(p, obs, legal, idx):
        toks, feats = encode_decision(obs)
        assert all(0 <= t < VOCAB for t in toks)
        assert feats.shape[0] == len(legal)
    for g in range(10):
        play_round([RandomAgent(i) for i in range(4)],
                   rng=random.Random(g), sample_cb=cb)


def test_rule_beats_random():
    from fabledan.evaluate import evaluate
    wr, _ = evaluate(lambda: RuleAgent(), lambda: RandomAgent(0),
                     games=100, seed=1)
    assert wr > 0.7, wr


def test_ring_runner_with_belief():
    import numpy as np
    from fabledan.ring import RingRunner
    rng = np.random.default_rng(0)
    r = RingRunner(ring=4, seed=1, eps=0.05, top_k=5)
    total = 0
    for _ in range(150):
        reqs = r.collect_requests()
        r.step({req[0]: rng.standard_normal(req[2].shape[0]).astype(np.float32)
                for req in reqs})
        for toks, feat, z, belief in r.pop_episodes():
            total += 1
            assert belief.shape == (45,) and -1 <= z <= 1
            # normalized counts: per-rank 0..8 cards -> /4 gives 0..2
            assert 0 <= belief.min() and belief.max() <= 2.0
            # three hidden hands hold <= 81 cards in total -> /4
            assert belief.sum() <= 81 / 4 + 1e-5
    assert r.episodes_done > 0 and total > 0


def test_danzero_filter_adapter():
    import numpy as np
    from fabledan.danzero_filter import (DanZeroFilter, _card_vector,
                                         encode_danzero_candidates)
    from fabledan.engine import GuandanRound, random_tribute_mode

    rng = random.Random(17)
    rnd = GuandanRound(0, rng, random_tribute_mode(rng))
    obs = next(rnd.play_steps())
    features = encode_danzero_candidates(obs)
    assert features.shape == (len(obs["legal"]), 567)
    filter_path = os.path.join(os.path.dirname(__file__),
                               "../Danzero_plus/actor_torch/q_network.ckpt")
    filter_model = DanZeroFilter.from_checkpoint(filter_path)
    scores = filter_model.score(features).numpy()
    assert scores.shape == (len(obs["legal"]),)
    assert all(not parameter.requires_grad for parameter in filter_model.parameters())
    assert np.flatnonzero(_card_vector([4])).tolist() == [0]
    assert np.flatnonzero(_card_vector([7])).tolist() == [2]
    assert np.flatnonzero(_card_vector([5])).tolist() == [3]
    assert np.flatnonzero(_card_vector([0])).tolist() == [48]
    assert np.flatnonzero(_card_vector([52, 53])).tolist() == [52, 53]
    teammate = (obs["player"] + 2) % 4
    obs_with_finished_teammate = dict(obs)
    obs_with_finished_teammate["done"] = list(obs["done"])
    obs_with_finished_teammate["done"][teammate] = True
    finished_features = encode_danzero_candidates(obs_with_finished_teammate)
    assert np.all(finished_features[:, 174:228] == -1)
    from fabledan.agents import TorchAgent
    from fabledan.model_torch import FableDanNet, ModelConfig
    selector = FableDanNet(ModelConfig(n_blocks=1))
    selected = TorchAgent(selector).with_danzero_filter(filter_model, 2).act(obs)
    assert 0 <= selected < len(obs["legal"])


def test_ring_candidate_index_mapping():
    import numpy as np
    from fabledan.ring import RingRunner

    runner = RingRunner(ring=1, seed=11, eps=0.0)
    slot, _tokens, features, _filter_features = runner.collect_requests()[0]
    selected_index = min(1, len(features) - 1)
    runner.step({slot: (np.array([selected_index]), np.array([0.0]))})
    assert np.array_equal(runner.games[slot].samples[0][1],
                          features[selected_index])


def test_parallel_evaluation_seeds():
    from fabledan.evaluate import evaluate, evaluate_worker

    seed = 9103
    games = 8
    rng = random.Random(seed)
    seeds = [(index, rng.getrandbits(48)) for index in range(games)]
    rewards = evaluate_worker(('rule', 'rule', '', 2, False, seeds))
    win_rate, mean_reward = evaluate(lambda: RuleAgent(), lambda: RuleAgent(),
                                     games=games, seed=seed)
    assert win_rate == sum(reward > 0 for reward in rewards) / games
    assert mean_reward == sum(rewards) / games


def test_ppo_factory_shares_model():
    import pickle
    import tempfile
    import torch
    from pathlib import Path
    from Danzero_plus.learner_torch.model import MLPActorCritic
    from fabledan.evaluate import make_agent

    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'weights.pth'
        model = MLPActorCritic((2, 624), 2)
        with checkpoint.open('wb') as stream:
            pickle.dump(model.state_dict(), stream)
        factory = make_agent('danzero-ppo:' + str(checkpoint), filter_model=object())
        first, second = factory(), factory()
        assert first is not second
        assert first.model is second.model
        assert next(first.model.parameters()).device.type == 'cpu'


if __name__ == "__main__":
    test_claim_roundtrip(); print("claim roundtrip OK")
    test_engine_invariants(); print("engine invariants OK")
    test_encode(); print("encode OK")
    test_rule_beats_random(); print("rule>random OK")
    test_ring_runner_with_belief(); print("ring+belief OK")
    test_danzero_filter_adapter(); print("Danzero filter adapter OK")
    test_ring_candidate_index_mapping(); print("candidate index mapping OK")
    test_parallel_evaluation_seeds(); print("parallel evaluation seeds OK")
    test_ppo_factory_shares_model(); print("PPO model sharing OK")
    print("ALL TESTS PASSED")
