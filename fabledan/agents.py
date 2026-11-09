# -*- coding: utf-8 -*-
"""Agents: Random, greedy Rule baseline, and numpy-model agent."""

import random

import numpy as np

from .combos import (BOMB, PASS, ROCKET, SFLUSH, SINGLE, PAIR, TRIPLE)
from .encode import encode_decision


class RandomAgent:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, obs):
        return self.rng.randrange(len(obs["legal"]))


class RuleAgent:
    """Greedy heuristic: lead with longest/smallest combo, follow with the
    smallest beating move; avoid breaking bombs unless few cards left."""

    def act(self, obs):
        legal = obs["legal"]
        lead = obs["lead"]
        if lead is None or lead.type == PASS:
            # leading: prefer long non-bomb combos with small key
            def lead_score(i):
                m = legal[i]
                bombish = m.is_bombish()
                return (1 if bombish else 0, -m.size, m.key)
            return min(range(len(legal)), key=lead_score)
        # following: smallest beating move, prefer non-bomb; pass if only
        # bombs and many cards left
        cand = [(i, m) for i, m in enumerate(legal) if m.type != PASS]
        if not cand:
            return 0
        nonbomb = [(i, m) for i, m in cand if not m.is_bombish()]
        if nonbomb:
            return min(nonbomb, key=lambda im: (im[1].key, im[1].size))[0]
        # only bombs can beat: use one if our team is behind, else 30%
        left = obs["left"]
        me = obs["player"]
        opp_min = min(left[(me + 1) % 4], left[(me + 3) % 4])
        if opp_min <= 8 or len(obs["hand"]) <= 10:
            return min(cand, key=lambda im: (im[1].bomb_tier(), im[1].key))[0]
        return 0  # pass


class NumpyAgent:
    """Greedy w.r.t. numpy model Q-values."""

    def __init__(self, model, eps=0.0, rng=None):
        self.model = model
        self.eps = eps
        self.rng = rng or random.Random()

    def act(self, obs):
        if self.eps > 0 and self.rng.random() < self.eps:
            return self.rng.randrange(len(obs["legal"]))
        toks, feats = encode_decision(obs)
        q = self.model.q_values(toks, feats)
        return int(np.argmax(q))


class TorchAgent:
    """Greedy w.r.t. torch model (used inside training actors / eval)."""

    def __init__(self, model, device="cpu", eps=0.0, top_k=0, rng=None):
        import torch
        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.eps = eps
        self.top_k = top_k
        self.rng = rng or random.Random()
        self.filter_model = None
        self.candidate_k = 0
        self.filter_only = False

    def with_danzero_filter(self, filter_model, candidate_k=2,
                            filter_only=False):
        self.filter_model = filter_model
        self.candidate_k = candidate_k
        self.filter_only = filter_only
        return self

    def act(self, obs):
        import torch
        from .encode import pad_tokens
        toks, feats = encode_decision(obs)
        n = len(obs["legal"])
        candidates = np.arange(n)
        filter_q = None
        if self.filter_model is not None:
            from .danzero_filter import encode_danzero_candidates
            filter_feats = encode_danzero_candidates(obs)
            filter_q = self.filter_model.score(filter_feats).numpy()
            candidates = np.argsort(filter_q)[-min(self.candidate_k, n):]
            if self.filter_only:
                return int(candidates[np.argmax(filter_q[candidates])])
            if self.eps > 0 and self.rng.random() < self.eps:
                return int(self.rng.choice(list(candidates)))
        if self.eps > 0 and self.rng.random() < self.eps:
            if self.top_k and self.top_k > 1:
                pass  # fall through to model, sample among top-k below
            else:
                return self.rng.randrange(n)
        with torch.no_grad():
            t = torch.tensor([toks], dtype=torch.long, device=self.device)
            ln = torch.tensor([len(toks)], device=self.device)
            f = torch.tensor(feats[candidates][None], dtype=torch.float32,
                             device=self.device)
            q, _ = self.model(t, ln, f)
            q = q[0].cpu().numpy()
        if self.eps > 0 and self.top_k and self.top_k > 1 \
                and self.rng.random() < self.eps:
            k = min(self.top_k, len(candidates))
            top = np.argsort(q)[-k:]
            return int(candidates[self.rng.choice(list(top))])
        return int(candidates[int(np.argmax(q))])


class DanZeroDMCOnlyAgent:
    def __init__(self, filter_model):
        self.filter_model = filter_model

    def act(self, obs):
        from .danzero_filter import encode_danzero_candidates
        scores = self.filter_model.score(encode_danzero_candidates(obs)).numpy()
        return int(np.argmax(scores))


class DanZeroPPOAgent:
    def __init__(self, checkpoint, filter_model, model=None):
        import io
        import pickle
        import torch
        from Danzero_plus.learner_torch.model import MLPActorCritic

        self.torch = torch
        self.filter_model = filter_model
        if model is None:
            class CPUUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'torch.storage' and name == '_load_from_bytes':
                        return lambda payload: torch.load(io.BytesIO(payload), map_location='cpu')
                    return super().find_class(module, name)
            model = MLPActorCritic((2, 624), 2).cpu().eval()
            with open(checkpoint, "rb") as stream:
                model.load_state_dict(CPUUnpickler(stream).load())
        self.model = model

    def act(self, obs):
        from .danzero_filter import encode_danzero_candidates

        features = encode_danzero_candidates(obs)
        dmc_scores = self.filter_model.score(features).numpy()
        candidate_ids = np.argsort(dmc_scores)[-min(2, len(features)):][::-1]
        base = np.concatenate((features[0, :-54], np.zeros(3, dtype=np.float32)))
        action_features = features[candidate_ids, -54:].reshape(-1)
        if len(candidate_ids) < 2:
            action_features = np.concatenate((action_features, -np.ones(54, dtype=np.float32)))
        state = np.concatenate((base, action_features)).astype(np.float32)
        legal = np.zeros(2, dtype=np.float32)
        legal[:len(candidate_ids)] = 1
        with self.torch.no_grad():
            distribution, _, _ = self.model(
                self.torch.from_numpy(state), None,
                self.torch.from_numpy(legal))
            choice = int(self.torch.argmax(distribution.logits).item())
        return int(candidate_ids[choice])
