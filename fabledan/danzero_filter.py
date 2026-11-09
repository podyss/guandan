"""Compatibility adapter for DanZero+'s frozen 567-feature DMC model."""

import pickle

import numpy as np
import torch
from torch import nn

from .cards import rank_of, suit_of


def _danzero_card_index(card):
    rank = rank_of(card)
    if rank >= 13:
        return 52 + rank - 13
    danzero_suit = (0, 3, 1, 2)[suit_of(card)]
    danzero_rank = 12 if rank == 0 else rank - 1
    return danzero_rank * 4 + danzero_suit


def _card_vector(cards):
    result = np.zeros(54, dtype=np.float32)
    for card in cards:
        result[_danzero_card_index(card)] += 1
    return result


def _universal_features(hand, current_rank):
    result = np.zeros(12, dtype=np.float32)
    wildcard = (current_rank - 1) * 4
    if hand[wildcard] == 0:
        return result
    result[0] = 1
    for suit in range(4):
        ranks = [hand[suit + rank * 4] if suit + rank * 4 != wildcard
                 else 0 for rank in range(13)]
        if any(sum(value == 0 for value in ranks[start:start + 5]) <= 1
               for start in range(9)):
            result[1] = 1
            break
    rank_counts = [sum(hand[suit + rank * 4]
                       for suit in range(4) if suit + rank * 4 != wildcard)
                   for rank in range(13)]
    max_count = max(rank_counts)
    if max_count >= 6:
        result[2:8] = 1
    elif max_count == 5:
        result[3:8] = 1
    elif max_count == 4:
        result[4:8] = 1
    elif max_count == 3:
        result[5:8] = 1
    elif max_count == 2:
        result[6:8] = 1
    else:
        result[7] = 1
    for rank in range(13):
        if rank >= 1:
            if (rank_counts[rank] == 2 and rank_counts[rank - 1] >= 3 or
                    rank_counts[rank] >= 3 and rank_counts[rank - 1] == 2):
                result[9] = 1
            elif rank_counts[rank] == rank_counts[rank - 1] == 2:
                result[11] = 1
        if rank >= 2 and any((rank_counts[rank - 2] == 1 and
                              rank_counts[rank - 1] >= 2 and
                              rank_counts[rank] >= 2,
                              rank_counts[rank - 2] >= 2 and
                              rank_counts[rank - 1] == 1 and
                              rank_counts[rank] >= 2,
                              rank_counts[rank - 2] >= 2 and
                              rank_counts[rank - 1] >= 2 and
                              rank_counts[rank] == 1)):
            result[10] = 1
    trailing_run = 0
    for count in rank_counts:
        trailing_run = trailing_run + 1 if count else 0
    if trailing_run >= 4:
        result[8] = 1
    return result


def encode_danzero_candidates(obs):
    """Encode legal moves in DanZero+'s original DMC feature layout."""
    me = obs["player"]
    hand = _card_vector(obs["hand"])
    current_rank = 13 if obs["level"] == 0 else obs["level"]
    universal = _universal_features(hand, current_rank)
    played_by_player = [[] for _ in range(4)]
    play_events = [event for event in obs["events"]
                   if event[0] in ("play", "pass")]
    for event in play_events:
        if event[0] == "play":
            played_by_player[event[1]].extend(event[2].cards)

    unseen = np.full(54, 2, dtype=np.float32)
    for player, cards in enumerate(played_by_player):
        if player != me:
            unseen -= _card_vector(cards)
    unseen = np.maximum(unseen, 0)
    role_played = [_card_vector(played_by_player[(me + rel) % 4])
                   for rel in (1, 2, 3)]
    last_action = np.full(54, -1, dtype=np.float32)
    teammate_action = np.full(54, -1, dtype=np.float32)
    teammate = (me + 2) % 4
    for event in play_events:
        if event[0] == "pass":
            action = np.full(54, -1, dtype=np.float32)
        else:
            action = _card_vector(event[2].cards)
        last_action = action
        if event[1] == teammate and not obs["done"][teammate]:
            teammate_action = action

    rank_one_hot = np.zeros(13, dtype=np.float32)
    rank_one_hot[current_rank - 1] = 1
    counts_one_hot = []
    for rel in (1, 2, 3):
        left = obs["left"][(me + rel) % 4]
        vector = np.zeros(28, dtype=np.float32)
        vector[min(left, 27)] = 1
        counts_one_hot.append(vector)

    base = np.concatenate((hand, universal, unseen, last_action,
                           teammate_action, *role_played, *counts_one_hot,
                           rank_one_hot, rank_one_hot, rank_one_hot))
    rows = []
    for move in obs["legal"]:
        rows.append(np.concatenate((base, _card_vector(move.cards))))
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (len(obs["legal"]), 567):
        raise ValueError(f"unexpected DanZero feature shape: {result.shape}")
    return result


class DanZeroFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Sequential(*[
            layer for index in range(6)
            for layer in ((nn.Linear(567 if index == 0 else 512,
                                     1 if index == 5 else 512), nn.Tanh())
                          if index < 5 else
                          (nn.Linear(512, 1), nn.Identity()))
        ])

    @classmethod
    def from_checkpoint(cls, path):
        with open(path, "rb") as checkpoint_file:
            weights = pickle.load(checkpoint_file)
        model = cls()
        keys = [f"{layer}.{suffix}" for layer in (0, 2, 4, 6, 8, 10)
                for suffix in ("weight", "bias")]
        state = {}
        for key, weight in zip(keys, weights):
            tensor = torch.as_tensor(np.asarray(weight).T.copy()
                                     if key.endswith("weight") else
                                     np.asarray(weight).copy(), dtype=torch.float32)
            state[f"q.{key}"] = tensor
        model.load_state_dict(state)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def score(self, features):
        with torch.no_grad():
            return self.q(torch.as_tensor(
                features, dtype=torch.float32,
                device=next(self.parameters()).device)).squeeze(-1)
