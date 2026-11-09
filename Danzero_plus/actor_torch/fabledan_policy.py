import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fabledan.cards import level_rank, order_of, rank_of
from fabledan.combos import (BOMB, FULL, PAIR, PASS, PLATE, ROCKET, SFLUSH,
                             SINGLE, STRAIGHT, TRIPLE, TUBE, Move, gen_moves)
from fabledan.danzero_filter import DanZeroFilter
from fabledan.encode import encode_decision, pad_tokens
from fabledan.model_torch import load_ckpt


ACTION_TYPES = {
    'PASS': PASS,
    'Single': SINGLE,
    'Pair': PAIR,
    'Trips': TRIPLE,
    'ThreeWithTwo': FULL,
    'Straight': STRAIGHT,
    'ThreePair': PLATE,
    'TwoTrips': TUBE,
    'Bomb': BOMB,
    'StraightFlush': SFLUSH,
    'Rocket': ROCKET,
}
SUIT_INDEX = {'H': 0, 'D': 1, 'S': 2, 'C': 3}
RANK_INDEX = {'A': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5,
              '7': 6, '8': 7, '9': 8, 'T': 9, 'J': 10, 'Q': 11,
              'K': 12}
NATIVE_RANK = {value: key for key, value in RANK_INDEX.items()}
NATIVE_RANK.update({'B': 13, 'R': 14})


def level_id(value):
    if isinstance(value, int):
        if value in range(13):
            return value
        if value == 14:
            return 0
        if value == 13:
            return 12
        raise ValueError(f'unknown native level index {value!r}')
    value = str(value)
    if value == 'T':
        value = '10'
    return level_rank(value)


def card_id(name, duplicate=0):
    if name == 'SB':
        return 52 + 54 * duplicate
    if name == 'HR':
        return 53 + 54 * duplicate
    if len(name) != 2 or name[0] not in SUIT_INDEX or name[1] not in RANK_INDEX:
        raise ValueError(f'unknown native card {name!r}')
    return RANK_INDEX[name[1]] * 4 + SUIT_INDEX[name[0]] + 54 * duplicate


def native_card_name(card):
    base = card % 54
    if base == 52:
        return 'SB'
    if base == 53:
        return 'HR'
    suits = ('H', 'D', 'S', 'C')
    return suits[base % 4] + NATIVE_RANK[base // 4]


def cards_to_ids(names):
    occurrences = defaultdict(int)
    result = []
    for name in names:
        duplicate = occurrences[name]
        occurrences[name] += 1
        result.append(card_id(name, duplicate))
    return result


def action_signature(action):
    action_type, _, cards = action
    if action_type == 'PASS' or cards == 'PASS':
        return PASS, ()
    return ACTION_TYPES[action_type], tuple(sorted(cards))


def move_from_native(action, level):
    action_type, rank_name, names = action
    move_type = ACTION_TYPES[action_type]
    if move_type == PASS:
        return Move(PASS, 0, [], [])
    cards = cards_to_ids(names)
    actual_ranks = [rank_of(card) for card in cards]
    wildcard = [rank_of(card) == level and card % 54 % 4 == 0
                for card in cards]
    primary_rank = NATIVE_RANK.get(str(rank_name), actual_ranks[0])
    if move_type in (SINGLE, PAIR, TRIPLE, BOMB, SFLUSH, ROCKET):
        claims = [primary_rank] * len(cards)
    elif move_type == FULL:
        natural_counts = Counter(rank for rank, is_wild in zip(actual_ranks, wildcard)
                                 if not is_wild)
        trip_rank = max(natural_counts, key=natural_counts.get)
        pair_ranks = [rank for rank, count in natural_counts.items()
                      if rank != trip_rank and count >= 2]
        pair_rank = pair_ranks[0] if pair_ranks else trip_rank
        claims = [trip_rank] * 3 + [pair_rank] * 2
    else:
        claims = [rank for rank, is_wild in zip(actual_ranks, wildcard)
                  if not is_wild]
        claims.extend([primary_rank] * (len(cards) - len(claims)))
    claims = claims[:len(cards)]
    return Move(move_type, order_of(primary_rank, level), cards, claims)


def history_move(action, level):
    action_type, _, cards = action
    move_type = ACTION_TYPES[action_type]
    if move_type == PASS:
        return Move(PASS, 0, [], [])
    card_ids = cards_to_ids(cards)
    signature = (move_type, tuple(sorted(cards)))
    for move in gen_moves(card_ids, level):
        if (move.type, tuple(sorted(native_card_name(card)
                                    for card in move.cards))) == signature:
            return move
    claims = [rank_of(card) for card in card_ids]
    key = max((order_of(rank, level) for rank in claims), default=0)
    return Move(move_type, key, card_ids, claims)


class FableDanPolicy:
    def __init__(self, checkpoint, dmc_checkpoint, inference_endpoint=None,
                 device='cpu', precision='fp32'):
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.use_amp = precision == 'bf16'
        if self.use_amp and self.device.type != 'cuda':
            raise ValueError('bf16 inference requires a CUDA device')
        self.inference_client = None
        if inference_endpoint:
            from fabledan.eval_inference import InferenceClient
            self.inference_client = InferenceClient(inference_endpoint)
            metadata = self.inference_client.request('metadata')
            self.candidate_k = int(metadata['candidate_k'])
            self.model = self.filter = None
        else:
            self.model, self.ckpt = load_ckpt(checkpoint, device=device)
            self.candidate_k = int(self.ckpt.get('meta', {}).get('candidate_k', 2))
            self.model.eval()
            self.filter = DanZeroFilter.from_checkpoint(dmc_checkpoint)
            if self.device.type == 'cuda':
                self.filter = self.filter.to(self.device)
            del self.ckpt
        if self.candidate_k < 1:
            raise ValueError('checkpoint candidate_k must be positive')
        self.events = []
        self.level = None
        self.mypos = 0
        self._move_cache_key = None
        self._move_cache = None

    def _moves_by_signature(self, hand, level):
        cache_key = (level, tuple(sorted(hand)))
        if cache_key != self._move_cache_key:
            generated = [Move(PASS, 0, [], [])] + gen_moves(hand, level)
            by_signature = {}
            for move in generated:
                names = tuple(sorted(native_card_name(card)
                                     for card in move.cards))
                by_signature.setdefault((move.type, names), move)
            self._move_cache_key = cache_key
            self._move_cache = by_signature
        return self._move_cache

    def observe(self, message):
        stage = message.get('stage')
        if message.get('type') != 'notify':
            return
        if stage == 'beginning':
            self.events = []
            self.level = level_id(message['curRank'])
            self.mypos = message['myPos']
        elif stage == 'play' and message.get('curAction'):
            action = message['curAction']
            if action[0] == 'PASS' or action[2] == 'PASS':
                move = Move(PASS, 0, [], [])
                self.events.append(('pass', message['curPos'], move))
            else:
                move = history_move(action, self.level)
                self.events.append(('play', message['curPos'], move))
        elif stage == 'tribute':
            for payer, _, card in message.get('result', []):
                self.events.append(('tribute', payer, rank_of(cards_to_ids([card])[0])))
        elif stage == 'back':
            for item in message.get('result') or []:
                if len(item) >= 3:
                    player, _, card = item[:3]
                    self.events.append(('return', player, rank_of(cards_to_ids([card])[0])))
        elif stage == 'episodeOver':
            self.events = []

    def choose(self, message, dmc_features):
        level = level_id(message['curRank'])
        hand = cards_to_ids(message['handCards'])
        legal_native = message['actionList']
        if self.inference_client is not None:
            candidate_ids = self.inference_client.request(
                'candidates', features=dmc_features)
        else:
            dmc_scores = self.filter.score(dmc_features).cpu().numpy()
            candidate_ids = np.argsort(dmc_scores)[-min(self.candidate_k, len(legal_native)):][::-1]

        by_signature = self._moves_by_signature(hand, level)

        candidate_moves = []
        for index in candidate_ids:
            signature = action_signature(legal_native[int(index)])
            move = by_signature.get(signature)
            if move is None:
                move = move_from_native(legal_native[int(index)], level)
            candidate_moves.append(move)

        me = self.mypos
        public = message.get('publicInfo', [])
        left = [int(player.get('rest', 27)) for player in public]
        if len(left) != 4:
            left = [27] * 4
        done = [count == 0 for count in left]
        lead = None
        greater = message.get('greaterAction')
        if isinstance(greater, list) and len(greater) == 3 and greater[0] in ACTION_TYPES:
            if greater[0] != 'PASS':
                lead = history_move(greater, level)
        obs = {'player': me, 'level': level, 'hand': hand, 'left': left,
               'done': done, 'lead': lead, 'events': self.events,
               'legal': candidate_moves}
        tokens, features = encode_decision(obs)
        if self.inference_client is not None:
            selected = self.inference_client.request(
                'select', tokens=tokens, features=features)
            return int(candidate_ids[selected])
        padded, length = pad_tokens(
            tokens, length=len(tokens) if self.device.type == 'cuda' else None)
        with torch.inference_mode(), torch.autocast(
                'cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            q, _ = self.model(torch.as_tensor(padded[None], dtype=torch.long, device=self.device),
                              torch.as_tensor([length], device=self.device),
                              torch.as_tensor(features[None], dtype=torch.float32, device=self.device))
        selected = int(torch.argmax(q[0]).item())
        return int(candidate_ids[selected])
