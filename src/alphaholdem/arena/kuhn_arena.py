from __future__ import annotations

import numpy as np
from .policy.ppo_kuhn_policy import PPOKuhnPolicy
from .policy.cfr_kuhn_policy import CFRKuhnPolicy
from ..poker.kuhn_poker_env import KuhnPokerEnv

class KuhnNode():
    def __init__(
        self,
        player: int,
        son: list[KuhnNode],
        action_history: str,
        action_prob: list[float],
        is_terminal: bool,
        hole_cards: str,
        payoff: list[float] = [0, 0],
    ) -> None:
        self.player = player
        self.son = son
        self.action_history = action_history
        self.action_prob = action_prob
        self.is_terminal = is_terminal
        self.hole_cards = hole_cards
        self.payoff = payoff
    
    def dfs_ev(self, reach_prob: float) -> float:
        if self.is_terminal:
            return reach_prob * self.payoff[0]
        payoff = 0.0
        for i, child in enumerate(self.son):
            payoff += child.dfs_ev(reach_prob * self.action_prob[i]) 
        return payoff

class KuhnTree():
    def __init__(self, strategy: list[dict]):
        self.root = self.create_root(strategy)

    def dfs_ev(self) -> float:
        return self.root.dfs_ev(reach_prob=1.0)

    def create_root(self, strategy: list[dict]) -> KuhnNode:
        hole_cards_combo = ['JQ', 'JK', 'QJ', 'QK', 'KJ', 'KQ']
        son = [
            self.create_node(
                strategy=strategy,
                player=0,
                hole_cards=h,
                action_history='',
            )
            for h in hole_cards_combo
        ]
        return KuhnNode(
            player=2,
            son=son,
            action_history="",
            action_prob=[1/6] * 6,
            is_terminal=False,
            hole_cards="",
            payoff=[],
        )

    def create_node(self, strategy: dict, player: int, hole_cards: str, action_history: str) -> KuhnNode:
        if action_history in ['cc', 'crf', 'crc', 'rf', 'rc']:
            mapping = {'J': 0, 'Q': 1, 'K': 2}
            showdown = 1 if mapping[hole_cards[0]] > mapping[hole_cards[1]] else -1
            payoff = {
                'cc': [showdown, -showdown],
                'crf': [-1.0, 1.0],
                'crc': [2 * showdown, -2 * showdown],
                'rf': [1.0, -1.0],
                'rc': [2 * showdown, -2 * showdown],
            }[action_history]
            return KuhnNode(
                player=2,
                son=[],
                action_history=action_history,
                action_prob=[],
                is_terminal=True,
                hole_cards=hole_cards,
                payoff=payoff,
            )

        key = hole_cards[player] + ':' + action_history
        action_prob = strategy[player][key]
        if action_history.endswith('r'):
            # f / c
            son=[
                self.create_node(strategy, player=1 - player, hole_cards=hole_cards, action_history=action_history + 'f'),
                self.create_node(strategy, player=1 - player, hole_cards=hole_cards, action_history=action_history + 'c'),
            ]
        else:
            # c / r
            son=[
                self.create_node(strategy, player=1 - player, hole_cards=hole_cards, action_history=action_history + 'c'),
                self.create_node(strategy, player=1 - player, hole_cards=hole_cards, action_history=action_history + 'r'),
            ]
        return KuhnNode(
            player=player,
            son=son,
            action_history=action_history,
            action_prob=action_prob,
            is_terminal=False,
            hole_cards=hole_cards,
            payoff=[],
        )

class KuhnArena():
    def __init__(self, strategy_path: str = 'strategy/kuhn.txt') -> None:
        self.cfr = CFRKuhnPolicy(strategy_path)

    def cfr_self_play(
        self,
        cfr1: CFRKuhnPolicy = None,
        cfr2: CFRKuhnPolicy = None,
        runs: int = 1024,
    ) -> tuple[float, float]:
        cfr1 = self.cfr if cfr1 is None else cfr1
        cfr2 = self.cfr if cfr2 is None else cfr2
        return (KuhnTree([cfr1.policy, cfr2.policy]).dfs_ev() - KuhnTree([cfr2.policy, cfr1.policy]).dfs_ev()) / 2 * 100, 0

    def _to_cfr_policy(self, strategy: np.ndarray) -> CFRKuhnPolicy:
        # Fold Check Call Raise
        node_name = ['J:', 'Q:', 'K:', 'J:cr', 'Q:cr', 'K:cr', 'J:c', 'Q:c', 'K:c', 'J:r', 'Q:r', 'K:r']
        policy = { node_name[i]: strategy[i] for i in range(12) }
        return CFRKuhnPolicy(policy = policy)

    def ppo_vs_cfr(
        self,
        ppo: PPOKuhnPolicy,
        cfr: CFRKuhnPolicy = None,
        runs: int = 1024,
        batch_size: int = 32,
    ) -> tuple[float, float]:
        return self.cfr_self_play(
            cfr1=self._to_cfr_policy(ppo.get_range_policy()),
            cfr2=cfr,
            runs=runs
        )

    def ppo_vs_ppo(
        self,
        ppo1: PPOKuhnPolicy,
        ppo2: PPOKuhnPolicy,
        runs: int = 1024,
        batch_size: int = 32,
    ) -> tuple[float, float]:
        return self.cfr_self_play(
            cfr1=self._to_cfr_policy(ppo1.get_range_policy()),
            cfr2=self._to_cfr_policy(ppo2.get_range_policy()),
            runs=runs
        )

    def ppos_melee(
        self,
        ppos: list[PPOKuhnPolicy],
        runs: int = 1024,
        batch_size: int = 32,
    ) -> list[float]:
        cfrs = [self._to_cfr_policy(ppo.get_range_policy()) for ppo in ppos]
        scores = [0] * len(ppos)
        for i in range(len(ppos)):
            for j in range(i + 1, len(ppos)):
                mean, var = self.cfr_self_play(cfr1=cfrs[i], cfr2=cfrs[j], runs=runs)
                scores[i] += mean / (len(ppos) - 1)
                scores[j] -= mean / (len(ppos) - 1)
        return scores