from __future__ import annotations

from typing_extensions import override
import numpy as np

from alphaholdem.arena.policy.policy import Policy
from .arena import Arena
from .policy.ppo_kuhn_policy import PPOKuhnPolicy
from .policy.ppo_range_kuhn_policy import PPORangeKuhnPolicy
from .policy.lookup_kuhn_policy import LookupKuhnPolicy

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
            action_prob=[1 / 6] * 6,
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

class KuhnArena(Arena):
    def __init__(self, nash_path: str = 'strategy/kuhn.txt') -> None:
        super().__init__()
        self.nash = LookupKuhnPolicy(nash_path)

    def _to_lookup_policy(self, strategy: np.ndarray) -> LookupKuhnPolicy:
        # Fold Check Call Raise
        node_name = ['J:', 'Q:', 'K:', 'J:cr', 'Q:cr', 'K:cr', 'J:c', 'Q:c', 'K:c', 'J:r', 'Q:r', 'K:r']
        policy = { node_name[i]: strategy[i] for i in range(12) }
        return LookupKuhnPolicy(policy = policy)

    @override
    @property
    def nash_policy(self) -> LookupKuhnPolicy:
        return self.nash

    @override
    def validate_policy(self, policy: Policy) -> None:
        assert type(policy) in [LookupKuhnPolicy, PPOKuhnPolicy, PPORangeKuhnPolicy]

    @override
    def policy_vs_policy(
        self,
        policy0: Policy,
        policy1: Policy,
        runs: int = 1024
    ) -> tuple[float, float]:
        self.validate_policy(policy0)
        self.validate_policy(policy1)
        if type(policy0) in [PPOKuhnPolicy, PPORangeKuhnPolicy]:
            policy0 = self._to_lookup_policy(policy0.get_range_policy())
        if type(policy1) in [PPOKuhnPolicy, PPORangeKuhnPolicy]:
            policy1 = self._to_lookup_policy(policy1.get_range_policy())
        return (KuhnTree([policy0.policy, policy1.policy]).dfs_ev()
                - KuhnTree([policy1.policy, policy0.policy]).dfs_ev()) / 2 * 100, 0
