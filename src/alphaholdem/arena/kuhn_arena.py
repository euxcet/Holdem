from __future__ import annotations

from typing_extensions import override
import numpy as np

from alphaholdem.arena.policy.policy import Policy
from .arena import Arena
from .policy.kuhn.ppo_kuhn_policy import PPOKuhnPolicy
from .policy.kuhn.ppo_range_kuhn_policy import PPORangeKuhnPolicy
from .policy.kuhn.lookup_kuhn_policy import LookupKuhnPolicy
from .tree.kuhn_tree import KuhnTree
from .cfr.game import kuhn_rules
from .cfr.strategy import Strategy, StrategyProfile

class KuhnArena(Arena):
    def __init__(self, nash_path: str = 'strategy/kuhn.txt') -> None:
        super().__init__()
        self.nash = LookupKuhnPolicy(nash_path)
        self.rule = kuhn_rules()
        self.keys = [
            'J:/:', 'Q:/:', 'K:/:',
            'J:/cr:', 'Q:/cr:', 'K:/cr:',
            'J:/c:', 'Q:/c:', 'K:/c:',
            'J:/r:', 'Q:/r:', 'K:/r:'
        ]
        self.position = [
            (1, 0), (1, 0), (1, 0),
            (2, 1), (2, 1), (2, 1),
            (1, 0), (1, 0), (1, 0),
            (2, 1), (2, 1), (2, 1),
        ]

# check raise
# J:/: 0.764 0.236
# Q:/: 1.0 0.0
# K:/: 0.29 0.71

# J:/: 0.236 0.764 0.0
# Q:/: 0.0 1.0 0.0
# K:/: 0.71 0.29 0.0

# fold call
# J:/cr: 1.0 0.0
# Q:/cr: 0.43 0.57
# K:/cr: 0.0 1.0

# J:/cr: 0.0 0.0 1.0
# Q:/cr: 0.0 0.57 0.43
# K:/cr: 0.0 1.0 0.0

# ccall raise
# J:/c: 0.667 0.333
# Q:/c: 1.0 0.0
# K:/c: 0.0 1.0

# J:/c: 0.333 0.667 0.0
# Q:/c: 0.0 1.0 0.0
# K:/c: 1.0 0.0 0.0

# fold call
# J:/r: 0.0 0.01.0
# Q:/r: 0.0 0.333 0.667
# K:/r: 0.0 1.0 0.0



# raise ccall fold

    def _mapping(self, x: np.ndarray, position: tuple) -> np.ndarray:
        y = np.zeros(3)
        y[position[0]] = x[0]
        y[position[1]] = x[1]
        return y

    def _to_lookup_policy(self, strategy: np.ndarray) -> LookupKuhnPolicy:
        # Fold Check Call Raise
        policy = { self.keys[i]: self._mapping(strategy[i], self.position[i]) for i in range(len(self.keys)) }
        return LookupKuhnPolicy(policy = policy)

    @override
    @property
    def nash_policy(self) -> LookupKuhnPolicy:
        return self.nash

    @override
    def validate_policy(self, policy: Policy) -> None:
        assert type(policy) in [LookupKuhnPolicy, PPOKuhnPolicy, PPORangeKuhnPolicy]

    @override
    def get_exploitability(self, policy: Policy) -> int:
        self.validate_policy(policy)
        if type(policy) in [PPOKuhnPolicy, PPORangeKuhnPolicy]:
            policy = self._to_lookup_policy(policy.get_all_policy())
        else:
            return 0

    # @override
    # def policy_vs_policy(
    #     self,
    #     policy0: Policy,
    #     policy1: Policy,
    #     runs: int = 1024
    # ) -> tuple[float, float]:
    #     self.validate_policy(policy0)
    #     self.validate_policy(policy1)
    #     if type(policy0) in [PPOKuhnPolicy, PPORangeKuhnPolicy]:
    #         policy0 = self._to_lookup_policy(policy0.get_all_policy())
    #     if type(policy1) in [PPOKuhnPolicy, PPORangeKuhnPolicy]:
    #         policy1 = self._to_lookup_policy(policy1.get_all_policy())
    #     ev0 = KuhnTree([policy0.policy, policy1.policy]).dfs_ev() * 50
    #     ev1 = -KuhnTree([policy1.policy, policy0.policy]).dfs_ev() * 50
    #     return (ev0 + ev1) / 2, 0

    @override
    def policy_vs_policy(
        self,
        policy0: Policy,
        policy1: Policy,
        runs: int = 1024,
    ) -> tuple[float, float]:
        self.validate_policy(policy0)
        self.validate_policy(policy1)
        if type(policy0) in [PPOKuhnPolicy, PPORangeKuhnPolicy]:
            policy0 = self._to_lookup_policy(policy0.get_all_policy())
        if type(policy1) in [PPOKuhnPolicy, PPORangeKuhnPolicy]:
            policy1 = self._to_lookup_policy(policy1.get_all_policy())
        policy0 = policy0.to_strategy()[0]
        policy1 = policy1.to_strategy()[0]

        e0 = StrategyProfile(self.rule, [policy0, policy1]).expected_value()
        e1 = StrategyProfile(self.rule, [policy1, policy0]).expected_value()
        ev = (e0[0] + e1[1]) * 25

        p0 = StrategyProfile(self.rule, [policy0, policy0])
        p1 = StrategyProfile(self.rule, [policy1, policy1])

        br0 = p0.best_response()[1]
        exploitability0 = (br0[0] + br0[1]) / 2
        br1 = p1.best_response()[1]
        exploitability1 = (br1[0] + br1[1]) / 2

        return ev, (exploitability0, exploitability1)
