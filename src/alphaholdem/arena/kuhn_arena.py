from __future__ import annotations

from typing_extensions import override
import numpy as np

from alphaholdem.arena.policy.policy import Policy
from .arena import Arena
from .policy.kuhn.ppo_kuhn_policy import PPOKuhnPolicy
from .policy.kuhn.ppo_range_kuhn_policy import PPORangeKuhnPolicy
from .policy.kuhn.lookup_kuhn_policy import LookupKuhnPolicy
from .tree.kuhn_tree import KuhnTree

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
            policy0 = self._to_lookup_policy(policy0.get_all_policy())
        if type(policy1) in [PPOKuhnPolicy, PPORangeKuhnPolicy]:
            policy1 = self._to_lookup_policy(policy1.get_all_policy())
        ev0 = KuhnTree([policy0.policy, policy1.policy]).dfs_ev() * 50
        ev1 = -KuhnTree([policy1.policy, policy0.policy]).dfs_ev() * 50
        print(ev0, ev1)
        return (ev0 + ev1) / 2, 0
