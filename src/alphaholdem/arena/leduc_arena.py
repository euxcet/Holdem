import numpy as np
from typing_extensions import override
from .policy.policy import Policy
from .policy.leduc.ppo_leduc_policy import PPOLeducPolicy
from .policy.leduc.ppo_range_leduc_policy import PPORangeLeducPolicy
from .policy.leduc.lookup_leduc_policy import LookupLeducPolicy
from .tree.leduc_tree import LeducTree

class LeducArena():
    def __init__(self, nash_path: str = 'strategy/leduc.txt') -> None:
        self.nash = LookupLeducPolicy(nash_path)
        self.keys = sorted(self.nash.policy.keys())

    def _to_lookup_policy(self, strategy: np.ndarray) -> LookupLeducPolicy:
        # Fold Check Call Raise
        policy = { self.keys[i]: strategy[i] for i in range(len(self.keys)) }
        return LookupLeducPolicy(policy = policy)

    @property
    @override
    def nash_policy(self) -> Policy:
        return self.nash

    @override
    def validate_policy(self, policy: Policy) -> None:
        assert type(policy) in [LookupLeducPolicy, PPOLeducPolicy, PPORangeLeducPolicy]

    @override
    def policy_vs_policy(
        self,
        policy0: Policy,
        policy1: Policy,
        runs: int = 1024,
    ) -> tuple[float, float]:
        self.validate_policy(policy0)
        self.validate_policy(policy1)
        if type(policy0) in [PPOLeducPolicy, PPORangeLeducPolicy]:
            policy0 = self._to_lookup_policy(policy0.get_all_policy(self.keys))
        if type(policy1) in [PPOLeducPolicy, PPORangeLeducPolicy]:
            policy1 = self._to_lookup_policy(policy1.get_all_policy(self.keys))
        ev0 = LeducTree([policy0.policy, policy1.policy]).dfs_ev() * 50
        ev1 = -LeducTree([policy1.policy, policy0.policy]).dfs_ev() * 50
        print(ev0, ev1)
        return (ev0 + ev1) / 2, 0
