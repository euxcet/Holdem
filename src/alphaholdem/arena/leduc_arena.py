import numpy as np
from typing_extensions import override
from .policy.policy import Policy
from .policy.leduc.ppo_leduc_policy import PPOLeducPolicy
from .policy.leduc.ppo_range_leduc_policy import PPORangeLeducPolicy
from .policy.leduc.lookup_leduc_policy import LookupLeducPolicy
from .tree.leduc_tree import LeducTree
from .cfr.game import leduc_rules
from .cfr.strategy import Strategy, StrategyProfile

class LeducArena():
    def __init__(self, nash_path: str = 'strategy/leduc.txt') -> None:
        self.nash = LookupLeducPolicy(nash_path)
        self.keys = sorted(self.nash.policy.keys())
        self.rule = leduc_rules()

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

    # return: ev, exploitability
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
        

    # @override
    # def policy_vs_policy(
    #     self,
    #     policy0: Policy,
    #     policy1: Policy,
    #     runs: int = 1024,
    # ) -> tuple[float, float]:
    #     self.validate_policy(policy0)
    #     self.validate_policy(policy1)
    #     if type(policy0) in [PPOLeducPolicy, PPORangeLeducPolicy]:
    #         policy0 = self._to_lookup_policy(policy0.get_all_policy(self.keys))
    #     if type(policy1) in [PPOLeducPolicy, PPORangeLeducPolicy]:
    #         policy1 = self._to_lookup_policy(policy1.get_all_policy(self.keys))
    #     ev0 = LeducTree([policy0.policy, policy1.policy]).dfs_ev() * 50
    #     ev1 = -LeducTree([policy1.policy, policy0.policy]).dfs_ev() * 50
    #     print(ev0, ev1)
    #     return (ev0 + ev1) / 2, 0

        #     x0, x1 = LookupLeducPolicy('./strategy/leduc_nash.txt').to_strategy()
        # y0, y1 = LookupLeducPolicy('./strategy/leduc_ppo.txt').to_strategy()

        # rules = leduc_rules()
        # e0 = StrategyProfile(rules, [x0, y1]).expected_value()
        # e1 = StrategyProfile(rules, [y0, x1]).expected_value()
        # ev = (e0[0] + e1[1]) * 25
        # print(ev * 25)

        # s0 = Strategy(0)
        # s1 = Strategy(1)
        # s0.load_from_file('strategy/leduc_ppo.txt')
        # s1.load_from_file('strategy/leduc_ppo.txt')
        # profile = StrategyProfile(rules, [s0, s1])
        # brev = profile.best_response()
        # print(brev[1])