import numpy as np
from typing_extensions import override
from .policy.policy import Policy
from .policy.ppo_leduc_policy import PPOLeducPolicy
from .policy.lookup_leduc_policy import LookupLeducPolicy
from .envs.limit_leduc_holdem_env import create_limit_holdem_env
from .tree.leduc_tree import LeducTree
from ..poker.limit_leduc_holdem_env import LimitLeducHoldemEnv

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
        assert type(policy) in [LookupLeducPolicy, PPOLeducPolicy]

    @override
    def policy_vs_policy(
        self,
        policy0: Policy,
        policy1: Policy,
        runs: int = 1024,
    ) -> tuple[float, float]:
        self.validate_policy(policy0)
        self.validate_policy(policy1)
        if type(policy0) in [PPOLeducPolicy]:
            policy0 = self._to_lookup_policy(policy0.get_all_policy(self.keys))
        if type(policy1) in [PPOLeducPolicy]:
            policy1 = self._to_lookup_policy(policy1.get_all_policy(self.keys))
        return (LeducTree([policy0.policy, policy1.policy]).dfs_ev()
                - LeducTree([policy1.policy, policy0.policy]).dfs_ev()) / 2 * 100, 0

    # def cfr_self_play(
    #     self,
    #     policy0: LookupLeducPolicy,
    #     policy1: LookupLeducPolicy,
    #     runs: int = 1024,
    # ) -> tuple[float, float]:
    #     return (LeducTree([policy0.policy, policy1.policy]).dfs_ev()
    #             - LeducTree([policy1.policy, policy0.policy]).dfs_ev()) / 2 * 100, 0

    # def ppo_vs_cfr(
    #     self,
    #     policy0: PPOLeducPolicy,
    #     policy1: LookupLeducPolicy,
    #     runs: int = 1024,
    #     batch_size: int = 32,
    # ) -> tuple[float, float]:
    #     envs: list[LimitLeducHoldemEnv] = [create_limit_holdem_env() for _ in range(batch_size)]
    #     for env in envs:
    #         env.reset()
    #     finished = 0
    #     result = []
    #     while finished < runs:
    #         # cfr
    #         for env_id, env in enumerate(envs):
    #             while env.current_agent_id() != env_id % 2:
    #                 env.step(policy1.sample_action(env.observe_current(), env.game.observe_current()))
    #                 if env.is_over(): 
    #                     finished += 1
    #                     result.append(env.agent_payoff()[env_id % 2] * env.payoff_max * 50)
    #                     env.reset()
    #         # ppo
    #         ppo_actions = policy0.sample_actions(
    #             env_obs_list=list(map(lambda env: env.observe_current(), envs)),
    #             game_obs_list=list(map(lambda env: env.game.observe_current(), envs)),
    #         )
    #         for env_id, env in enumerate(envs):
    #             env.step(ppo_actions[env_id])
    #             if env.is_over():
    #                 finished += 1
    #                 result.append(env.agent_payoff()[env_id % 2] * env.payoff_max * 50)
    #                 env.reset()
    #     return np.mean(result), np.std(result)