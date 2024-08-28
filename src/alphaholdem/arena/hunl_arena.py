import numpy as np
from .policy.policy import Policy
from typing_extensions import override
from .envs.no_limit_texas_holdem_env import create_no_limit_holdem_env
from ..poker.no_limit_texas_holdem_env import NoLimitTexasHoldemEnv
from .policy.ppo_poker_policy import PPOPokerPolicy
from .policy.hunl.deepstack_hunl_policy import DeepstackHunlPolicy

# TODO: refactor
class HunlArena():
    def __init__(self, model_path: str = './checkpoint/supervise/best.pt') -> None:
        self.nash = DeepstackHunlPolicy(model_path=model_path)

    @override
    @property
    def nash_policy(self) -> Policy:
        return self.nash

    @override
    def validate_policy(self, policy: Policy) -> None:
        ...

    @override
    def policy_vs_policy(
        self,
        policy0: Policy,
        policy1: Policy,
        runs: int = 1024,
    ) -> tuple[float, float]:
        env: NoLimitTexasHoldemEnv = create_no_limit_holdem_env()
        result = []
        for run in range(runs):
            env.reset()
            while not env.is_over():
                env_obs = env.observe_current()
                game_obs = env.game.observe_current()
                if env.current_agent_id() == run % 2:
                    env.step(policy0.sample_action(env_obs, game_obs))
                else:
                    env.step(policy1.sample_action(env_obs, game_obs))
            result.append(env.agent_payoff()[run % 2] * env.payoff_max * 50)
        return np.mean(result), np.std(result)

    def ppo_vs_tf(
        self,
        ppo: PPOPokerPolicy,
        runs: int = 1024,
    ) -> tuple[float, float]:
        env: NoLimitTexasHoldemEnv = create_no_limit_holdem_env()
        result = []
        for run in range(runs):
            env.reset()
            while not env.is_over():
                env_obs = env.observe_current()
                game_obs = env.game.observe_current()
                if env.current_agent_id() == run % 2:
                    env.step(ppo.sample_action(env_obs, game_obs))
                else:
                    env.step(self.tf.sample_action(env_obs, game_obs))
            result.append(env.agent_payoff()[run % 2] * env.payoff_max * 50)
        return np.mean(result), np.std(result)