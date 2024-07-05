import numpy as np
from .policy.ppo_poker_policy import PPOPokerPolicy
from .policy.cfr_kuhn_policy import CFRKuhnPolicy
from .envs.limit_leduc_holdem_env import create_limit_holdem_env
from ..poker.kuhn_poker_env import KuhnPokerEnv

class KuhnArena():
    def __init__(self, strategy_path: str = 'strategy/kuhn.txt') -> None:
        self.cfr = CFRKuhnPolicy(strategy_path)

    def cfr_self_play(
        self,
        runs: int = 1024,
    ) -> tuple[float, float]:
        env: KuhnPokerEnv = KuhnPokerEnv()
        result = []
        for run in range(runs):
            env.reset()
            while not env.is_over():
                env_obs = env.observe_current()
                game_obs = env.game.observe_current()
                env.step(self.cfr.sample_action(env_obs, game_obs))
            result.append(env.agent_payoff()[run % 2] * env.payoff_max * 50)
        return np.mean(result), np.std(result)

    def ppo_vs_cfr(
        self,
        ppo: PPOPokerPolicy,
        runs: int = 1024,
        batch_size: int = 32,
    ) -> tuple[float, float]:
        envs: list[KuhnPokerEnv] = [create_limit_holdem_env() for _ in range(batch_size)]
        for env in envs:
            env.reset()
        finished = 0
        result = []
        while finished < runs:
            # cfr
            for env_id, env in enumerate(envs):
                while env.current_agent_id() != env_id % 2:
                    env.step(self.cfr.sample_action(env.observe_current(), env.game.observe_current()))
                    if env.is_over(): 
                        finished += 1
                        result.append(env.agent_payoff()[env_id % 2] * env.payoff_max * 50)
                        env.reset()
            # ppo
            ppo_actions = ppo.sample_actions(
                env_obs_list=list(map(lambda env: env.observe_current(), envs)),
                game_obs_list=list(map(lambda env: env.game.observe_current(), envs)),
            )
            for env_id, env in enumerate(envs):
                env.step(ppo_actions[env_id])
                if env.is_over():
                    finished += 1
                    result.append(env.agent_payoff()[env_id % 2] * env.payoff_max * 50)
                    env.reset()
        return np.mean(result), np.std(result)

    def ppo_vs_ppo(
        self,
        ppo0: PPOPokerPolicy,
        ppo1: PPOPokerPolicy,
        runs: int = 1024,
        batch_size: int = 32,
    ) -> tuple[float, float]:
        # TODO
        ...

    def ppos_melee(
        self,
        ppos: list[PPOPokerPolicy],
        runs: int = 1024,
        batch_size: int = 32,
    ) -> list[float]:
        ...