import numpy as np
from .policy.ppo_kuhn_policy import PPOKuhnPolicy
from .policy.cfr_kuhn_policy import CFRKuhnPolicy
from ..poker.kuhn_poker_env import KuhnPokerEnv

class KuhnArena():
    def __init__(self, strategy_path: str = 'strategy/kuhn.txt') -> None:
        self.cfr = CFRKuhnPolicy(strategy_path)

    def cfr_self_play(
        self,
        cfr1: CFRKuhnPolicy = None,
        cfr2: CFRKuhnPolicy = None,
        runs: int = 1024,
    ) -> tuple[float, float]:
        env: KuhnPokerEnv = KuhnPokerEnv()
        result = []
        cfr1 = self.cfr if cfr1 is None else cfr1
        cfr2 = self.cfr if cfr2 is None else cfr2
        for run in range(runs):
            env.reset()
            while not env.is_over():
                env_obs = env.observe_current()
                game_obs = env.game.observe_current()
                if env.current_agent_id() == run % 2:
                    env.step(cfr1.sample_action(env_obs, game_obs))
                else:
                    env.step(cfr2.sample_action(env_obs, game_obs))
            result.append(env.agent_payoff()[run % 2] * env.payoff_max * 50)
        return np.mean(result), np.std(result)

    def _to_cfr_policy(self, strategy: np.ndarray) -> CFRKuhnPolicy:
        # Fold Check Call Raise
        node_name = ['J:', 'Q:', 'K:', 'J:cr', 'Q:cr', 'K:cr', 'J:c', 'Q:c', 'K:c', 'J:r', 'Q:r', 'K:r']
        policy = {}
        for i in range(strategy.shape[0]):
            if node_name[i].endswith('r'):
                policy[node_name[i]] = [strategy[i][0], 1.0 - strategy[i][0]] # fold or call
            else:
                policy[node_name[i]] = [strategy[i][1], 1.0 - strategy[i][1]] # check or raise
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