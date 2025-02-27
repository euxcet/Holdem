import numpy as np
from typing_extensions import override
from ..policy import Policy
from ...cfr.strategy import Strategy
from ....poker.component.observation import Observation

class LookupKuhnPolicy(Policy):
    def __init__(self, strategy_path: str | None = None, policy: dict | None = None) -> None:
        if policy is not None:
            self.strategy_path = None
            self.policy = policy
        else:
            self.strategy_path = strategy_path
            self.policy = self._load_from_file(strategy_path)

    def _reverse(self, policy: dict[str, list[float]]) -> dict[str, list[float]]:
        return {x[0]: list(reversed(x[1])) for x in policy.items()}

    def to_strategy(self) -> tuple[Strategy, Strategy]:
        s0 = Strategy(0)
        s0.load(self._reverse(self.policy))
        s1 = Strategy(1)
        s1.load(self._reverse(self.policy))
        return (s0, s1)

    def _load_from_file(self, path: str) -> dict[str, list[float]]:
        result = dict()
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                s = line.strip().split(' ')
                result[s[0]] = list(map(float, s[1:]))
        return result

    def _get_history(self, env_obs: dict) -> str:
        observation = env_obs['observation']
        action_history = env_obs['action_history']
        history = ('J' if observation[0] > 0.5 else ('Q' if observation[1] > 0.5 else 'K')) + ':'
        for i in range(action_history.shape[0]):
            if action_history[i][3] > 0.05:
                history += 'r'
            elif action_history[i].any():
                history += 'c'
        return history

    @override
    def sample_action(self, env_obs: dict, game_obs: Observation) -> int:
        policy = self.get_policy(env_obs, game_obs)
        return np.random.choice(len(policy), p=policy / sum(policy))

    # Fold Check Call Raise
    # 0    1     2    3
    @override
    def get_policy(self, env_obs: dict, game_obs: Observation) -> np.ndarray:
        cfr_policy = self.policy[self._get_history(env_obs)]
        policy = np.zeros(4)
        # raise
        policy[3] = cfr_policy[0]
        # check or call
        policy[1 if env_obs['action_mask'][1] == 1 else 2] = cfr_policy[1]
        # fold
        policy[0] = cfr_policy[2]
        return policy / sum(policy)

    @override
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> list[float]:
        ...

    @override
    def get_all_policy(self) -> np.ndarray:
        ...