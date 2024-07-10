import numpy as np
from typing_extensions import override
from .policy import Policy
from ...poker.component.observation import Observation

class CFRKuhnPolicy(Policy):
    def __init__(self, strategy_path: str = None, policy: dict = None) -> None:
        if policy is not None:
            self.strategy_path = None
            self.policy = policy
        else:
            self.strategy_path = strategy_path
            self.policy = self._load_from_file(strategy_path)

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
        return np.random.choice(len(policy), p=policy)

    # Fold Check Call Raise
    # 0    1     2    3
    @override
    def get_policy(self, env_obs: dict, game_obs: Observation) -> list[float]:
        cfr_policy = self.policy[self._get_history(env_obs)]
        policy = [0.0] * 4
        # 0 check or fold
        # 1 call or raise
        policy[0 if env_obs['action_mask'][0] == 1 else 1] = cfr_policy[0]
        policy[2 if env_obs['action_mask'][2] == 1 else 3] = cfr_policy[1]
        return policy

    @override
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> list[float]:
        ...