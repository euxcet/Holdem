import numpy as np
from typing_extensions import override
from .policy import Policy
from ...poker.component.observation import Observation

class CFRLeducPolicy(Policy):
    def __init__(self, p0_policy_path: str, p1_policy_path: str) -> None:
        self.policy = self._load_from_file(p0_policy_path)
        self.policy.update(self._load_from_file(p1_policy_path))

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
        history = ''.join(map(lambda x: 'J' if x[2] == 9 else ('Q' if x[2] == 10 else 'K'), np.argwhere(observation > 0.5))) + ':/'
        actions:list[list[int]] = np.argwhere(action_history > 0.05).tolist()
        actions.sort(key=lambda x: x[0] * 100 + (x[1] % 6) * 10 + x[1] // 6)
        street_action = 0
        for index, action in enumerate(actions):
            if index > 0 and action[0] != actions[index - 1][0]:
                history += '/'
                street_action = 0
            if action[2] == 3:
                history += 'r'
                street_action += 1
            else:
                history += 'c'
                street_action += 1
        if history[-1] == 'c' and street_action > 1:
            history += '/'
        history += ':'
        return history

    @override
    def sample_action(self, env_obs: dict, game_obs: Observation) -> int:
        policy = self.get_policy(env_obs)
        return np.random.choice(len(policy), p=policy)

    # Fold Check Call NONE Raise NONE NONE NONE
    # 0    1     2    3    4     5    6    7
    @override
    def get_policy(self, env_obs: dict, game_obs: Observation) -> list[float]:
        cfr_policy = self.policy[self._get_history(env_obs)]
        policy = [0.0] * 8
        # raise
        policy[4] = cfr_policy[0]
        # check or call
        policy[1 if env_obs['action_mask'][1] == 1 else 2] = cfr_policy[1]
        # fold
        policy[0] = cfr_policy[2]
        return policy

    @override
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> list[float]:
        ...