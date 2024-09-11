import random
import numpy as np
from typing_extensions import override
from ...poker.component.observation import Observation
from .policy import Policy

class RandomPolicy(Policy):
    def __init__(self) -> None:
        ...

    @override
    def sample_action(self, env_obs: dict, game_obs: Observation) -> int:
        actions = []
        for id, action in enumerate(game_obs.legal_actions):
            if action is not None:
                actions.append(id)
        return random.choice(actions)

    @override
    def get_policy(self, env_obs: dict, game_obs: Observation) -> np.ndarray:
        ...

    @override
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> list[float]:
        ...

    @override
    def get_all_policy(self) -> np.ndarray:
        ...

    def log(self) -> None:
        ...