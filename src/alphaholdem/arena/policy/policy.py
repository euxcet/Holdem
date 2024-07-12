import numpy as np
from abc import ABC, abstractmethod
from ...poker.component.observation import Observation

class Policy(ABC):
    @abstractmethod
    def sample_action(self, env_obs: dict, game_obs: Observation) -> int:
        ...

    def sample_actions(self, env_obs_list: list[dict], game_obs_list: list[Observation]) -> list[int]:
        actions = []
        for env_obs, game_obs in zip(env_obs_list, game_obs_list):
            actions.append(self.sample_action(env_obs, game_obs_list))
        return actions

    # TODO: use np.ndarray instead of list[float]?
    @abstractmethod
    def get_policy(self, env_obs: dict, game_obs: Observation) -> np.ndarray:
        ...

    def get_policies(self, env_obs_list: list[dict], game_obs_list: list[Observation]) -> list[list[float]]:
        policies = []
        for env_obs, game_obs in zip(env_obs_list, game_obs_list):
            policies.append(self.get_policy(env_obs, game_obs_list))
        return policies

    @abstractmethod
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> list[float]:
        ...

    def log(self) -> None:
        ...