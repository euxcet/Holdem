from __future__ import annotations

import numpy as np
from copy import deepcopy
from typing_extensions import override
from ..ppo_poker_policy import PPOPokerPolicy
from ....poker.component.card import Card
from ....poker.component.observation import Observation

class PPOHunlPolicy(PPOPokerPolicy):

    @override
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> np.ndarray:
        ...

    @override
    def get_all_policy(self) -> np.ndarray:
        ...

    @staticmethod
    def load_policy_from_run(run_folder: str) -> PPOHunlPolicy:
        return PPOHunlPolicy(model_path=PPOPokerPolicy._load_all_model_path(run_folder)[-1])

    @staticmethod
    def load_policies_from_run(run_folder: str) -> list[PPOHunlPolicy]:
        policies = []
        for model_path in PPOHunlPolicy._load_all_model_path(run_folder):
            policies.append(PPOHunlPolicy(model_path=model_path))
        return policies