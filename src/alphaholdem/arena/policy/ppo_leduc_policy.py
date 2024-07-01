from __future__ import annotations

import numpy as np
from copy import deepcopy
from typing_extensions import override
from .ppo_poker_policy import PPOPokerPolicy
from ...poker.component.card import Card
from ...poker.component.observation import Observation

class PPOLeducPolicy(PPOPokerPolicy):
    # TODO: use model_action_8
    @override
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> np.ndarray:
        env_obs_list = []
        for hole_card in Card.from_str_list(['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']):
            obs = deepcopy(env_obs)
            obs['observation'][0] = np.zeros((4, 13))
            obs['observation'][0][hole_card.suit][hole_card.rank] = 1.0
            env_obs_list.append(obs)
        return np.around(np.array(self.get_policies(env_obs_list)), 2)

    @staticmethod
    def load_policy_from_run(run_folder: str) -> PPOLeducPolicy:
        return PPOLeducPolicy(model_path=PPOPokerPolicy._load_all_model_path(run_folder)[-1])

    @staticmethod
    def load_policies_from_run(run_folder: str) -> list[PPOLeducPolicy]:
        policies = []
        for model_path in PPOLeducPolicy._load_all_model_path(run_folder):
            policies.append(PPOLeducPolicy(model_path=model_path))
        return policies