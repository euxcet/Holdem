from __future__ import annotations

import numpy as np
from copy import deepcopy
from typing_extensions import override
from .ppo_poker_policy import PPOPokerPolicy
from ...poker.component.card import Card
from ...poker.component.observation import Observation

class PPOKuhnPolicy(PPOPokerPolicy):

    def _create_env_obs(self, card: str, action_history: str):
        # Fold Check Call Raise
        return {
            'observation': np.array([card == 'J', card == 'Q', card == 'K']).astype(np.float32),
            'action_history': {
                '': np.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
                'cr': np.array([[0, 1, 0, 0], [0, 0, 0, 1]]),
                'c': np.array([[0, 1, 0, 0], [0, 0, 0, 0]]),
                'r': np.array([[0, 0, 0, 1], [0, 0, 0, 0]]),
            }[action_history].astype(np.float32),
            'action_mask': {
                '': np.array([0, 1, 0, 1]),
                'cr': np.array([1, 0, 1, 0]),
                'c': np.array([0, 1, 0, 1]),
                'r': np.array([1, 0, 1, 0]),
            }[action_history].astype(np.float32),
        }

    @override
    def get_range_policy(self) -> np.ndarray:
        result = self.get_policies([
            self._create_env_obs('J', ''),
            self._create_env_obs('Q', ''),
            self._create_env_obs('K', ''),
            self._create_env_obs('J', 'cr'),
            self._create_env_obs('Q', 'cr'),
            self._create_env_obs('K', 'cr'),
            self._create_env_obs('J', 'c'),
            self._create_env_obs('Q', 'c'),
            self._create_env_obs('K', 'c'),
            self._create_env_obs('J', 'r'),
            self._create_env_obs('Q', 'r'),
            self._create_env_obs('K', 'r'),
        ], None)
        return result

    @staticmethod
    def load_policy_from_run(run_folder: str) -> PPOKuhnPolicy:
        return PPOKuhnPolicy(model_path=PPOPokerPolicy._load_all_model_path(run_folder)[-1])

    @staticmethod
    def load_policies_from_run(run_folder: str) -> list[PPOKuhnPolicy]:
        policies = []
        for model_path in PPOKuhnPolicy._load_all_model_path(run_folder):
            policies.append(PPOKuhnPolicy(model_path=model_path))
        return policies