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
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> list[float]:
        ...

    @override
    def get_all_policy(self) -> np.ndarray:
        observations = [
            ('J', ''), ('Q', ''), ('K', ''),
            ('J', 'cr'), ('Q', 'cr'), ('K', 'cr'), 
            ('J', 'c'), ('Q', 'c'), ('K', 'c'),
            ('J', 'r'), ('Q', 'r'), ('K', 'r')
        ]
        result = self.get_policies([self._create_env_obs(obs[0], obs[1]) for obs in observations], None)
        strategy = np.zeros((12, 2)).astype(np.float32)
        for i in range(12):
            if observations[i][1].endswith('r'):
                strategy[i] = [result[i][0], 1.0 - result[i][0]]
            else:
                strategy[i] = [result[i][1], 1.0 - result[i][1]]
        return strategy

    @staticmethod
    def load_policy_from_run(run_folder: str) -> PPOKuhnPolicy:
        return PPOKuhnPolicy(model_path=PPOPokerPolicy._load_all_model_path(run_folder)[-1])

    @staticmethod
    def load_policies_from_run(run_folder: str) -> list[PPOKuhnPolicy]:
        policies = []
        for model_path in PPOKuhnPolicy._load_all_model_path(run_folder):
            policies.append(PPOKuhnPolicy(model_path=model_path))
        return policies