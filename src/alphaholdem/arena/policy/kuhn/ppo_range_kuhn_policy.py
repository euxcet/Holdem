from __future__ import annotations

import torch
import numpy as np
from copy import deepcopy
from typing_extensions import override
from ..ppo_poker_policy import PPOPokerPolicy
from ....poker.component.card import Card
from ....poker.component.observation import Observation

class PPORangeKuhnPolicy(PPOPokerPolicy):

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
    def get_policies(self, env_obs_list: list[dict], game_obs_list: list[Observation]) -> np.ndarray:
        observations = []
        action_historys = []
        action_masks = []
        for env_obs in env_obs_list:
            observations.append(env_obs['observation'])
            action_historys.append(env_obs['action_history'])
            action_masks.append(env_obs['action_mask'])
        obs = {
            'obs': {
                'observation': torch.from_numpy(np.stack(observations)).to('cuda'),
                'action_history': torch.from_numpy(np.stack(action_historys)).to('cuda'),
                'action_mask': torch.from_numpy(np.stack(action_masks)).to('cuda'),
            }
        }
        return self.model(obs)[0].detach().cpu().numpy()

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
            result[i][0] = max(0, min(result[i][0], 1))
            strategy[i] = [result[i][0], 1.0 - result[i][0]]
        return strategy

    def log(self):
        result = self.get_range_policy()
        for i in range(result.shape[0]):
            for j in range(result[i].shape[0]):
                print(round(result[i][j], 3), end = ' ')
            print()

    @staticmethod
    def load_policy_from_run(run_folder: str) -> PPORangeKuhnPolicy:
        return PPORangeKuhnPolicy(model_path=PPOPokerPolicy._load_all_model_path(run_folder)[-1])

    @staticmethod
    def load_policies_from_run(run_folder: str) -> list[PPORangeKuhnPolicy]:
        policies = []
        for model_path in PPORangeKuhnPolicy._load_all_model_path(run_folder):
            policies.append(PPORangeKuhnPolicy(model_path=model_path))
        return policies