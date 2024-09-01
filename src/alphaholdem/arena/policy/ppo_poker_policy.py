from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing_extensions import override
from .policy import Policy
from ...poker.component.observation import Observation

class PPOPokerPolicy(Policy):
    def __init__(self, model_path: str = None, model: nn.Module = None, device: str = 'cuda') -> None:
        if model is not None:
            self.model = model
        else:
            self.model_path = model_path
            self.model = torch.load(model_path)
        self.model.to(device)
        self.model.eval()

    @override
    def sample_action(self, env_obs: dict, game_obs: Observation) -> int:
        return self.sample_actions([env_obs], [game_obs])[0]

    # Fold Check Call NONE Raise NONE NONE NONE
    # 0    1     2    3    4     5    6    7
    @override
    def sample_actions(self, env_obs_list: list[dict], game_obs_list: list[Observation]) -> list[int]:
        policies = self.get_policies(env_obs_list, game_obs_list)
        # print('ppo', policies)
        return [np.random.choice(len(policy), p=policy) for policy in policies]
    
    @override
    def get_policy(self, env_obs: dict, game_obs: Observation) -> list[float]:
        return self.get_policies([env_obs], [game_obs])[0]

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
        return F.softmax(self.model(obs)[0]).detach().cpu().numpy()

    @override
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> list[float]:
        ...

    @override
    def get_all_policy(self) -> np.ndarray:
        ...

    @staticmethod
    def _load_all_model_path(run_folder: str) -> list[str]:
        result = []
        for sub_dir in os.listdir(run_folder):
            if sub_dir.startswith('PPO'):
                for checkpoint_dir in os.listdir(os.path.join(run_folder, sub_dir)):
                    if checkpoint_dir.startswith('checkpoint'):
                        result.append(os.path.join(run_folder, sub_dir, checkpoint_dir, 'policies', 'learned', 'model.pt'))
        result.sort()
        return result
