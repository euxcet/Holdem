from __future__ import annotations

import torch
import numpy as np
from copy import deepcopy
from typing_extensions import override
from ..policy import Policy
from ....poker.component.card import Card
from ....poker.component.observation import Observation
from ....model.hunl_supervise_one_hot import HUNLSuperviseOneHot, HUNLSuperviseOneHot50
from ....model.hunl_supervise_model import HUNLSuperviseModel
from ....model.hunl_supervise_resnet import HUNLSuperviseResnet

class DeepstackHunlPolicy(Policy):
    def __init__(self, model_path: str = None, device: str = 'cuda') -> None:
        self.model = HUNLSuperviseResnet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to('cuda')
        self.model.eval()

    @override
    def sample_action(self, env_obs: dict, game_obs: Observation) -> int:
        policy = self.get_policy(env_obs, game_obs)
        policy /= policy.sum()
        return np.random.choice(len(policy), p=policy)

    @override
    def get_policy(self, env_obs: dict, game_obs: Observation) -> np.ndarray:
        cards = torch.from_numpy(env_obs['observation'][None, :]).to('cuda')
        actions = torch.from_numpy(env_obs['action_history'][None, :]).to('cuda')
        policy: np.ndarray = self.model(cards, actions).detach().cpu().numpy()
        policy = policy.clip(0, 1)
        policy /= policy.sum()
        # fold check/call raise all_in
        # ->
        # fold check call all_in raise
        if env_obs['action_mask'][1] > 0.5:
            strategy = np.array([policy[0][0], policy[0][1], 0, policy[0][3], policy[0][2]])
        else:
            strategy = np.array([policy[0][0], 0, policy[0][1], policy[0][3], policy[0][2]])
        index = np.argmax(strategy)
        strategy = np.array([(1 if i == index else 0) for i in range(5)]).astype(np.float32)
        return strategy


    @override
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> list[float]:
        ...

    @override
    def get_all_policy(self) -> np.ndarray:
        ...
    