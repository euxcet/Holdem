from __future__ import annotations

import numpy as np
from copy import deepcopy
from typing_extensions import override
from ..ppo_poker_policy import PPOPokerPolicy
from ....poker.component.card import Card
from ....poker.component.observation import Observation

class PPOTexasPolicy(PPOPokerPolicy):

    @override
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> np.ndarray:
        ...

    @override
    def get_all_policy(self) -> np.ndarray:
        ...
