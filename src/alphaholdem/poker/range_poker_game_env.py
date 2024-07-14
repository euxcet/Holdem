from __future__ import annotations

import random
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from .component.action import Action
from .component.observation import Observation
from .range_poker_game import RangePokerGame
from .poker_game_env import PokerGameEnv

class RangePokerGameEnv(PokerGameEnv):
    def __init__(
        self,
        num_players: int,
        game: RangePokerGame,
        circular_train: bool = False,
        payoff_max: float = 200,
    ) -> None:
        super().__init__(
            num_players=num_players,
            game=game,
            circular_train=circular_train,
            payoff_max=payoff_max,
        )