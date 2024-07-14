from __future__ import annotations

import math
import numpy as np
from numpy.random import Generator
from enum import Enum
from copy import deepcopy
from .component.action import Action, ActionType
from .component.card import Card
from .component.dealer import Dealer
from .component.deck import Deck
from .component.judger import Judger
from .component.observation import Observation
from .component.street import Street
from .component.player_name import get_players_name
from .agent.agent import Agent
from .agent.random_agent import RandomAgent
from ..utils.logger import log

from .poker_game import PokerGame

class RangePokerGame(PokerGame):

    DEFAULT_INITIAL_CHIPS = 200 # 200SB = 100BB
    DEFAULT_DECK_SIZE = 52

    def __init__(
        self,
        num_players: int,
        agents: list[Agent] = None,
        initial_chips: list[int] | int = None,
        initial_deck: Deck = None,
        num_hole_cards: int = 2,
        blinds: list[int] = [1, 2],
        antes: list[int] = None,
        legal_action_type: list[bool] = ActionType.default_legal(),
        verbose: bool = False,
        num_runs: int = 10,
        max_num_actions_street: int = 6,
        max_num_raises_street: int = 6,
        custom_board_cards: list[Card] = None,
        custom_player_hole_cards: list[list[Card]] = None,
        raise_pot_size: list[float] = [0.5, 0.75, 1, 1.5, 2],
        legal_raise_pot_size: list[float] = [0.5, 0.75, 1, 1.5, 2],
        showdown_street: Street = Street.Showdown,
        num_street_board_cards: list[int] = [3, 1, 1],
        action_shape: int = None,
    ) -> None:
        super().__init__(
            num_players=num_players,
            agents=agents,
            initial_chips=initial_chips,
            initial_deck=initial_deck,
            num_hole_cards=num_hole_cards,
            blinds=blinds,
            antes=antes,
            legal_action_type=legal_action_type,
            verbose=verbose,
            num_runs=num_runs,
            max_num_actions_street=max_num_actions_street,
            max_num_raises_street=max_num_raises_street,
            custom_board_cards=custom_board_cards,
            custom_player_hole_cards=custom_player_hole_cards,
            raise_pot_size=raise_pot_size,
            legal_raise_pot_size=legal_raise_pot_size,
            showdown_street=showdown_street,
            num_street_board_cards=num_street_board_cards,
            action_shape=action_shape,
        )