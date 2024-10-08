import numpy as np
from copy import deepcopy
from numpy.random import Generator
from .range_poker_game import RangePokerGame
from .agent.agent import Agent
from .component.action import Action, ActionType
from .component.card import Card
from .component.deck import Deck
from .component.street import Street
from .component.observation import Observation
from .component.range_kuhn_judger import RangeKuhnJudger

class RangeKuhnPoker(RangePokerGame):
    def __init__(
        self,
        num_players: int = 2,
        agents: list[Agent] = None,
        verbose: bool = False,
        showdown_street: Street = Street.Flop,
        custom_board_cards: list[Card] = None,
        custom_player_hole_cards: list[list[Card]] = None,
        num_street_board_cards: list[int] = [0, 0, 0],
    ) -> None:
        super().__init__(
            num_players=num_players,
            agents=agents,
            initial_chips=4, # players only need 2 chips, the 4 chips are to avoid an all in.
            initial_deck=Deck.deck_kuhn(),
            blinds=None,
            antes=[1, 1],
            num_hole_cards=1,
            verbose=verbose,
            num_runs=1,
            raise_pot_size=[],
            showdown_street=showdown_street,
            custom_board_cards=custom_board_cards,
            custom_player_hole_cards=custom_player_hole_cards,
            num_street_board_cards=num_street_board_cards,
            action_shape=4,
        )
        self.judger = RangeKuhnJudger()

    def reset(self, seed: int = None, rng: Generator = None) -> Observation:
        self.player_range = [[1.0, 1.0, 1.0] for i in range(self.num_players)]
        self.factor = 1.0
        return super().reset(seed, rng)

    def observe(self, player: int) -> Observation:
        return Observation(
            player = player,
            num_players = self.num_players,
            hole_cards = self.player_hole_cards[player],
            chips = self.player_chips[player],
            current_player = self.current_player,
            player_chips = self.player_chips,
            player_street_bet = self.player_street_bet,
            player_payoff = self.player_payoff,
            pot = self.pot,
            street = self.street,
            board_cards = self.board_cards,
            legal_actions = self.get_legal_action(player),
            log_action = deepcopy(self.log_action),
            is_over = self.is_over(),
            player_range=self.player_range,
        )

    def _calculate_payoff(self) -> list[float]:
        return self.judger.judge(
            pot = self.pot,
            board_cards = self.board_cards,
            player_fold = self.player_fold,
            player_bet = self.player_bet,
            player_range = self.player_range,
            factor = self.factor,
        )

    def step(self, action: np.ndarray) -> Observation:
        action = action.clip(0, 1)
        legal_actions = self.get_legal_action(self.current_player)
        legal_actions = list(filter(lambda x: x is not None, self.get_legal_action(self.current_player)))

        player_range = self.player_range[self.current_player]
        # prob = (action[0] * player_range[0] + action[1] * player_range[1] + action[2] * player_range[2]) / sum(player_range)
        # prob = (action[0] + action[1] + action[2]) / 3

        # print(action)
        # for i in range(3):
        #     self.player_range[self.current_player][i] *= action[i]
        # return super().step(legal_actions[0])
        prob = 0.5
        self.factor *= 2
        sample = 0 if np.random.random() < prob else 1
        for i in range(3):
            self.player_range[self.current_player][i] *= action[i] if sample == 0 else (1 - action[i])
        return super().step(legal_actions[sample])
    
    def get_legal_action(self, player: int) -> list[Action]:
        legal_actions = [
            self.create_action(ActionType.Fold) if self._is_fold_legal() else None,
            self.create_action(ActionType.Check) if self._is_check_legal() else None,
            self.create_action(ActionType.Call) if self._is_call_legal() else None,
        ]
        if sum(self.player_street_num_raises) < 1:
            raise_chip, raise_pot = self._calculate_raise_chip(raise_chip=1)
            legal_actions.append(self.create_action(
                ActionType.Raise,
                raise_chip = raise_chip,
                raise_pot = raise_pot,
                raise_to = raise_chip + self.player_street_bet[player],
            ))
        else:
            legal_actions.append(None)
        return legal_actions