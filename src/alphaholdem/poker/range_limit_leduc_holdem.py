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
from .component.range_leduc_judger import RangeLeducJudger

class RangeLimitLeducHoldem(RangePokerGame):
    def __init__(
        self,
        num_players: int,
        agents: list[Agent] = None,
        initial_chips: list[int] = None,
        verbose: bool = False,
        num_runs: int = 10,
        showdown_street: Street = Street.Turn,
        custom_board_cards: list[Card] = None,
        custom_player_hole_cards: list[list[Card]] = None,
        num_street_board_cards: list[int] = [1, 0, 0],
    ) -> None:
        super().__init__(
            num_players=num_players,
            agents=agents,
            initial_chips=initial_chips,
            initial_deck=Deck.deck_leduc(),
            blinds=None,
            antes=[1, 1],
            num_hole_cards=1,
            verbose=verbose,
            num_runs=num_runs,
            raise_pot_size=[],
            showdown_street=showdown_street,
            custom_board_cards=custom_board_cards,
            custom_player_hole_cards=custom_player_hole_cards,
            num_street_board_cards=num_street_board_cards,
            action_shape=4,
        )
        self.judger = RangeLeducJudger()
    
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

    def _refresh_board(self):
        if self.street == Street.Preflop or self.street == Street.Showdown:
            return
        # self.factor *= 3
        need_to_deal = self.num_street_board_cards[self.street.value - 1]
        already_dealed = sum(self.num_street_board_cards[:self.street.value - 1])
        if self.custom_board_cards is not None and len(self.custom_board_cards) >= already_dealed + need_to_deal:
            self.board_cards.extend(self.custom_board_cards[already_dealed:already_dealed + need_to_deal])
        else:
            self.board_cards.extend(self.dealer.deal(need_to_deal))

    def step(self, action: np.ndarray) -> Observation:
        action = action.clip(0, 1)
        action_prob = action.copy()
        # for i in range(12):
        #     action_prob[i] += 1e-5
        legal_actions = list(self.get_legal_action(self.current_player))

        action_mask = np.array([
            0 if legal_actions[i] is None else 1
            for i in range(4)
        ]).astype(np.float32)
        for i in range(3):
            if sum(action_prob[i * 4 : (i + 1) * 4] * action_mask) < 1e-5:
                action_prob[i * 4 : (i + 1) * 4] = action_mask
            s = sum(action_prob[i * 4 : (i + 1) * 4] * action_mask)
            action_prob[i * 4 : (i + 1) * 4] *= action_mask / s
        # print(action_prob)

        # prob = np.zeros(4)
        # for i in range(12):
        #     prob[i % 4] += self.player_range[self.current_player][i // 4] * action_prob[i]
        # prob /= np.sum(prob)
        # sample = np.random.choice(self.action_shape, p=prob)

        num_actions = 0
        for i in range(4):
            if legal_actions[i] is not None:
                num_actions += 1
            # elif i != 0:
            #     for j in range(3):
            #         action_prob[j * 4] += action_prob[j * 4 + i]
            #         action_prob[j * 4 + i] = 0
        self.factor *= num_actions

        sample = np.random.randint(0, num_actions)
        # if self.current_player == 0:
        #     sample = num_actions - 1
        # else:
        #     sample = 0

        s = [sum(action_prob[0:4]), sum(action_prob[4:8]), sum(action_prob[8:12])]

        for i in range(4):
            if legal_actions[i] is not None:
                if sample == 0:
                    for j in range(3):
                        self.player_range[self.current_player][j] *= action_prob[j * 4 + i] / s[j]
                    return super().step(legal_actions[i])
                else:
                    sample -= 1

    def _calculate_payoff(self) -> list[float]:
        return self.judger.judge(
            pot = self.pot,
            board_cards = self.board_cards,
            player_fold = self.player_fold,
            player_bet = self.player_bet,
            player_range = self.player_range,
            factor = self.factor,
        )
    
    def get_legal_action(self, player: int) -> list[Action]:
        legal_actions = [
            self.create_action(ActionType.Fold) if self._is_fold_legal() else None,
            self.create_action(ActionType.Check) if self._is_check_legal() else None,
            self.create_action(ActionType.Call) if self._is_call_legal() else None,
        ]
        base_factor = 2 if self.street == Street.Preflop else 4
        street_raises = sum(self.player_street_num_raises)
        if street_raises < 2:
            raise_chip, raise_pot = self._calculate_raise_chip(raise_chip=base_factor * (street_raises + 1))
            legal_actions.append(self.create_action(
                ActionType.Raise,
                raise_chip = raise_chip,
                raise_pot = raise_pot,
                raise_to = raise_chip + self.player_street_bet[player],
            ))
        else:
            legal_actions.append(None)
        return legal_actions