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

class RangePokerGame():

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
        self.verbose = verbose
        self.num_players = num_players
        self.num_hole_cards = num_hole_cards
        self.num_blinds = 0 if blinds is None else len(blinds)
        self.blinds = blinds
        self.antes = antes
        self.num_runs = num_runs
        self.player_name = get_players_name(num_players=num_players)
        self.agents = [RandomAgent() for _ in range(num_players)] if agents is None else agents
        self.legal_action_type = legal_action_type
        self.dealer = Dealer(Deck.deck_52() if initial_deck is None else initial_deck)
        self.judger = Judger()
        self.showdown_street = showdown_street
        self.max_num_actions_street = max_num_actions_street
        self.max_num_raises_street = max_num_raises_street
        self.custom_board_cards = custom_board_cards
        self.custom_player_hole_cards = custom_player_hole_cards
        self.raise_pot_size = raise_pot_size
        self.legal_raise_pot_size = legal_raise_pot_size
        self.action_shape = 4 + len(self.raise_pot_size) if action_shape is None else action_shape
        self.num_street_board_cards = num_street_board_cards
        self.num_board_cards = sum(self.num_street_board_cards)
        if initial_chips is None:
            self.initial_chips = [self.DEFAULT_INITIAL_CHIPS] * num_players
        elif type(initial_chips) is int:
            self.initial_chips = [initial_chips] * num_players
        else:
            self.initial_chips = initial_chips
        self.reset()

    def reset(self, seed: int = None, rng: Generator = None) -> Observation:
        # Use different random number generators for the game logic and agents
        self.seed = seed
        if self.verbose:
            log.info(f'Seed: {self.seed}')
        if rng is not None:
            self.game_rng = deepcopy(rng)
        else:
            self.game_rng = np.random.default_rng(self.seed)
        # Used to create an identical game
        self.game_rng_bak = deepcopy(self.game_rng)
        for i in range(len(self.agents)):
            self.agents[i].set_rng(np.random.default_rng(self.seed if self.seed is None else self.seed + i + 1))
        self.dealer.reset(self.game_rng)
        self.dealer.shuffle()
        self.street = Street.Preflop
        self.player_hole_cards = []
        for i in range(self.num_players):
            if self.custom_player_hole_cards is None or \
               self.custom_player_hole_cards[i] is None or \
               len(self.custom_player_hole_cards[i]) != self.num_hole_cards:
                self.player_hole_cards.append(self.dealer.deal(self.num_hole_cards))
            else:
                self.player_hole_cards.append(self.custom_player_hole_cards[i])
                self.dealer.burn(self.custom_player_hole_cards[i])
        # self.player_hole_cards = [self.dealer.deal(self.num_hole_cards) for _ in range(self.num_players)]
        # Player's remaining chips
        self.player_chips = deepcopy(self.initial_chips)
        # Can the player raise
        self.player_can_raise = [True] * self.num_players
        # If the player has folded
        self.player_fold = [False] * self.num_players
        # If the player all in
        self.player_all_in = [False] * self.num_players
        # The amount of chips the player bet in this game
        self.player_bet = [0] * self.num_players
        # The amount of chips the player bet on this street
        self.player_street_bet = [0] * self.num_players
        # If the player complete an action on this street
        self.player_street_act = [False] * self.num_players
        self.player_street_num_actions = [0] * self.num_players
        self.player_street_num_raises = [0] * self.num_players
        self.player_payoff = [0] * self.num_players
        self.board_cards = []
        self.pot = 0
        # The current maximum bet size for players on this street
        self.street_raise = 0
        # Minimum chip difference for the next raise on this street
        self.street_raise_delta = 0
        self.log_action = []

        if self.antes is not None:
            for player, ante in enumerate(self.antes):
                self.pot += ante
                self.player_bet[player] += ante
                self.player_chips[player] -= ante
        if self.blinds is not None:
            for player, blind in enumerate(self.blinds):
                self.pot += blind
                self.player_chips[player] -= blind
                self.player_bet[player] = blind
                self.player_street_bet[i] = blind
                self.street_raise = blind
                self.street_raise_delta = blind

        self.current_player = self.num_blinds % self.num_players
        return self.observe(self.current_player)

    def _next_unfinished_player(self, player: int) -> int:
        player = player % self.num_players
        initial_player = player
        while self._is_player_finished(player):
            player = (player + 1) % self.num_players
            if player == initial_player:
                return player
        return player
    
    def _is_player_finished(self, player: int) -> bool:
        return self.player_fold[player] or self.player_all_in[player]

    def _is_street_end(self) -> bool:
        if self.num_folded_players == self.num_players - 1:
            return True
        street_bet = 0
        for player in range(self.num_players):
            street_bet = max(street_bet, self.player_street_bet[player])
        for player in range(self.num_players):
            if not self._is_player_finished(player) and not self.player_street_act[player]:
                return False
            if not self._is_player_finished(player) and \
               street_bet > 0 and self.player_street_bet[player] != street_bet:
                return False
        return True

    def is_over(self) -> bool:
        return self.street.value >= self.showdown_street.value or self._is_street_end() and self.num_unfinished_players <= 1

    @property
    def num_folded_players(self) -> int:
        return self.player_fold.count(True)

    @property
    def num_all_ined_players(self) -> int:
        return self.player_all_in.count(True)

    @property
    def num_unfinished_players(self) -> int:
        return self.num_players - self.num_folded_players - self.num_all_ined_players

    def create_action(
        self,
        type: ActionType,
        raise_chip: int = 0,
        raise_pot: float = 0,
        raise_to: int = 0,
        all_in_chip: int = 0,
        all_in_type: ActionType = ActionType.Call,
    ) -> Action:
        return Action(
            type = type,
            player = self.current_player,
            player_name = self.player_name[self.current_player],
            street = self.street,
            raise_chip = raise_chip,
            raise_pot = raise_pot,
            raise_to = raise_to,
            all_in_chip = all_in_chip,
            all_in_type = all_in_type,
        )

    def observe_current(self) -> Observation:
        return self.observe(self.current_player)

    def get_legal_action(self, player: int) -> list[Action]:
        legal_actions = [
            self.create_action(ActionType.Fold) if self._is_fold_legal() else None,
            self.create_action(ActionType.Check) if self._is_check_legal() else None,
            self.create_action(ActionType.Call) if self._is_call_legal() else None,
            self.create_action(ActionType.All_in) if self._is_all_in_legal() else None,
        ]
        for raise_pot in self.raise_pot_size:
            legal_actions.append(None)
            if raise_pot in self.legal_raise_pot_size:
                raise_chip, raise_pot = self._calculate_raise_chip(raise_pot = raise_pot)
                if self._is_raise_legal(raise_chip):
                    legal_actions[-1] = self.create_action(
                        ActionType.Raise,
                        raise_chip = raise_chip,
                        raise_pot = raise_pot,
                        raise_to = raise_chip + self.player_street_bet[player],
                    )
        return legal_actions
        
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
        )

    def _is_fold_legal(self) -> bool:
        return self._is_fold_legal_msg()[0]

    def _is_fold_legal_msg(self) -> tuple[bool, str]:
        if not self.legal_action_type[ActionType.Fold.value]:
            return False, 'This game does not support players to fold.'
        return self.street_raise != 0, 'Fold without opponent raising.'

    def _is_check_legal(self) -> bool:
        return self._is_check_legal_msg()[0]

    def _is_check_legal_msg(self) -> tuple[bool, str]:
        if self.player_street_num_actions[self.current_player] >= self.max_num_actions_street:
            return False, 'Player exhausts the number of actions on this street.'
        if not self.legal_action_type[ActionType.Check.value]:
            return False, 'This game does not support players to check.'
        return self.street_raise <= self.player_street_bet[self.current_player], 'Check when facing a raise.'

    def _is_call_legal(self) -> bool:
        return self._is_call_legal_msg()[0]

    def _is_call_legal_msg(self) -> tuple[bool, str]:
        if self.player_street_num_actions[self.current_player] >= self.max_num_actions_street:
            return False, 'Player exhausts the number of actions on this street.'
        if not self.legal_action_type[ActionType.Check.value]:
            return False, 'This game does not support players to call.'
        if (self.player_chips[self.current_player] <=
            self.street_raise - self.player_street_bet[self.current_player]):
            return False, 'Player should go all_in instead of call_ing.'
        return self.street_raise > self.player_street_bet[self.current_player], 'Call when there is no raise, please use check instead.'

    def _is_raise_legal(self, raise_chip: int) -> bool:
        return self._is_raise_legal_msg(raise_chip)[0]

    def _is_raise_legal_msg(self, raise_chip: int) -> tuple[bool, str]:
        if self.player_street_num_actions[self.current_player] >= self.max_num_actions_street:
            return False, 'Player exhausts the number of actions on this street.'
        if not self.legal_action_type[ActionType.Raise.value]:
            return False, 'This game does not support players to raise.'
        raise_to = raise_chip + self.player_street_bet[self.current_player]
        if raise_chip > self.player_chips[self.current_player]:
            return False, 'Player bets more chips than he/she has.'
        if raise_chip == self.player_chips[self.current_player]:
            return False, 'Player should go all_in instead of raising.'
        elif not self.player_can_raise[self.current_player]:
            return False, 'Player cannot raise twice in a row on the same street.'
        elif raise_to - self.street_raise < self.street_raise_delta:
            return False, 'The raise amount does not meet the minimum raise requirement.'
        return True, ''

    def _is_all_in_legal(self) -> bool:
        return self._is_all_in_legal_msg()[0]

    def _is_all_in_legal_msg(self) -> bool:
        if self.player_street_num_actions[self.current_player] >= self.max_num_actions_street:
            return False, 'Player exhausts the number of actions on this street.'
        if not self.legal_action_type[ActionType.All_in.value]:
            return False, 'This game does not support players to all-in.'
        all_in_chip = self.player_chips[self.current_player]
        all_in_to = all_in_chip + self.player_street_bet[self.current_player]
        _, raise_pot = self._calculate_raise_chip(raise_chip = all_in_chip)
        if raise_pot > 2.0:
            return False, 'Does not support all in over 2x pot.'
        if all_in_to - self.street_raise >= self.street_raise_delta:
            if not self.player_can_raise[self.current_player]:
                return False, 'Player cannot raise twice in a row on the same street.'
        return True, ''
    
    def _calculate_raise_chip(self, raise_chip: int = 0, raise_pot: float = 0) -> tuple[int, float]:
        if raise_chip == 0 and raise_pot > 0:
            delta = self.street_raise - self.player_street_bet[self.current_player]
            raise_chip = math.floor((self.pot + delta) * raise_pot + delta)
        if raise_chip > 0 and raise_pot < 1e-5:
            delta = self.street_raise - self.player_street_bet[self.current_player]
            raise_pot = (raise_chip - delta) / (self.pot + delta)
        return raise_chip, raise_pot

    def _perform_fold(self, action: Action, player: int) -> str:
        self.player_fold[player] = True
        self.player_street_act[player] = True
        return None

    def _perform_check(self, action: Action, player: int) -> str:
        if not (result := self._is_check_legal_msg())[0]:
            return result[1]
        else:
            self.player_street_act[player] = True
        return None

    def _perform_call(self, action: Action, player: int) -> str:
        if not (result := self._is_call_legal_msg())[0]:
            return result[1]
        else:
            delta = self.street_raise - self.player_street_bet[player]
            self.pot += delta
            self.player_chips[player] -= delta
            self.player_street_bet[player] += delta
            self.player_bet[player] += delta
            self.player_street_act[player] = True
        return None

    def _perform_raise(self, action: Action, player: int) -> str:
        action.raise_chip, action.raise_pot = self._calculate_raise_chip(action.raise_chip, action.raise_pot)
        raise_chip = action.raise_chip
        raise_to = raise_chip + self.player_street_bet[player]
        if not (result := self._is_raise_legal_msg(raise_chip))[0]:
            return result[1]
        else:
            self.player_can_raise = [i != player for i in range(self.num_players)]
            self.pot += raise_chip
            self.player_street_bet[player] += raise_chip
            self.player_bet[player] += raise_chip
            self.player_chips[player] -= raise_chip
            self.player_street_act[player] = True
            self.player_street_num_raises[player] += 1
            self.street_raise_delta = raise_to - self.street_raise
            self.street_raise = raise_to
        return None
    
    def _perform_all_in(self, action: Action, player: int) -> str:
        action.raise_chip = 0
        action.raise_pot = 0
        action.all_in_chip = self.player_chips[player]
        all_in_to = action.all_in_chip + self.player_street_bet[player]

        if not (result := self._is_all_in_legal_msg())[0]:
            return result[1]
        else:
            if all_in_to - self.street_raise >= self.street_raise_delta:
                action.all_in_type = ActionType.Raise
                self.player_street_bet[player] += action.all_in_chip
                self.player_bet[player] += action.all_in_chip
                self.pot += action.all_in_chip
                self.player_chips[player] -= action.all_in_chip
                self.player_all_in[player] = True
                self.player_can_raise = [i != player for i in range(self.num_players)]
                self.player_street_act[player] = True
                self.street_raise_delta = all_in_to - self.street_raise
                self.street_raise = all_in_to
            else:
                action.all_in_type = ActionType.Call
                self.player_street_bet[player] += action.all_in_chip
                self.player_bet[player] += action.all_in_chip
                self.pot += action.all_in_chip
                self.player_chips[player] -= action.all_in_chip
                self.player_all_in[player] = True
                self.player_street_act[player] = True
                if all_in_to > self.street_raise:
                    self.street_raise_delta += all_in_to - self.street_raise
                    self.street_raise = all_in_to
        return None

    def _perform_action(self, action: Action, player: int):
        self.player_street_num_actions[player] += 1
        return {
            ActionType.Fold: self._perform_fold,
            ActionType.Check: self._perform_check,
            ActionType.Call: self._perform_call,
            ActionType.Raise: self._perform_raise,
            ActionType.All_in: self._perform_all_in,
        }[action.type](action, player)

    def _enter_new_street(self):
        self.street = self.street.next()
        self.street_raise = 0
        self.street_raise_delta = 0
        self.player_can_raise = [True] * self.num_players
        self.player_street_bet = [0] * self.num_players
        self.player_street_act = [False] * self.num_players
        self.player_street_num_actions = [0] * self.num_players
        self.player_street_num_raises = [0] * self.num_players
        self.current_player = self._next_unfinished_player(0)

    def _refresh_board(self):
        if self.street == Street.Preflop or self.street == Street.Showdown:
            return
        need_to_deal = self.num_street_board_cards[self.street.value - 1]
        already_dealed = sum(self.num_street_board_cards[:self.street.value - 1])
        if self.custom_board_cards is not None and len(self.custom_board_cards) >= already_dealed + need_to_deal:
            self.board_cards.extend(self.custom_board_cards[already_dealed:already_dealed + need_to_deal])
        else:
            self.board_cards.extend(self.dealer.deal(need_to_deal))

    def _calculate_payoff(self) -> list[float]:
        if len(self.board_cards) == self.num_board_cards:
            return self.judger.judge(
                pot = self.pot,
                num_players = self.num_players,
                board_cards = self.board_cards,
                player_bet = self.player_bet,
                player_hole_cards = self.player_hole_cards,
                player_fold = self.player_fold,
            )
        payoff = [0.0] * self.num_players
        for _ in range(self.num_runs):
            board = deepcopy(self.board_cards)
            board.extend(self.dealer.deal_without_pop(self.num_board_cards - len(board)))
            self.run_payoff = self.judger.judge(
                pot = self.pot,
                num_players = self.num_players,
                board_cards = board,
                player_bet = self.player_bet,
                player_hole_cards = self.player_hole_cards,
                player_fold = self.player_fold,
            )
            payoff = [x + y / self.num_runs for x, y in zip(payoff, self.run_payoff)]
        return payoff
    
    def step(self, action: Action) -> Observation:
        if self.verbose:
            log.info(f"[yellow][bold]Observation[/bold][/yellow]\n{self.observe(self.current_player)}", extra=dict(markup=True))
            log.info(f"[red][bold]Take Action[/bold][/red]\n{action}", extra=dict(markup=True))
        action = action if action is not None else self.create_action(ActionType.Fold)
        action.street = self.street
        player = self.current_player
        if self.is_over():
            self.current_player = self._next_unfinished_player(player + 1)
            return self.observe(self.current_player)
        
        error = self._perform_action(action, player)
        # When a player makes an invalid action, it is considered as a fold
        if error is not None:
            if self.verbose:
                log.error(error)
            self._perform_fold(self.create_action(ActionType.Fold), player)

        self.log_action.append(action)

        if not self._is_street_end():
            self.current_player = self._next_unfinished_player(player + 1)
        else:
            self._enter_new_street()
            if self.is_over():
                self.player_payoff = self._calculate_payoff()
                self.player_chips = [x + y for x, y in zip(self.player_payoff, self.initial_chips)]
                self.current_player = self._next_unfinished_player(player + 1)
                self.street = Street.Showdown
            else:
                self._refresh_board()

        return self.observe(self.current_player)
        
    def run(self, seed: int = None) -> list[Observation]:
        observation = self.reset(seed)
        history = [observation]
        while True:
            action = self.agents[observation.player].step(observation)
            observation = self.step(action)
            history.append(observation)
            if self.is_over():
                break
        if self.verbose:
            log.info("[green][bold]RESULT[/bold][/green]", extra=dict(markup=True))
        for i in range(self.num_players):
            history.append(self.observe(i))
            if self.verbose:
                log.info(history[-1])
        return history