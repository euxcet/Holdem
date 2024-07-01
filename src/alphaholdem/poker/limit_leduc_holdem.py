from .poker_game import PokerGame
from .agent.agent import Agent
from .component.action import Action, ActionType
from .component.card import Card
from .component.deck import Deck
from .component.street import Street

class LimitLeducHoldem(PokerGame):
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
            raise_pot_size=[0.75],
            showdown_street=showdown_street,
            custom_board_cards=custom_board_cards,
            custom_player_hole_cards=custom_player_hole_cards,
            num_street_board_cards=num_street_board_cards,
        )
    
    def get_legal_action(self, player: int) -> list[Action]:
        legal_actions = [
            self.create_action(ActionType.Fold) if self._is_fold_legal() else None,
            self.create_action(ActionType.Check) if self._is_check_legal() else None,
            self.create_action(ActionType.Call) if self._is_call_legal() else None,
            None,
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