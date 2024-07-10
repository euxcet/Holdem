from .poker_game import PokerGame
from .agent.agent import Agent
from .component.action import Action, ActionType
from .component.card import Card
from .component.deck import Deck
from .component.street import Street

class KuhnPoker(PokerGame):
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