from .poker_game import PokerGame
from .agent.agent import Agent
from .component.card import Card
from .component.deck import Deck
from .component.street import Street

class NoLimitTexasHoldem(PokerGame):
    def __init__(
        self,
        num_players: int,
        agents: list[Agent] = None,
        initial_chips: list[int] = None,
        verbose: bool = False,
        num_runs: int = 10,
        showdown_street: Street = Street.Showdown,
        custom_board_cards: list[Card] = None,
        custom_player_hole_cards: list[list[Card]] = None,
        legal_raise_pot_size: list[float] = [0.25, 0.5, 0.75, 1.25],
    ) -> None:
        super().__init__(
            num_players=num_players,
            agents=agents,
            initial_chips=initial_chips,
            initial_deck=Deck.deck_52(),
            verbose=verbose,
            num_runs=num_runs,
            showdown_street=showdown_street,
            custom_board_cards=custom_board_cards,
            custom_player_hole_cards=custom_player_hole_cards,
            legal_raise_pot_size=legal_raise_pot_size,
        )