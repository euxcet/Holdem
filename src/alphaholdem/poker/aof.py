from .poker_game import PokerGame
from .agent.agent import Agent
from .component.action import ActionType
from .component.deck import Deck

# All-in or Fold
class AoF(PokerGame):
    def __init__(
        self,
        num_players: int,
        agents: list[Agent] = None,
        initial_chips: list[int] = None,
        num_blinds: int = 2,
        verbose: bool = False,
        num_runs: int = 10,
    ) -> None:
        super().__init__(
            num_players = num_players,
            agents = agents,
            initial_chips = [20] * num_players,
            initial_deck = Deck.deck_52(),
            num_hole_cards = num_blinds,
            legal_action_type = ActionType.aof_legal(),
            verbose = verbose,
            num_runs = num_runs,
        )