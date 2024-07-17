from .action import Action
from .card import Card
from .street import Street
from .player_name import get_players_name

class Observation():
    def __init__(
        self,
        player: int,
        num_players: int,
        hole_cards: list[Card],
        chips: int,
        current_player: int,
        player_chips: list[int],
        player_street_bet: list[int],
        player_payoff: list[int],
        pot: int,
        street: Street,
        board_cards: list[Card],
        legal_actions: list[Action],
        log_action: list[Action],
        is_over: bool,
        players_range: list[list[float]] = None,
    ) -> None:
        self.player = player
        self.num_players = num_players
        self.hole_cards = hole_cards
        self.chips = chips
        self.current_player = current_player
        self.player_chips = player_chips
        self.player_street_bet = player_street_bet
        self.player_payoff = player_payoff
        self.pot = pot
        self.street = street
        self.board_cards = board_cards
        self.legal_actions = legal_actions
        self.log_action = log_action
        self.player_name = get_players_name(num_players=num_players)
        self.is_over = is_over
        self.players_range = players_range

    def __str__(self) -> str:
        return f'player={self.player_name[self.player]} pot={self.pot / 2}BB chips={self.chips / 2}BB ' + \
               f'current_player={self.player_name[self.current_player]}\n' + \
               f'Hole cards: {self.hole_cards}   Board cards: {self.board_cards}'\
               '\nAction Log:\n' + \
               ''.join(map(str, self.log_action)) + \
               'payoff = ' + str(list(map(lambda x: x / 2, self.player_payoff))) + \
               '\n'

    def __repr__(self) -> str:
        return self.__str__()