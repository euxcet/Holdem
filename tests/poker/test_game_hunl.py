from rich import print
from alphaholdem.poker.no_limit_texas_holdem import NoLimitTexasHoldem
from alphaholdem.poker.no_limit_leduc_holdem import NoLimitLeducHoldem
from alphaholdem.poker.kuhn_poker import KuhnPoker
from alphaholdem.poker.limit_leduc_holdem import LimitLeducHoldem
from alphaholdem.poker.component.observation import Observation
from alphaholdem.poker.component.card import Card
from alphaholdem.poker.component.street import Street

class TestHunlGame():
    def test_hunl_game(self):
        game = NoLimitTexasHoldem(
            num_players=2,
            initial_chips=[200, 200],
            custom_board_cards=Card.from_str_list(['Qh', '3h', 'As', '3s', 'Qs']),
            showdown_street=Street.Showdown,
            num_runs=100,
            raise_pot_size=[1],
            legal_raise_pot_size=[1],
            custom_player_hole_cards=[
                Card.from_str_list(['Qc', '4c']),
                Card.from_str_list(['Qd', '5d']),
            ],
            preflop_strategy='./strategy/hunl/simple',
        )
        game.reset()
        # obs = game.observe_current()
        # print(obs)
        # game.step(obs.legal_actions[2])
        # obs = game.observe_current()
        # game.step(obs.legal_actions[1])
        # obs = game.observe_current()
        # game.step(obs.legal_actions[1])
        # obs = game.observe_current()
        # game.step(obs.legal_actions[1])
        # obs = game.observe_current()
        # game.step(obs.legal_actions[1])
        # obs = game.observe_current()
        # game.step(obs.legal_actions[1])
        # obs = game.observe_current()
        # game.step(obs.legal_actions[1])
        # obs = game.observe_current()
        # game.step(obs.legal_actions[1])
        # obs = game.observe_current()
        # print(obs)
