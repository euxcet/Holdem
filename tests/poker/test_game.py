from rich import print
from alphaholdem.poker.no_limit_texas_holdem import NoLimitTexasHoldem
from alphaholdem.poker.no_limit_leduc_holdem import NoLimitLeducHoldem
from alphaholdem.poker.kuhn_poker import KuhnPoker
from alphaholdem.poker.limit_leduc_holdem import LimitLeducHoldem
from alphaholdem.poker.component.observation import Observation
from alphaholdem.poker.component.card import Card
from alphaholdem.poker.component.street import Street

class TestGame():
    # Fold Check Call All_in Raise_25% Raise_50% Raise_75% Raise_125%
    # 0    1     2    3      4         5         6         7
    def test_all_in_equity(self):
        game = NoLimitTexasHoldem(
            num_players=2,
            initial_chips=[200, 200],
            custom_board_cards=Card.from_str_list(['Ac', '8c', '5h', '6c', '7h']),
            showdown_street=Street.Showdown,
            num_runs=1000,
            custom_player_hole_cards=[
                Card.from_str_list(['Ks', '7s']),
                Card.from_str_list(['Kc', '7c']),
            ]
        )
        # game.reset()
        # obs = game.observe_current()
        # game.step(obs.simple_legal_actions[2])
        # obs = game.observe_current()
        # game.step(obs.simple_legal_actions[1])
        # obs = game.observe_current()
        # game.step(obs.simple_legal_actions[1])
        # obs = game.observe_current()
        # game.step(obs.simple_legal_actions[1])
        # obs = game.observe_current()
        # game.step(obs.simple_legal_actions[1])
        # obs = game.observe_current()
        # game.step(obs.simple_legal_actions[1])
        # obs = game.observe_current()
        # game.step(obs.simple_legal_actions[7])
        # obs = game.observe_current()
        # game.step(obs.simple_legal_actions[2])
        # obs = game.observe_current()
        # print(obs)

    # Fold Check Call All_in Raise_25% Raise_50% Raise_75% Raise_125%
    # 0    1     2    3      4         5         6         7
    def test_leduc_holdem(self):
        game = LimitLeducHoldem(
            num_players=2,
            initial_chips=[100, 100],
            showdown_street=Street.Turn,
            num_runs=10,
            # custom_board_cards=Card.from_str_list([]),
            # custom_player_hole_cards=[],
        )
        game.reset()
        obs = game.observe_current()
        game.step(obs.legal_actions[4])
        obs = game.observe_current()
        game.step(obs.legal_actions[4])
        obs = game.observe_current()
        game.step(obs.legal_actions[2])
        obs = game.observe_current()
        game.step(obs.legal_actions[4])
        obs = game.observe_current()
        game.step(obs.legal_actions[4])
        obs = game.observe_current()
        game.step(obs.legal_actions[2])
        obs = game.observe_current()
        # print(obs)

    def test_kuhn(self):
        # Fold Check Call Raise
        return
        game = KuhnPoker()
        game.reset()
        obs = game.observe_current()
        print(obs.legal_actions, obs)
        game.step(obs.legal_actions[1])
        obs = game.observe_current()
        print(obs.legal_actions, obs)
        game.step(obs.legal_actions[3])
        obs = game.observe_current()
        print(obs.legal_actions, obs)
        game.step(obs.legal_actions[2])
        obs = game.observe_current()
        print(obs.legal_actions, obs)