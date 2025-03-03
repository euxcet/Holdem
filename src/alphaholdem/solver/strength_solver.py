import os
import math
import torch
import numpy as np
from ..model.hunl_conv_model import HUNLConvModel
from ..poker.no_limit_texas_holdem_env import NoLimitTexasHoldemEnv
from ..poker.component.card import Card
from ..poker.component.observation import Observation
from ..poker.component.street import Street
from ..poker.component.dealer import Dealer
from ..poker.component.judger import Judger

class StrengthSolver():
    def __init__(
        self,
        model_path: str,
        showdown_street: Street,
        epoch: int,
    ) -> None:
        model_with_epoch = model_path[:-3] + '_' + str(epoch) + '.pt'
        if os.path.exists(model_with_epoch):
            self.model: HUNLConvModel = torch.load(model_with_epoch)
        else:
            self.model: HUNLConvModel = torch.load(model_path)
        print(type(self.model))
        self.model.to('cuda')
        self.model.eval()
        self.showdown_street = showdown_street

    def map_suit(self, card0: Card, card1: Card, suit_dict: dict, suit_c: int):
        if card0 < card1:
            card0, card1 = card1, card0
        if card0.suit not in suit_dict:
            suit_dict[card0.suit] = suit_c
            suit_c += 1
        if card1.suit not in suit_dict:
            suit_dict[card1.suit] = suit_c
            suit_c += 1
        card0.suit = suit_dict[card0.suit]
        card1.suit = suit_dict[card1.suit]
        return card0, card1

    def get_range_policy(
        self,
        judger: Judger,
        dealer: Dealer,
        board_cards: list[Card],
        obs: dict,
        suit_dict: dict,
        suit_c: int
    ) -> list[list]:
        policy = []
        strength = judger.get_all_strength(dealer, board_cards)
        cnt = 0
        for i in range(52):
            for j in range(i + 1, 52):
                # TODO: batch
                card0, card1 = self.map_suit(Card(suit_first_id=i), Card(suit_first_id=j), suit_dict.copy(), suit_c)
                for hole_card in [card0, card1]:
                    obs['obs']['observation'][0][0][hole_card.suit][hole_card.rank] = 1.0
                obs['obs']['strength'][0][0] = strength[cnt]
                prob = torch.exp(self.model(obs)[0])
                prob = prob / torch.sum(prob)
                prob = prob.detach().cpu().numpy().squeeze()
                policy.append(prob)
                for hole_card in [card0, card1]:
                    obs['obs']['observation'][0][0][hole_card.suit][hole_card.rank] = 0
                cnt += 1
        return np.array(policy)


    def query(
        self,
        board_cards: list[str],
        action_history: list[int],
    ) -> tuple[np.ndarray, Observation]:
        board_cards: list[Card] = Card.from_str_list(board_cards)
        board_cards = sorted(board_cards[:3], reverse=True) + board_cards[3:]
        env = NoLimitTexasHoldemEnv(
            num_players=2,
            initial_chips=200,
            showdown_street=self.showdown_street,
            custom_board_cards=board_cards.copy(),
            raise_pot_size=[1],
            legal_raise_pot_size=[1],
        )
        env.reset()
        for action in action_history:
            env.step(action)
        game_obs = env.game.observe_current()
        observation = env.observe_current()

        # Fixed suit
        suit_dict = {}
        suit_c = 0
        num_board = 0
        if game_obs.street == Street.Flop:
            num_board = 3
        elif game_obs.street == Street.Turn:
            num_board = 4
        elif game_obs.street == Street.River:
            num_board = 5
        for i in range(num_board):
            if board_cards[i].suit not in suit_dict:
                suit_dict[board_cards[i].suit] = suit_c
                suit_c += 1
        obs = {
            'obs': {
                'observation': torch.from_numpy(observation['observation'])[np.newaxis, :].to('cuda'),
                'action_history': torch.from_numpy(observation['action_history'])[np.newaxis, :].to('cuda'),
                'action_mask': torch.from_numpy(observation['action_mask'])[np.newaxis, :].to('cuda'),
                'strength': torch.from_numpy(np.array([0], dtype=np.float32)[np.newaxis, :]).to('cuda'),
            }
        }
        obs['obs']['observation'][0][0] = torch.zeros((4, 13))
        return self.get_range_policy(env.game.judger, env.game.dealer, board_cards, obs, suit_dict, suit_c), game_obs