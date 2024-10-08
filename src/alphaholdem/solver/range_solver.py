import os
import torch
from torch import nn
import numpy as np
from ..model.hunl_supervise_range_model import HUNLSuperviseRangeModel
from ..poker.no_limit_texas_holdem_env import NoLimitTexasHoldemEnv
from ..poker.component.card import Card
from ..poker.component.observation import Observation
from ..poker.component.street import Street

class RangeSolver():
    def __init__(
        self,
        model_folder: str,
        # model_path: str,
    ) -> None:
        self.model_folder = model_folder
        # self.model: HUNLSuperviseRangeModel = HUNLSuperviseRangeModel()
        # self.model.load_state_dict(torch.load(model_path))
        # self.model.to('cuda')
        # self.model.eval()

    # def query(
    #     self,
    #     board_cards: list[str],
    #     action_history: list[int],
    # ) -> tuple[np.ndarray, Observation]:
    #     env = NoLimitTexasHoldemEnv(
    #         num_players=2,
    #         initial_chips=200,
    #         showdown_street=Street.Showdown,
    #         custom_board_cards=Card.from_str_list(board_cards),
    #         raise_pot_size=[1],
    #         legal_raise_pot_size=[1],
    #     )
    #     env.reset()
    #     for action in action_history:
    #         env.step(action)
    #     game_obs = env.game.observe_current()
    #     observation = env.observe_current()

    #     cards = torch.from_numpy(observation['observation'][np.newaxis, 1:, :]).to('cuda')
    #     actions = torch.from_numpy(observation['action_history'][np.newaxis, :]).to('cuda')
    #     action_mask = observation['action_mask']

    #     can_check = action_mask[1] > 0.5
    #     prob: np.ndarray = self.model(cards, actions).detach().cpu().numpy().reshape((1326, 4))
    #     empty = np.zeros((1326), dtype=np.float32)
    #     if can_check:
    #         prob = np.stack((prob[:, 0], prob[:, 1], empty, prob[:, 3], prob[:, 2]), axis=1)
    #     else:
    #         prob = np.stack((prob[:, 0], empty, prob[:, 1], prob[:, 3], prob[:, 2]), axis=1)
    #     return prob, game_obs

    def load_model(
        self,
        street: Street,
    ) -> nn.Module:
        model_name = 'range_' + street.name.lower() + '.pt'
        model: HUNLSuperviseRangeModel = HUNLSuperviseRangeModel()
        model.load_state_dict(torch.load(os.path.join(self.model_folder, model_name)))
        model.to('cuda')
        model.eval()
        return model

    def query(
        self,
        board_cards: list[str],
        action_history: list[int],
    ) -> tuple[np.ndarray, Observation]:
        env = NoLimitTexasHoldemEnv(
            num_players=2,
            initial_chips=200,
            showdown_street=Street.Showdown,
            custom_board_cards=Card.from_str_list(board_cards),
            raise_pot_size=[1],
            legal_raise_pot_size=[1],
        )
        env.reset()
        for action in action_history:
            env.step(action)
        game_obs = env.game.observe_current()
        observation = env.observe_current()

        cards = torch.from_numpy(observation['observation'][np.newaxis, 1:, :]).to('cuda')
        actions = torch.from_numpy(observation['action_history'][np.newaxis, :]).to('cuda')
        action_mask = observation['action_mask']

        can_check = action_mask[1] > 0.5

        model = self.load_model(game_obs.street)

        prob: np.ndarray = model(cards, actions).detach().cpu().numpy().reshape((1326, 4))
        empty = np.zeros((1326), dtype=np.float32)
        if can_check:
            prob = np.stack((prob[:, 0], prob[:, 1], empty, prob[:, 3], prob[:, 2]), axis=1)
        else:
            prob = np.stack((prob[:, 0], empty, prob[:, 1], prob[:, 3], prob[:, 2]), axis=1)
        return prob, game_obs
