import torch
import numpy as np
from ..model.hunl_conv_model import HUNLConvModel
from ..model.hunl_supervise_model import HUNLSuperviseModel, HUNLSuperviseSimpleModel
from ..model.hunl_supervise_resnet import HUNLSuperviseResnet, HUNLSuperviseResnet50
from ..poker.no_limit_texas_holdem_env import NoLimitTexasHoldemEnv
from ..poker.component.card import Card
from ..poker.component.observation import Observation
from ..poker.component.street import Street

class Solver():
    def __init__(
        self,
        model_path: str,
        showdown_street: Street,
    ) -> None:
        self.model: HUNLSuperviseResnet = HUNLSuperviseResnet()
        # self.model: HUNLConvModel = torch.load(model_path)
        # self.model: HUNLSuperviseModel = HUNLSuperviseModel()
        # self.model: HUNLSuperviseModel = HUNLSuperviseSimpleModel()
        # self.model: HUNLSuperviseModel = HUNLSuperviseResnet50()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to('cuda')
        self.model.eval()
        self.showdown_street = showdown_street
        self.hole_card_tensor = self._create_hole_card_tensor()
        
    def _create_hole_card_tensor(self) -> np.ndarray:
        tensor = np.zeros((1326, 1, 4, 13), dtype=np.float32)
        hole_id = 0
        for i in range(52):
            for j in range(i + 1, 52):
                card0 = Card(suit_first_id=i)
                card1 = Card(suit_first_id=j)
                for hole_card in [card0, card1]:
                    tensor[hole_id][0][hole_card.suit][hole_card.rank] = 1.0
                hole_id += 1
        return tensor

    def get_range_policy(self, cards: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor) -> np.ndarray:
        # print(obs['obs']['action_mask'].shape, obs['obs']['action_mask'])
        can_check = action_mask[0][1] > 0.5
        prob: np.ndarray = self.model(cards, actions).detach().cpu().numpy()
        print(prob)
        empty = np.zeros((prob.shape[0]), dtype=np.float32)
        if can_check:
            prob = np.stack((prob[:, 0], prob[:, 1], empty, prob[:, 3], prob[:, 2]), axis=1)
        else:
            prob = np.stack((prob[:, 0], empty, prob[:, 1], prob[:, 3], prob[:, 2]), axis=1)
        return prob

    def query(
        self,
        board_cards: list[str],
        action_history: list[int],
    ) -> tuple[np.ndarray, Observation]:
        env = NoLimitTexasHoldemEnv(
            num_players=2,
            initial_chips=200,
            showdown_street=self.showdown_street,
            custom_board_cards=Card.from_str_list(board_cards),
            raise_pot_size=[1],
            legal_raise_pot_size=[1],
        )
        env.reset()
        for action in action_history:
            env.step(action)
        game_obs = env.game.observe_current()
        observation = env.observe_current()

        cards = np.tile(observation['observation'][np.newaxis, 1:, :], (1326, 1, 1, 1))
        cards = np.concatenate((self.hole_card_tensor, cards), axis=1)
        actions = np.tile(observation['action_history'][np.newaxis, :], (1326, 1, 1, 1))
        action_mask = np.tile(observation['action_mask'][np.newaxis, :], (1326, 1))
        cards = torch.from_numpy(cards).to('cuda')
        actions = torch.from_numpy(actions).to('cuda')
        action_mask = torch.from_numpy(action_mask).to('cuda')
        # cards:       1326 * 4 * 4 * 13
        # actions:     1326 * 4 * 12 * 5
        # action_mask: 1326 * 5
        print(game_obs)
        return self.get_range_policy(cards, actions, action_mask), game_obs