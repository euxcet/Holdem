import torch
import numpy as np
from ..model.hunl_conv_model import HUNLConvModel
from ..model.hunl_supervise_resnet import HUNLSuperviseResnet
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
        # self.model: HUNLConvModel = torch.load(model_path)
        self.model: HUNLSuperviseResnet = HUNLSuperviseResnet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to('cuda')
        self.model.eval()
        self.showdown_street = showdown_street

    def get_range_policy(self, obs: dict) -> list[list]:
        policy = []
        for i in range(52):
            for j in range(i + 1, 52):
                # TODO: batch
                card0 = Card(suit_first_id=i)
                card1 = Card(suit_first_id=j)
                for hole_card in [card0, card1]:
                    obs['obs']['observation'][0][0][hole_card.suit][hole_card.rank] = 1.0

                prob = self.model(obs['obs']['observation'], obs['obs']['action_history'])
                # prob = torch.exp(self.model(obs)[0])
                prob = prob / torch.sum(prob)
                prob = prob.detach().cpu().numpy().squeeze()
                # fold check/call raise all_in
                # -> fold check call all_in raise25 raise50 raise75 raise125
                policy.append([prob[0], prob[1], 0, prob[3], 0, 0, 0, prob[2]])

                for hole_card in [card0, card1]:
                    obs['obs']['observation'][0][0][hole_card.suit][hole_card.rank] = 0.0
        return np.array(policy)


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
        obs = {
            'obs': {
                'observation': torch.from_numpy(observation['observation'])[np.newaxis, :].to('cuda'),
                'action_history': torch.from_numpy(observation['action_history'])[np.newaxis, :].to('cuda'),
                'action_mask': torch.from_numpy(observation['action_mask'])[np.newaxis, :].to('cuda'),
            }
        }
        obs['obs']['observation'][0][0] = torch.zeros((4, 13))
        return self.get_range_policy(obs), game_obs