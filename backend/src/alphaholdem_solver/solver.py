import math
import torch
import numpy as np
from alphaholdem.model.hunl_conv_model import HUNLConvModel
from alphaholdem.poker.no_limit_texas_holdem_env import NoLimitTexasHoldemEnv
from alphaholdem.poker.component.card import Card
from alphaholdem.poker.component.observation import Observation
from alphaholdem.poker.component.street import Street

class Solver():
    def __init__(
        self,
        model_path: str,
        showdown_street: Street,
    ) -> None:
        self.model: HUNLConvModel = torch.load(model_path)
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

                prob = torch.exp(self.model(obs)[0])
                prob = prob / torch.sum(prob)
                prob = prob.detach().cpu().numpy().squeeze()
                policy.append(prob)

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
            legal_raise_pot_size=[0.75],
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