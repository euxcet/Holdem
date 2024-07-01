import os
import math
import time
import torch
import torch.nn.functional as F
import numpy as np
from alphaholdem.model.hunl_conv_model import HUNLConvModel
from alphaholdem.poker.no_limit_leduc_holdem_env import NoLimitLeducHoldemEnv
from alphaholdem.poker.limit_leduc_holdem_env import LimitLeducHoldemEnv
from alphaholdem.poker.component.card import Card
from alphaholdem.poker.component.observation import Observation
from alphaholdem.poker.component.street import Street
from alphaholdem.poker.component.action import ActionType

class TestPolicy():
    pass
    # def get_range_policy(self, obs: dict) -> list[list]:
    #     policy = []
    #     for i in range(52):
    #         for j in range(i + 1, 52):
    #             # TODO: batch
    #             card0 = Card(suit_first_id=i)
    #             card1 = Card(suit_first_id=j)
    #             for hole_card in [card0, card1]:
    #                 obs['obs']['observation'][0][0][hole_card.suit][hole_card.rank] = 1.0

    #             prob = torch.exp(self.model(obs)[0])
    #             prob = prob / torch.sum(prob)
    #             prob = prob.detach().cpu().numpy().squeeze()
    #             policy.append(prob)

    #             for hole_card in [card0, card1]:
    #                 obs['obs']['observation'][0][0][hole_card.suit][hole_card.rank] = 0.0
    #     return np.array(policy)

    # Fold Check Call All_in Raise_25% Raise_50% Raise_75% Raise_125%
    # 0    1     2    3      4         5         6         7