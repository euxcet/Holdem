import os
import ray
import numpy as np
import supersuit as ss
import math
from ray import train, tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from torch import nn
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn, aec_wrapper_fn
from pettingzoo.test import parallel_api_test, api_test
from .poker.poker_game_env import PokerGameEnv
from .poker.aof_env import AoFEnv
from .poker.component.card import Card
from .model.fully_connected_model import FullyConnectedModelV2
from .model.aof_conv_model import AoFConvModel

def wrap(env: PokerGameEnv):
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def get_range_policy(agent: Algorithm, action_history) -> list[list]:
    policy = [[0 for i in range(13)] for j in range(13)]
    for i in range(13):
        for j in range(13):
            # 'observation': spaces.Box(
            #     low=0.0, high=1.0, shape=(4, 4, 13), dtype=np.float32
            # ),
            # # 4(preflop, flop, turn, river) * num_players * max_num_actions_street, 5(fold, check, call, raise, all_in)
            # 'action_history': spaces.Box( 
            #     low=0.0, high=5.0, shape=(4, num_players * self.max_num_actions_street, 5), dtype=np.float32
            # ),
            observation_mat = np.zeros((4, 4, 13), np.float32)
            action_mask = np.ones(2, np.int8)
            if i <= j: # pair or offsuit
                card0 = Card(rank=i, suit=0)
                card1 = Card(rank=j, suit=1)
            else:
                card0 = Card(rank=i, suit=0)
                card1 = Card(rank=j, suit=0)
            for hole_card in [card0, card1]:
                observation_mat[0][hole_card.suit][hole_card.rank] = 1.0

            # for run in range(100):
            action = agent.compute_single_action(
                {"observation": observation_mat, "action_history": action_history, "action_mask": action_mask},
                policy_id='learned',
                full_fetch=True,
            )
            prob = action[2]['action_dist_inputs']
            policy[i][j] = math.exp(prob[1]) / (math.exp(prob[0]) + math.exp(prob[1]))
                # exit(0)
                # policy[i][j] += action / 100.0
    return policy

def print_range_policy(policy):
    for i in range(12, -1, -1):
        for j in range(12, -1, -1):
            print('%.2f' % policy[i][j], end = ' ')
        print()


def main():
    ray.init(
        num_gpus=8
    )

    env_name = "aof"
    register_env(env_name, lambda _: PettingZooEnv(
        wrap(env = AoFEnv(num_players=2))
    ))

    ModelCatalog.register_custom_model("BaselineModel", AoFConvModel)

    agent = PPO.from_checkpoint('/home/clouduser/ray_results/PPO_2024-04-27_11-34-14/PPO_aof_1329d_00000_0_2024-04-27_11-34-14/checkpoint_000002')

    print('SB policy')
    action_history = np.zeros((4, 12, 5), np.float32)
    print_range_policy(get_range_policy(agent, action_history))

    print('BB policy')
    action_history = np.zeros((4, 12, 5), np.float32)
    action_history[0][0][4] = 1
    print_range_policy(get_range_policy(agent, action_history))

"""
SB policy
1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.97 0.94 0.77 0.66
1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.99 1.00 0.99 0.90 0.84 0.74
1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.97 0.82 0.81 0.72
1.00 1.00 1.00 1.00 1.00 1.00 0.98 1.00 0.99 0.96 0.80 0.56 0.42
1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.93 0.60 0.18 0.18
1.00 1.00 1.00 1.00 1.00 1.00 0.97 0.98 1.00 0.82 0.28 0.03 0.06
1.00 1.00 1.00 1.00 1.00 1.00 0.99 1.00 0.98 0.75 0.12 0.02 0.02
1.00 1.00 1.00 1.00 1.00 0.99 1.00 0.98 0.70 0.39 0.07 0.02 0.03
1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.92 0.48 0.07 0.03 0.01 0.00
1.00 1.00 1.00 1.00 1.00 0.98 0.98 0.93 0.36 0.12 0.01 0.00 0.02
1.00 1.00 1.00 0.98 0.97 0.54 0.63 0.44 0.25 0.08 0.04 0.00 0.02
0.99 0.99 0.86 0.81 0.47 0.24 0.17 0.12 0.10 0.13 0.05 0.02 0.02
1.00 0.99 0.87 0.78 0.44 0.21 0.15 0.07 0.03 0.09 0.13 0.04 0.00
BB policy
1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.97 0.51 0.16 0.18
1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.97 0.64 0.30 0.31
1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.89 0.28 0.09 0.14
1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.86 0.16 0.10 0.09
1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.99 0.99 0.70 0.13 0.04 0.04
1.00 1.00 1.00 1.00 1.00 1.00 0.99 1.00 0.96 0.13 0.02 0.00 0.02
1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.99 0.95 0.16 0.04 0.02 0.00
1.00 1.00 1.00 1.00 0.99 1.00 0.98 0.95 0.38 0.07 0.01 0.01 0.03
1.00 1.00 1.00 1.00 1.00 1.00 0.98 0.74 0.09 0.00 0.02 0.01 0.00
1.00 1.00 1.00 1.00 0.99 0.94 0.86 0.35 0.08 0.02 0.00 0.01 0.01
1.00 1.00 1.00 0.94 0.73 0.10 0.07 0.07 0.06 0.00 0.00 0.00 0.00
0.99 0.97 0.64 0.31 0.06 0.02 0.02 0.02 0.00 0.02 0.02 0.00 0.00
0.98 1.00 0.76 0.29 0.06 0.04 0.02 0.01 0.00 0.04 0.01 0.00 0.00
"""