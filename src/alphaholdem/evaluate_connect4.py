import ray
import numpy as np
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from pettingzoo.classic import connect_four_v3
from .model.connect4_mask_conv_model import Connect4MaskConvModel

def display(observation):
    board_obs: np.ndarray = observation['observation']

    board = [['_' for i in range(7)] for j in range(6)]
    for i in range(6):
        for j in range(7):
            if board_obs[i][j][0]:
                board[i][j] = '+'
            if board_obs[i][j][1]:
                board[i][j] = '#'
    print('Board:')
    print("0123456")
    for i in range(6):
        print(''.join(board[i]))
    print()

def main():
    ray.init(num_gpus=8)
    env_name = "connect_four"
    register_env(env_name, lambda _: PettingZooEnv(
        connect_four_v3.env()
    ))
    ModelCatalog.register_custom_model("BaselineModel", Connect4MaskConvModel)
    PPOagent = PPO.from_checkpoint('/home/clouduser/ray_results/PPO_2024-04-25_23-38-05/PPO_connect_four_dd80c_00000_0_2024-04-25_23-38-06/checkpoint_000129')
    env = connect_four_v3.env()
    while True:
        env.reset()
        reward_sum = 0
        id = 0
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            reward_sum += reward
            display(observation)
            if termination or truncation:
                action = None
            else:
                if id == 0:
                    action = int(input('Input your action:'))
                else:
                    action = PPOagent.compute_single_action(observation, policy_id='learned')
            id = (id + 1) % 2
            print('action =', action)
            env.step(action)
        env.close()
        print(reward_sum)