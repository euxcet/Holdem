import os
import shutil
import torch
from torch import nn
import numpy as np
from alphaholdem.model.hunl_supervise_range_model import HUNLSuperviseRangeModel
from alphaholdem.poker.no_limit_texas_holdem_env import NoLimitTexasHoldemEnv
from alphaholdem.poker.component.card import Card
from alphaholdem.poker.component.action import ActionType
from alphaholdem.poker.component.observation import Observation
from alphaholdem.poker.component.street import Street
from alphaholdem.poker.utils.format_utils import deepstack_to_ppo_strategy, trim_prob
from copy import deepcopy

def get_strategy(env: NoLimitTexasHoldemEnv, model: HUNLSuperviseRangeModel) -> tuple[np.ndarray, np.ndarray]:
    observation = env.observe_current()

    cards = torch.from_numpy(observation['observation'][np.newaxis, 1:, :]).to('cuda')
    actions = torch.from_numpy(observation['action_history'][np.newaxis, :]).to('cuda')
    action_mask = observation['action_mask']

    can_check = action_mask[1] > 0.5

    prob: np.ndarray = model(cards, actions).detach().cpu().numpy().reshape((1326, 4))
    empty = np.zeros((1326), dtype=np.float32)
    if can_check:
        prob = np.stack((prob[:, 0], prob[:, 1], empty, prob[:, 3], prob[:, 2]), axis=1)
    else:
        prob = np.stack((prob[:, 0], empty, prob[:, 1], prob[:, 3], prob[:, 2]), axis=1)
    return actions.detach().cpu().numpy(), trim_prob(deepstack_to_ppo_strategy(prob))[np.newaxis, :]

def dfs(env: NoLimitTexasHoldemEnv, model: HUNLSuperviseRangeModel) -> tuple[np.ndarray, np.ndarray]:
    game_obs = env.game.observe_current()
    if game_obs.street != Street.Preflop or game_obs.is_over:
        return None, None
    actions, prob = get_strategy(env, model)
    for i, action in enumerate(game_obs.legal_actions):
        if action is not None:
            c_env = deepcopy(env)
            c_env.step(i)
            c_actions, c_prob = dfs(c_env, model)
            if c_actions is not None:
                actions = np.concatenate((actions, c_actions))
                prob = np.concatenate((prob, c_prob))
    return actions, prob

if __name__ == '__main__':
    model_path = 'checkpoint/showdown/range_preflop.pt'
    model: HUNLSuperviseRangeModel = HUNLSuperviseRangeModel()
    model.load_state_dict(torch.load(model_path))
    model.to('cuda')
    model.eval()
    env = NoLimitTexasHoldemEnv(
        num_players=2,
        initial_chips=200,
        showdown_street=Street.Showdown,
        custom_board_cards=Card.from_str_list([]),
        raise_pot_size=[1],
        legal_raise_pot_size=[1],
    )
    env.reset()
    print('dfs')
    actions, prob = dfs(env, model)
    print('dfs done')
    print(actions.shape, prob.shape)
    save_folder = './strategy/hunl/simple'
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, 'actions.npy'), actions)
    np.save(os.path.join(save_folder, 'prob.npy'), prob)
