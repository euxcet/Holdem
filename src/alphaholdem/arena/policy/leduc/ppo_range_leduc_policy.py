from __future__ import annotations

import torch
import numpy as np
from copy import deepcopy
from typing_extensions import override
from ..ppo_poker_policy import PPOPokerPolicy
from ....poker.component.card import Card
from ....poker.component.observation import Observation

class PPORangeLeducPolicy(PPOPokerPolicy):
    @override
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> np.ndarray:
        env_obs_list = []
        for hole_card in Card.from_str_list(['Js', 'Js', 'Qs', 'Qs', 'Ks', 'Ks']):
            obs = deepcopy(env_obs)
            obs['observation'][0] = np.zeros((4, 13))
            obs['observation'][0][hole_card.suit][hole_card.rank] = 1.0
            env_obs_list.append(obs)
        return np.around(np.array(self.get_policies(env_obs_list)), 2)

    def _create_observation(self, history: str):
        max_num_actions_street = 2
        # Fold Check Call Raise All_in
        action_history = np.zeros((2, max_num_actions_street * 2, 5)).astype(np.float32)
        action_mask = np.zeros((4,)).astype(np.int8)
        board_card = np.zeros((3,)).astype(np.float32)
        street = -1
        player = 0
        num_action = [0, 0]
        num_raise = 0
        if history[1] in ['J', 'Q', 'K']:
            board_card[Card.from_str(history[1] + 's').rank - 9] = 1.0
        for action in history[:-1]:
            if action == '/':
                street += 1
                player = 0
                num_action = [0, 0]
                num_raise = 0
                continue
                
            if street >= 0:
                if action == 'c' and num_raise == 0: # check
                    action_history[street][player * max_num_actions_street + num_action[player]][1] = 1.0
                elif action == 'c' and num_raise > 0: # call
                    action_history[street][player * max_num_actions_street + num_action[player]][2] = 1.0
                elif action == 'r': # raise
                    num_raise += 1
                    action_history[street][player * max_num_actions_street + num_action[player]][3] = 1.0
                num_action[player] += 1
                player = 1 - player
        if num_raise > 0:
            action_mask[0] = 1
            action_mask[2] = 1
        if num_raise == 0:
            action_mask[1] = 1
        if num_raise < 2:
            action_mask[3] = 1
        return player, action_history, action_mask, board_card

    @override
    def get_policy(self, env_obs: list[dict]) -> np.ndarray:
        obs = {
            'obs': {
                'ranges': torch.from_numpy(env_obs['ranges'][np.newaxis, :]).to('cuda'),
                'action_history': torch.from_numpy(env_obs['action_history'][np.newaxis, :]).to('cuda'),
                'action_mask': torch.from_numpy(env_obs['action_mask'][np.newaxis, :]).to('cuda'),
                'board_card': torch.from_numpy(env_obs['board_card'][np.newaxis, :]).to('cuda'),
            }
        }
        return self.model(obs)[0].detach().cpu().numpy()

    def _policy_to_prob(self, policy: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        action_prob = policy.squeeze()[:12].copy() # discard vars
        for i in range(3):
            if sum(action_prob[i * 4 : (i + 1) * 4] * action_mask) < 1e-5:
                action_prob[i * 4 : (i + 1) * 4] = action_mask.astype(np.float32)
            s = sum(action_prob[i * 4 : (i + 1) * 4] * action_mask)
            action_prob[i * 4 : (i + 1) * 4] *= action_mask.astype(np.float32) / s
        return action_prob

    @override
    def get_all_policy(self, keys: list[str]) -> np.ndarray:
        all_ranges = {
            '?:/:': np.ones((2, 3)).astype(np.float32),
        }
        strategy = np.zeros((len(keys), 3)).astype(np.float32)
        for id, key in enumerate(keys):
# JJ:/crrc/r: 0.9919999837875366 0.00800000037997961 0.0
            player, action_history, action_mask, board_card = self._create_observation(key)
            range_key = '?' + key[1:]
            ranges = all_ranges[range_key] # remove the hole card
            prob = self._policy_to_prob(self.get_policy({
                'ranges': ranges,
                'action_history': action_history,
                'action_mask': action_mask,
                'board_card': board_card,
            }), action_mask)
            #  0 J_fold  1 J_check  2 J_call  3 J_raise
            #  4 Q_fold  5 Q_check  6 Q_call  7 Q_raise
            #  8 K_fold  9 K_check 10 K_call 11 K_raise

            offset = (Card.from_str(key[0] + 's').rank - 9) * 4
            strategy[id][0] = round(prob[offset + 3], 3)
            strategy[id][1] = round(prob[offset + 1] + prob[offset + 2], 3)
            strategy[id][2] = round(prob[offset + 0], 3)

            # if key in ['JJ:/crrc/r:', 'JJ:/rc/r:']:
            #     print(key, player, action_history, action_mask, board_card)
            #     print(prob, strategy[id])

            if action_mask[1] == 1: # check
                check_ranges = ranges.copy()
                if range_key[-2] == 'c' and range_key[1] == ':': # check check
                    check_ranges[player][0] *= prob[1] # J
                    check_ranges[player][1] *= prob[5] # Q
                    check_ranges[player][2] *= prob[9] # K
                    for board_card in ['J', 'Q', 'K']:
                        check_key = range_key[0] + board_card + range_key[1:-1] + 'c/:'
                        all_ranges[check_key] = check_ranges
                elif range_key[-2] != 'c': # first check in the street
                    check_key = range_key[:-1] + 'c:'
                    check_ranges[player][0] *= prob[1] # J
                    check_ranges[player][1] *= prob[5] # Q
                    check_ranges[player][2] *= prob[9] # K
                    all_ranges[check_key] = check_ranges

            if action_mask[2] == 1 and range_key[1] == ':': # call
                call_ranges = ranges.copy()
                call_ranges[player][0] *= prob[2] # J
                call_ranges[player][1] *= prob[6] # Q
                call_ranges[player][2] *= prob[10] # K
                for board_card in ['J', 'Q', 'K']:
                    call_key = range_key[0] + board_card + range_key[1:-1] + 'c/:'
                    all_ranges[call_key] = call_ranges

            if action_mask[3] == 1: # raise
                raise_ranges = ranges.copy()
                raise_key = range_key[:-1] + 'r:'
                raise_ranges[player][0] *= prob[3] # J
                raise_ranges[player][1] *= prob[7] # Q
                raise_ranges[player][2] *= prob[11] # K
                all_ranges[raise_key] = raise_ranges
            
        f = open('/home/clouduser/zcc/Holdem/strategy/leduc_ppo.txt', 'w')
        for i in range(len(keys)):
            f.write('{} {} {} {}\n'.format(keys[i], strategy[i][0], strategy[i][1], strategy[i][2]))

        return strategy


    @staticmethod
    def load_policy_from_run(run_folder: str) -> PPORangeLeducPolicy:
        return PPORangeLeducPolicy(model_path=PPOPokerPolicy._load_all_model_path(run_folder)[-1])

    @staticmethod
    def load_policies_from_run(run_folder: str) -> list[PPORangeLeducPolicy]:
        policies = []
        for model_path in PPORangeLeducPolicy._load_all_model_path(run_folder):
            policies.append(PPORangeLeducPolicy(model_path=model_path))
        return policies