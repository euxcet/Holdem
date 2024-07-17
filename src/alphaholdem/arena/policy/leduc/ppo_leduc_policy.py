from __future__ import annotations

import numpy as np
from copy import deepcopy
from typing_extensions import override
from ..ppo_poker_policy import PPOPokerPolicy
from ....poker.component.card import Card
from ....poker.component.observation import Observation

class PPOLeducPolicy(PPOPokerPolicy):
    @override
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> np.ndarray:
        env_obs_list = []
        for hole_card in Card.from_str_list(['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']):
            obs = deepcopy(env_obs)
            obs['observation'][0] = np.zeros((4, 13))
            obs['observation'][0][hole_card.suit][hole_card.rank] = 1.0
            env_obs_list.append(obs)
        return np.around(np.array(self.get_policies(env_obs_list)), 2)

    def _str_to_rank(self, c: str) -> int:
        if c == 'J':
            return 9
        elif c == 'Q':
            return 10
        else:
            return 11

    def _create_env_obs(self, history: str):
        # Fold Check Call Raise
        max_num_actions_street = 6
        observation = np.zeros((4, 4, 13)).astype(np.float32)
        # Fold Check Call Raise All_in
        action_history = np.zeros((4, max_num_actions_street * 2, 5)).astype(np.float32)
        action_mask = np.zeros((4,)).astype(np.int8)
        observation[0][0][self._str_to_rank(history[0])] = 1.0
        if history[1] in ['J', 'Q', 'K']:
            observation[1][0][self._str_to_rank(history[1])] = 1.0
        street = -1
        player = 0
        num_action = [0, 0]
        num_raise = 0
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

        return {
            'observation': observation,
            'action_history': action_history,
            'action_mask': action_mask,
        }

    @override
    def get_all_policy(self, keys: list[str]) -> np.ndarray:
        result = self.get_policies([self._create_env_obs(key) for key in keys], None)
        # fold check call raise -> raise check/call fold
        strategy = np.zeros((len(keys), 3)).astype(np.float32)
        for i in range(len(keys)):
            strategy[i][0] = result[i][3]
            strategy[i][1] = result[i][1] + result[i][2]
            strategy[i][2] = result[i][0]
        return strategy


    @staticmethod
    def load_policy_from_run(run_folder: str) -> PPOLeducPolicy:
        return PPOLeducPolicy(model_path=PPOPokerPolicy._load_all_model_path(run_folder)[-1])

    @staticmethod
    def load_policies_from_run(run_folder: str) -> list[PPOLeducPolicy]:
        policies = []
        for model_path in PPOLeducPolicy._load_all_model_path(run_folder):
            policies.append(PPOLeducPolicy(model_path=model_path))
        return policies