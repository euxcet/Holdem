import numpy as np
from typing_extensions import override
from .policy import Policy
from ...tensorflow.predictor import Predictor
from ...poker.component.observation import Observation

class TFTexasPolicy(Policy):
    def __init__(self, model_path: str = None, device: str = 'cuda') -> None:
        self.model = Predictor()
        self.model.init_model(model_path=model_path)

    # Fold Check Call All_in Raise_50% Raise_75% Raise_100% Raise_125% Raise_150%
    # 0    1     2    3      4         5         6          7          8
    @override
    def sample_action(self, env_obs: dict, game_obs: Observation) -> int:
        # 4 4 13
        cards = env_obs['observation']
        # 4 12 5
        action_history = env_obs['action_history']
        # 9
        action_mask = env_obs['action_mask']
        tf_cards = np.zeros((4, 13, 6))
        tf_cards[:, :, 0] = cards[0]
        tf_cards[:, :, 1] = cards[1]
        tf_cards[:, :, 2] = cards[2]
        tf_cards[:, :, 3] = cards[3]
        tf_cards[:, :, 4] = cards[1] + cards[2] + cards[3]
        tf_cards[:, :, 5] = cards[0] + cards[1] + cards[2] + cards[3]
        tf_action_history = np.zeros((4, 4, 9, 13))
        street_action_num = [0] * 4
        for action in game_obs.log_action:
            if action.type.value < 3:
                action_id = action.type.value
            elif action.type.value == 4: # all in
                action_id = 8
            else: # raise
                if action.raise_pot < 0.4: # 0.25
                    action_id = 3
                elif action.raise_pot >= 0.4 and action.raise_pot < 0.6: # 0.5
                    action_id = 4
                elif action.raise_pot >= 0.6 and action.raise_pot < 0.9 : # 0.75
                    action_id = 5
                elif action.raise_pot >= 0.9 and action.raise_pot < 1.5: # 1.0
                    action_id = 6
                else: # 2.0
                    action_id = 7
            street = action.street.value
            player = action.player
            tf_action_history[street][player][action_id][street_action_num[street]] = 1
            tf_action_history[street][2][action_id][street_action_num[street]] = 1
            street_action_num[street] += 1
        all_in = action_mask[3]
        action_mask[3:8] = action_mask[4:9]
        action_mask[8] = all_in
        tensor = np.concatenate((tf_cards.flatten(), tf_action_history.flatten(), action_mask))
        action = self.model.get_sample_action(tensor)[0]
        if action == 8:
            action = 3
        elif action >= 3:
            action += 1
        return action
    
    @override
    def get_policy(self, env_obs: dict, game_obs: Observation) -> list[float]:
        ...

    @override
    def get_range_policy(self, env_obs: dict, game_obs: Observation) -> list[float]:
        ...