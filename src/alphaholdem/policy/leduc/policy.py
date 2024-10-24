import os
import numpy as np
from ..heuristic_base import HeuristicBase

class LeducCFRHeuristic(HeuristicBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = self.load_strategy(args[2]['nash'])

    def load_strategy(self, save_file: str) -> dict[str, list[float]]:
        result = dict()
        with open(save_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                s = line.strip().split(' ')
                result[s[0]] = list(map(float, s[1:]))
        return result

    def get_from_strategy(self, observation, action_history, action_mask):
        history = ''.join(map(lambda x: 'J' if x[2] == 9 else ('Q' if x[2] == 10 else 'K'), np.argwhere(observation > 0.5))) + ':/'

        actions:list[list[int]] = np.argwhere(action_history > 0.05).tolist()
        actions.sort(key=lambda x: x[0] * 100 + (x[1] % 6) * 10 + x[1] // 6)
        street_action = 0
        for index, action in enumerate(actions):
            if index > 0 and action[0] != actions[index - 1][0]:
                history += '/'
                street_action = 0
            if action[2] == 3:
                history += 'r'
                street_action += 1
            else:
                history += 'c'
                street_action += 1
        if history[-1] == 'c' and street_action > 1:
            history += '/'
        history += ':'
        if history not in self.strategy:
            print('history', actions, action_mask, history)
        action = np.random.choice(3, p=self.strategy[history])
        if action == 0:
            action = 3
        elif action == 2:
            action = 0
        else:
            action = 1 if action_mask[1] == 1 else 2
        return action

    def _do_compute_actions(self, obs_batch):
        return [self.get_from_strategy(x, y, z) for x, y, z in zip(obs_batch['observation'], obs_batch['action_history'], obs_batch["action_mask"])], [], {}
