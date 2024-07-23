import numpy as np
from ..heuristic_base import HeuristicBase

class RangeLeducCFRHeuristic(HeuristicBase):
    def __init__(self, *args, **kwargs):
        if 'path' not in kwargs:
            self.strategy = self.load_strategy('/home/clouduser/zcc/Holdem/strategy/leduc_nash.txt')
        else:
            self.strategy = self.load_strategy(kwargs['path'])
            kwargs.pop('path')
        super().__init__(*args, **kwargs)

    def load_strategy(self, save_file: str) -> dict[str, list[float]]:
        result = dict()
        with open(save_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                s = line.strip().split(' ')
                result[s[0]] = list(map(float, s[1:]))
        return result

    def get_from_strategy(self, action_history, action_mask, board_card) -> list[float]:
        history = ':/'
        board = ''
        if board_card[0] == 1:
            board = 'J'
        elif board_card[1] == 1:
            board = 'Q'
        elif board_card[2] == 1:
            board = 'K'

        actions: list[list[int]] = np.argwhere(action_history > 0.05).tolist()
        actions.sort(key=lambda x: x[0] * 100 + (x[1] % 2) * 10 + x[1] // 2)
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

        result = []
        for hole_card in ['J', 'Q', 'K']:
            prob = self.strategy[hole_card + board + history]
            result.append(prob[2] if action_mask[0] == 1 else 0) # fold
            result.append(prob[1] if action_mask[1] == 1 else 0) # check
            result.append(prob[1] if action_mask[2] == 1 else 0) # call
            result.append(prob[0] if action_mask[3] == 1 else 0) # raise
        print(result)

        return result

    def _do_compute_actions(self, obs_batch):
        return [
            self.get_from_strategy(action_history, action_mask, board_card)
                for action_history, action_mask, board_card in zip(
                    obs_batch['action_history'],
                    obs_batch["action_mask"],
                    obs_batch["board_card"],
                )
        ], [], {}

class RangeLeducRandomHeuristic(HeuristicBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _do_compute_actions(self, obs_batch):
        return [
            # ([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
            ([0.25] * 12)
                for action_history, action_mask, board_card in zip(
                    obs_batch['action_history'],
                    obs_batch["action_mask"],
                    obs_batch["board_card"],
                )
        ], [], {}
