import numpy as np
from ..heuristic_base import HeuristicBase

class RangeKuhnCFRHeuristic(HeuristicBase):
    def __init__(self, *args, **kwargs):
        if 'path' not in kwargs:
            self.strategy = self.load_strategy('./strategy/kuhn_nash.txt')
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

    def get_from_strategy(self, action_history) -> np.ndarray:
        history = ':'
        for i in range(2):
            if action_history[i][1] > 0.5 or action_history[i][2] > 0.5:
                history += 'c'
            if action_history[i][3] > 0.5:
                history += 'r'
        result = []
        for hole_card in ['J', 'Q', 'K']:
            prob = self.strategy[hole_card + history]
            result.append(prob[0])
        return np.array(result)

    def _do_compute_actions(self, obs_batch):
        return [
            self.get_from_strategy(action_history) for action_history in obs_batch['action_history']
        ], [], {}

class RangeKuhnRandomHeuristic(HeuristicBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _do_compute_actions(self, obs_batch):
        return [
            np.array([0.5, 0.5, 0.5]) for _ in obs_batch['action_history']
        ], [], {}
