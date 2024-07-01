from .heuristic_base import HeuristicBase

class RandomHeuristic(HeuristicBase):
    def _do_compute_actions(self, obs_batch):
        return [self.pick_legal_action(x) for x in obs_batch["action_mask"]], [], {}
