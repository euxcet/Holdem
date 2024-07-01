import numpy as np
import random
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import restore_original_dimensions


class HeuristicBase(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = self._create_exploration()

    def learn_on_batch(self, samples):
        pass

    @override(Policy)
    def get_weights(self):
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights):
        """No weights to set."""
        pass

    @override(Policy)
    def export_model(self, export_dir: str, onnx=None) -> None:
        pass

    @override(Policy)
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        obs_batch = restore_original_dimensions(
            np.array(obs_batch, dtype=np.float32), self.observation_space, tensorlib=np
        )
        return self._do_compute_actions(obs_batch)

    def pick_legal_action(self, legal_action):
        legal_choices = np.arange(len(legal_action))[legal_action == 1]
        return np.random.choice(legal_choices)

class RandomHeuristic(HeuristicBase):
    def _do_compute_actions(self, obs_batch):
        return [self.pick_legal_action(x) for x in obs_batch["action_mask"]], [], {}

class BroadwayHeuristic(HeuristicBase):
    _rand_choice = random.choice(range(7))

    def _do_compute_actions(self, obs_batch):
        def select_action(observation):
            # 4(hole, flop, turn, river) * 4 * 13
            if observation[0, :, 8:].sum() == 2:
                return 1
            return 0

        return [select_action(x) for x in obs_batch["observation"]], [], {}

class AlwaysAllInHeuristic(HeuristicBase):
    def _do_compute_actions(self, obs_batch):
        return [1 for x in obs_batch["action_mask"]], [], {}

class AlwaysFoldHeuristic(HeuristicBase):
    def _do_compute_actions(self, obs_batch):
        return [0 for x in obs_batch["action_mask"]], [], {}