from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ..utils.logger import log

def create_leduc_play_with_best_callback(
    win_rate_threshold: float,
    payoff_max: float = 200,
) -> None:
    class SelfPlayCallback(DefaultCallbacks):
        def __init__(self):
            super().__init__()
            self.step_count = 0
            self.update_step_count = 0
            self.win_rate_threshold = win_rate_threshold
            self.policy_to_remove = None
            self.win_rate_buffer = []
            self.win_rate_buffer_size = 7

        def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs):
            main_reward = result["hist_stats"].pop("policy_learned_reward")
            win_rate = sum(main_reward) / len(main_reward) * 50.0 * payoff_max
            self.win_rate_buffer.append(max(win_rate, -100))
            if len(self.win_rate_buffer) > self.win_rate_buffer_size:
                self.win_rate_buffer.pop(0)
            result["win_rate"] = win_rate
            win_rate_smooth = sum(self.win_rate_buffer) / len(self.win_rate_buffer)
            result["win_rate_smooth"] = win_rate_smooth
            log.info(f"Iter={algorithm.iteration} win-rate={win_rate}")

        def on_evaluate_end(self, *, algorithm: Algorithm, evaluation_metrics: dict, **kwargs):
            ...
    return SelfPlayCallback
