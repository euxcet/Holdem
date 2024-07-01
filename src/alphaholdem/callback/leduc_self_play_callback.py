from collections import deque
import queue
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ..policy.hunl.policy import RandomHeuristic
from ..utils.logger import log
from rich import print_json
from ..arena.arena import LeducArena
from ..arena.envs.limit_leduc_holdem_env import create_limit_holdem_env
from ..arena.policy.ppo_leduc_policy import PPOLeducPolicy
from ..arena.policy.cfr_leduc_policy import CFRLeducPolicy

def create_leduc_self_play_callback(
    win_rate_threshold: float,
    opponent_policies: list[str],
    opponent_count: int = 10,
    num_update_iter: int = 20,
    payoff_max: float = 200,
    cfr: CFRLeducPolicy = None,
) -> None:
    class SelfPlayCallback(DefaultCallbacks):
        def __init__(self):
            super().__init__()
            self.update_iter_count = 0
            self.opponent_count = opponent_count
            self.opponent_policies = deque(opponent_policies, maxlen=opponent_count)
            self.current_opponent_id = 0
            self.win_rate_threshold = win_rate_threshold
            self.policy_to_remove = None
            self.win_rate_buffer = []
            self.win_rate_buffer_size = 7

        def select_policy(self, agent_id: str, episode: EpisodeV2, **kwargs):
            return ("learned" if episode.episode_id % 2 == int(agent_id.split('_')[-1])
                else np.random.choice(list(self.opponent_policies)))

        def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs):
            main_reward = result["hist_stats"].pop("policy_learned_reward")
            win_rate = sum(main_reward) / len(main_reward) * 50.0 * payoff_max
            self.win_rate_buffer.append(max(win_rate, -100))
            if len(self.win_rate_buffer) > self.win_rate_buffer_size:
                self.win_rate_buffer.pop(0)
            result["win_rate"] = win_rate
            win_rate_smooth = sum(self.win_rate_buffer) / len(self.win_rate_buffer)
            result["win_rate_smooth"] = win_rate_smooth
            mean, var = LeducArena().ppo_vs_cfr(
                ppo=PPOLeducPolicy(model=algorithm.get_policy('learned').model),
                cfr=cfr,
                runs=4096,
            )
            result["win_rate_vs_cfr"] = mean
            log.info(f"Iter={algorithm.iteration} win_rate={win_rate}")
            self.update_iter_count += 1
            if self.update_iter_count % num_update_iter == 0:
                policy_id = 'opponent_' + str(self.current_opponent_id % self.opponent_count)
                if self.current_opponent_id < self.opponent_count:
                    self.opponent_policies.append(policy_id)
                    algorithm.add_policy(
                        policy_id=policy_id,
                        policy_cls=type(algorithm.get_policy('learned')),
                        policy_mapping_fn=self.select_policy,
                    )
                algorithm.get_policy(policy_id).set_state(algorithm.get_policy('learned').get_state())
                algorithm.workers.sync_weights()
                self.current_opponent_id += 1
            result["learned_version"] = self.current_opponent_id

    return SelfPlayCallback
