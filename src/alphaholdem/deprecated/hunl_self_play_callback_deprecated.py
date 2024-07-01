from collections import deque
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ..policy.hunl.policy import RandomHeuristic
from ..utils.logger import log
from rich import print_json

def create_hunl_self_play_callback(
    win_rate_threshold: float,
    opponent_policies: list[str],
    opponent_count: int = 10,
    min_update_step_count: int = 20,
    max_update_step_count: int = 200,
    payoff_max: float = 200,
) -> None:
    class SelfPlayCallback(DefaultCallbacks):
        def __init__(self):
            super().__init__()
            self.step_count = 0
            self.update_step_count = 0
            self.learned_version = 0
            self.opponent_count = opponent_count
            self.opponent_policies = deque(opponent_policies, maxlen=opponent_count)
            self.win_rate_threshold = win_rate_threshold
            self.initial_policies = {
                "random": RandomHeuristic,
            }
            self.policy_to_remove = None
            self.win_rate_buffer = []
            self.win_rate_buffer_size = 7

        def select_policy(self, agent_id: str, episode: EpisodeV2, **kwargs):
            return ("learned" if episode.episode_id % 2 == int(agent_id.split('_')[-1])
                else np.random.choice(list(self.opponent_policies)))

        def add_policy(self, algorithm: Algorithm, policy_id: str, policy_cls, policy_state = None) -> str:
            policy_to_remove = None
            log.info(f"Iter={algorithm.iteration} Adding new opponent to the mix ({policy_id})")
            if len(self.opponent_policies) == self.opponent_policies.maxlen:
                policy_to_remove = self.opponent_policies[0]
            self.opponent_policies.append(policy_id)
            policy = algorithm.add_policy(
                policy_id=policy_id,
                policy_cls=policy_cls,
                policy_mapping_fn=self.select_policy,
            )
            if policy_state is not None:
                policy.set_state(policy_state)
            algorithm.workers.sync_weights()
            return policy_to_remove

        def remove_policy(self, algorithm: Algorithm, policy_id: str):
            log.info(f"Remove {policy_id} from opponent policies")
            algorithm.remove_policy(
                policy_id=policy_id,
                policy_mapping_fn=self.select_policy,
            )

        def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs):
            main_reward = result["hist_stats"].pop("policy_learned_reward")
            win_rate = sum(main_reward) / len(main_reward) * 50.0 * payoff_max
            self.win_rate_buffer.append(max(win_rate, -100))
            if len(self.win_rate_buffer) > self.win_rate_buffer_size:
                self.win_rate_buffer.pop(0)
            result["win_rate"] = win_rate
            win_rate_smooth = sum(self.win_rate_buffer) / len(self.win_rate_buffer)
            result["win_rate_smooth"] = win_rate_smooth
            result["learned_version"] = self.learned_version
            log.info(f"Iter={algorithm.iteration} win-rate={win_rate} policies={len(self.opponent_policies)}")

            self.update_step_count += 1
            self.step_count += 1
            min_threshold = min_update_step_count * 2 if self.step_count <= 200 else min_update_step_count

            if (self.step_count <= 600 and self.update_step_count > min_update_step_count) or \
               (self.step_count > 600 and self.update_step_count > max_update_step_count):
            # if (win_rate_smooth > self.win_rate_threshold and self.update_step_count > min_threshold) \
            #     or self.update_step_count > max_update_step_count:
                not_exist_initial_policies = [
                    policy for policy in self.initial_policies.keys()
                    if policy not in self.opponent_policies
                ]
                print("Not exist:", len(not_exist_initial_policies))
                if len(not_exist_initial_policies) > 0 and np.random.random() < 0.0:
                    policy_id = np.random.choice(not_exist_initial_policies)
                    self.policy_to_remove = self.add_policy(algorithm, policy_id, self.initial_policies[policy_id])
                else:
                    self.learned_version += 1
                    policy_id = f'learned_v{self.learned_version}'
                    self.update_step_count = 0
                    self.policy_to_remove = self.add_policy(
                        algorithm=algorithm,
                        policy_id=policy_id,
                        policy_cls=type(algorithm.get_policy('learned')),
                        policy_state=algorithm.get_policy('learned').get_state(),
                    )
                    algorithm.workers.sync_weights()
            else:
                log.info("Not good enough... Keep learning ...")

            result["league_size"] = len(self.opponent_policies) + 1

        def on_evaluate_end(self, *, algorithm: Algorithm, evaluation_metrics: dict, **kwargs):
            if self.policy_to_remove is not None:
                self.remove_policy(algorithm, self.policy_to_remove)
                self.policy_to_remove = None

    return SelfPlayCallback
