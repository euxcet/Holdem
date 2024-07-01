from collections import deque
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ..utils.logger import log
from rich import print_json
from ray.rllib.policy.policy import PolicySpec
from ..policy.connect4.policy import RandomHeuristic, AlwaysSameHeuristic, SmartHeuristic, BeatLastHeuristic, LinearHeuristic

def create_connect4_self_play_callback(win_rate_threshold: float, opponent_policies: list[str], opponent_count: int = 10):
    class SelfPlayCallback(DefaultCallbacks):
        def __init__(self):
            super().__init__()
            self.step_count = 0
            self.learned_version = 0
            self.opponent_count = opponent_count
            self.opponent_policies = deque(opponent_policies, maxlen=opponent_count)
            self.win_rate_threshold = win_rate_threshold
            self.initial_policies = {
                "random": RandomHeuristic,
                "always_same": AlwaysSameHeuristic,
                "beat_last": BeatLastHeuristic,
                "smart": SmartHeuristic,
                "linear": LinearHeuristic,
            }
            self.policy_to_remove = None

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
            self.step_count = 0
            return policy_to_remove

        def remove_policy(self, algorithm: Algorithm, policy_id: str):
            log.info(f"Remove {policy_id} from opponent policies")
            algorithm.remove_policy(
                policy_id=policy_id,
                policy_mapping_fn=self.select_policy,
            )

        def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs):
            main_reward = result["hist_stats"].pop("policy_learned_reward")
            opponent_reward = result["hist_stats"].pop("episode_reward")
            win_rate = sum(x > y for x, y in zip(main_reward, opponent_reward)) / len(main_reward)
            result["win_rate"] = win_rate
            result["learned_version"] = self.learned_version
            log.info(f"Iter={algorithm.iteration} win-rate={win_rate} policies={len(self.opponent_policies)}")

            self.step_count += 1
            if (win_rate > self.win_rate_threshold and self.step_count > 100) or self.step_count > 300:
            # if True:
                not_exist_initial_policies = [
                    policy for policy in self.initial_policies.keys()
                    if policy not in self.opponent_policies
                ]
                print("Not exist:", len(not_exist_initial_policies))
                if len(not_exist_initial_policies) > 0 and np.random.random() < 0.33:
                    policy_id = np.random.choice(not_exist_initial_policies)
                    self.policy_to_remove = self.add_policy(algorithm, policy_id, self.initial_policies[policy_id])
                else:
                    self.learned_version += 1
                    policy_id = f'learned_v{self.learned_version}'
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
