import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ..arena.policy.ppo_leduc_policy import PPOLeducPolicy
from ..utils.logger import log
from ..utils.window import Window
from ..utils.counter import Counter

def create_kuhn_self_play_callback(
    # cfr_strategy_checkpoint: str,
    opponent_policies: list[str],
    num_opponent_limit: int = 10,
    num_update_iter: int = 20,
    payoff_max: float = 200,
    win_rate_window_size: int = 7,
    arena_runs: int = 4096,
) -> None:
    class SelfPlayCallback(DefaultCallbacks):
        def __init__(self):
            super().__init__()
            self.num_opponent_limit = num_opponent_limit
            self.opponent_policies = Window(num_opponent_limit, opponent_policies)
            self.current_opponent_id = 0
            self.policy_to_remove = None
            self.win_rate_window = Window[float](win_rate_window_size)
            self.update_counter = Counter(num_update_iter)
            self.best_metric = -10000
            self.arena_runs = arena_runs

        def select_policy(self, agent_id: str, episode: EpisodeV2, **kwargs):
            if self.opponent_policies.capacity() == 0:
                return "learned"
            return ("learned" if episode.episode_id % 2 == int(agent_id.split('_')[-1])
                else np.random.choice(self.opponent_policies.window))

        def add_policy(self, algorithm: Algorithm):
            policy_id = 'opponent_' + str(self.current_opponent_id % self.num_opponent_limit)
            if self.current_opponent_id < self.num_opponent_limit:
                self.opponent_policies.push(policy_id)
                algorithm.add_policy(
                    policy_id=policy_id,
                    policy_cls=type(algorithm.get_policy('learned')),
                    policy_mapping_fn=self.select_policy,
                )
            algorithm.get_policy(policy_id).set_state(algorithm.get_policy('learned').get_state())
            algorithm.workers.sync_weights()
            self.current_opponent_id += 1

        def calc_metric(self, algorithm: Algorithm, result: dict):
            main_reward = result["hist_stats"].pop("policy_learned_reward")
            win_rate = sum(main_reward) / len(main_reward) * 50.0 * payoff_max
            self.win_rate_window.push(win_rate)
            result["win_rate"] = win_rate
            result["win_rate_smooth"] = self.win_rate_window.average()
            # mean, var = self.arena.ppo_vs_cfr(
            #     ppo=PPOLeducPolicy(model=algorithm.get_policy('learned').model),
            #     runs=self.arena_runs,
            # )
            # result['win_rate_vs_cfr'] = mean
            # result['win_rate_vs_cfr_var'] = var

        def calc_metric_for_update(self, algorithm: Algorithm, result: dict):
            return result['win_rate_vs_cfr']

        def log_metric(self, algorithm: Algorithm, result: dict):
            log.info(f"Iter={algorithm.iteration} win_rate={result['win_rate']}")

        def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs):
            self.calc_metric(algorithm, result)
            self.log_metric(algorithm, result)
            if self.update_counter.count():
                self.add_policy(algorithm)
            result["learned_version"] = self.current_opponent_id

    return SelfPlayCallback
