import numpy as np
from abc import ABC, abstractmethod
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ..arena.arena import Arena
from ..arena.policy.policy import Policy
from ..utils.logger import log
from ..utils.window import Window
from ..utils.counter import Counter

class SelfPlayCallback(DefaultCallbacks, ABC):
    POLICY_TO_LEARN = "learned"
    OPPONENT_PREFIX = "opponent_"
    WIN_RATE_WINDOW_SIZE = 6

    def __init__(
        self,
        num_opponent_limit: int,
        num_update_iter: int,
        arena: Arena,
        arena_runs: int,
        payoff_max: float,
        rule_based_policies: list[str],
        policy_type: type,
    ) -> None:
        super().__init__()
        self.rule_based_policies = rule_based_policies
        self.opponent_policies = Window(num_opponent_limit)
        self.opponent_circular_pointer = 0
        self.win_rate_window = Window(self.WIN_RATE_WINDOW_SIZE)
        self.update_counter = Counter(num_update_iter)
        self.num_opponent_limit = num_opponent_limit
        self.arena = arena
        self.arena_runs = arena_runs
        self.payoff_max = payoff_max
        self.policy_type = policy_type
        self.learned_version = 0

    def _get_policy_id(self, id: int) -> str:
        return self.OPPONENT_PREFIX + str(id)

    def select_policy(self, agent_id: str, episode: EpisodeV2, **kwargs) -> str:
        if self.opponent_policies.capacity() == 0:
            return self.POLICY_TO_LEARN
        return (self.POLICY_TO_LEARN if episode.episode_id % 2 == int(agent_id.split('_')[-1])
            else np.random.choice(self.opponent_policies.window))

    def add_policy(self, algorithm: Algorithm) -> None:
        policy_id = self._get_policy_id(self.opponent_policies.capacity())
        self.opponent_policies.push(policy_id)
        algorithm.add_policy(
            policy_id=policy_id,
            policy_cls=type(algorithm.get_policy(self.POLICY_TO_LEARN)),
            policy_mapping_fn=self.select_policy,
        )
        algorithm.workers.sync_weights()
        self.learned_version += 1

    def replace_policy(self, algorithm: Algorithm, policy_id: str) -> None:
        algorithm.get_policy(policy_id).set_state(algorithm.get_policy(self.POLICY_TO_LEARN).get_state())
        algorithm.workers.sync_weights()
        self.learned_version += 1

    @abstractmethod
    def new_policy(self, algorithm: Algorithm) -> None:
        ...

    def calc_metric(self, result: dict, policy: Policy) -> None:
        main_reward = result["hist_stats"].pop("policy_learned_reward")
        win_rate = sum(main_reward) / len(main_reward) * 50.0 * self.payoff_max
        self.win_rate_window.push(win_rate)
        result["win_rate"] = win_rate
        result["win_rate_smooth"] = self.win_rate_window.average()
        if self.arena.nash_policy is not None:
            mean, var = self.arena.policy_vs_policy(
                policy0=self.current_policy,
                policy1=self.arena.nash_policy,
                runs=self.arena_runs,
            )
            result['win_rate_vs_nash'] = mean
            result['win_rate_vs_nash_var'] = var

    def log_result(self, algorithm: Algorithm, result: dict, policy: Policy) -> None:
        log.info(f"Iter={algorithm.iteration} win_rate={result['win_rate']}")
        policy.log()

    def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs) -> None:
        self.current_policy = self.policy_type(model=algorithm.get_policy(self.POLICY_TO_LEARN).model)
        self.calc_metric(result, self.current_policy)
        self.log_result(algorithm, result, self.current_policy)
        if self.update_counter.count():
            self.new_policy(algorithm)
        result["learned_version"] = self.learned_version
