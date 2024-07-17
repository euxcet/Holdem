from typing_extensions import override
from ray.rllib.algorithms import Algorithm
from .self_play_callback import SelfPlayCallback
from ..arena.arena import Arena

class NaiveSelfPlayCallback(SelfPlayCallback):
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
        super().__init__(
            num_opponent_limit=num_opponent_limit,
            num_update_iter=num_update_iter,
            arena=arena,
            arena_runs=arena_runs,
            payoff_max=payoff_max,
            rule_based_policies=rule_based_policies,
            policy_type=policy_type
        )
        self.circular_pointer = 0
        self.best_metric = -100
        self.last_update = -10000

    @override
    def new_policy(self, algorithm: Algorithm, result: dict) -> None:
        # if algorithm.iteration > self.last_update + 50:
        #     self.last_update = algorithm.iteration
        #     if not self.opponent_policies.full():
        #         self.add_policy(algorithm)
        #     else:
        #         self.replace_policy(algorithm, self._get_policy_id(self.circular_pointer))
        #         self.circular_pointer = (self.circular_pointer + 1) % self.num_opponent_limit
        #     self.best_metric = -100
        # elif result['win_rate_vs_nash'] > self.best_metric:
        #     self.best_metric = result['win_rate_vs_nash']
        #     self.replace_policy(algorithm, self._get_policy_id(self.circular_pointer))

        if not self.opponent_policies.full():
            self.add_policy(algorithm)
        else:
            self.replace_policy(algorithm, self._get_policy_id(self.circular_pointer))
            self.circular_pointer = (self.circular_pointer + 1) % self.num_opponent_limit
