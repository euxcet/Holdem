import pytest
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from alphaholdem.policy.leduc.range_policy import RangeLeducCFRHeuristic, RangeLeducRandomHeuristic
from alphaholdem.poker.range_limit_leduc_holdem import RangeLimitLeducHoldem
from alphaholdem.poker.range_limit_leduc_holdem_env import RangeLimitLeducHoldemEnv

class TestRangeLeduc():
    SKIP = True

    def _observe(self, env: RangeLimitLeducHoldemEnv) -> dict:
        obs = env.observe_current()
        return {
            'action_history': [obs['action_history']],
            'action_mask': [obs['action_mask']],
            'board_card': [obs['board_card']],
        }

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_range_leduc(self):
        payoff_max = 10
        env = RangeLimitLeducHoldemEnv(payoff_max=payoff_max)
        cfr_policy = RangeLeducCFRHeuristic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=AlgorithmConfig(),
        )
        # random_policy = RangeLeducRandomHeuristic(
        #     observation_space=env.observation_space,
        #     action_space=env.action_space,
        #     config=AlgorithmConfig(),
        # )
        random_policy = RangeLeducCFRHeuristic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=AlgorithmConfig(),
            path='/home/clouduser/zcc/Holdem/strategy/bug.txt',
        )
        tot = 0
        l = 10
        for i in range(l):
            env.reset()
            while not env.is_over():
                obs = self._observe(env)
                if env.current_agent_id() % 2 == i % 2:
                    action = cfr_policy._do_compute_actions(obs)[0][0]
                else:
                    action = random_policy._do_compute_actions(obs)[0][0]
                env.step(action)
            print(env._cumulative_rewards['agent_' + str(i % 2)], end = ' ')
            tot += env._cumulative_rewards['agent_' + str(i % 2)]
        print()
        print(tot / l * 50 * payoff_max)