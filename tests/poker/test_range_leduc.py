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
        payoff_max = 1
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
            path='/home/clouduser/zcc/Holdem/strategy/leduc_ppo.txt',
        )
        tot = [0, 0]
        num = [0, 0]
        l = 1000000
        for i in range(l):
            if i % 1000 == 10:
                print(round(i / l, 2), tot[0] / num[0] * 50 * payoff_max, tot[1] / num[1] * 50 * payoff_max)
            env.reset()
            agent = i % 2
            while not env.is_over():
                obs = self._observe(env)
                if env.current_agent_id() % 2 == agent:
                    action = cfr_policy._do_compute_actions(obs)[0][0]
                else:
                    action = random_policy._do_compute_actions(obs)[0][0]
                env.step(action)
            tot[agent] += env._cumulative_rewards['agent_' + str(agent)]
            num[agent] += 1
        #     print(env._cumulative_rewards)
        #     print(env.game.log_action)
        #     print("\n")
        # print()
        print(tot[0] / num[0] * 50 * payoff_max, tot[1] / num[1] * 50 * payoff_max)
# 12.59449995986941 28.82889007740005
# 20.711695018634728 0