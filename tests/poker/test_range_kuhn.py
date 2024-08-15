import pytest
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
# from alphaholdem.policy.leduc.range_policy import RangeLeducCFRHeuristic, RangeLeducRandomHeuristic
from alphaholdem.policy.kuhn.range_policy import RangeKuhnCFRHeuristic, RangeKuhnRandomHeuristic
from alphaholdem.poker.range_kuhn_poker import RangeKuhnPoker
from alphaholdem.poker.range_kuhn_poker_env import RangeKuhnPokerEnv

class TestRangeKuhn():
    SKIP = True

    def _observe(self, env: RangeKuhnPokerEnv) -> dict:
        obs = env.observe_current()
        return {
            'action_history': [obs['action_history']],
        }

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_range_kuhn(self):
        payoff_max = 0.5
        env = RangeKuhnPokerEnv(payoff_max=payoff_max)
        cfr_policy = RangeKuhnCFRHeuristic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=AlgorithmConfig(),
        )
        # random_policy = RangeKuhnRandomHeuristic(
        #     observation_space=env.observation_space,
        #     action_space=env.action_space,
        #     config=AlgorithmConfig(),
        # )
        random_policy = RangeKuhnCFRHeuristic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=AlgorithmConfig(),
            path='/home/clouduser/zcc/Holdem/strategy/kuhn_best.txt',
        )
        tot = 0
        l = 10000
        for i in range(l):
            env.reset()
            while not env.is_over():
                obs = self._observe(env)
                if env.current_agent_id() % 2 == i % 2:
                # if env.current_agent_id() % 2 == 1:
                    action = cfr_policy._do_compute_actions(obs)[0][0]
                else:
                    action = random_policy._do_compute_actions(obs)[0][0]
                env.step(action)
            tot += env._cumulative_rewards['agent_' + str(i % 2)]
            # tot += env._cumulative_rewards['agent_1']
            # print(env.game.log_action)
            # print("\n")
        print()
        print(tot / l * 50 * payoff_max)
        # -3.690426856711293
        # 6.676917002816269
        # 1.5772391812990023
        # 8.329
        # l = 100000 1.41220
        # [0.5 0.5 0.5]
        # [0.66 1 0]

# cc 0.5 * (1.66 / 3)
    # JQ JK QJ QK KJ KQ
# crf 0.5 * (1.33 / 3) * 0.5
# crc 0.5 * (1.33 / 3) * 0.5
# rf 0.5 * (1.33 / 3)
# rc 0.5 * (1.66 / 3)

# JQ cc 0.05195199999999999 [-1, 1]
# JK cc 0.056153999999999996 [-1, 1]
# QJ cc 0.07966666666666666 [1, -1]
# QK cc 0.0735 [-1, 1]
# KJ cc 0.02310333333333333 [1, -1]
# KQ cc 0.019719999999999994 [1, -1]

# JQ crf 0.07538133333333333 [-1.0, 1.0]
# JK crf 0.07117933333333333 [-1.0, 1.0]
# QJ crf 0.03741 [-1.0, 1.0]
# QK crf 0.04006166666666667 [-1.0, 1.0]
# KJ crf 0.0 [-1.0, 1.0]
# KQ crf 0.0 [-1.0, 1.0]

# JQ crc 0.0 [-2, 2]
# JK crc 0.0 [-2, 2]
# QJ crc 0.049589999999999995 [2, -2]
# QK crc 0.053105 [-2, 2]
# KJ crc 0.025229999999999995 [2, -2]
# KQ crc 0.028613333333333327 [2, -2]

# JQ rf 0.03925466666666667 [1.0, -1.0]
# JK rf 0.00015733333333333333 [1.0, -1.0]
# QJ rf 0.0 [1.0, -1.0]
# QK rf 0.0 [1.0, -1.0]
# KJ rf 0.11821499999999999 [1.0, -1.0]
# KQ rf 0.11809666666666666 [1.0, -1.0]

# JQ rc 7.866666666666666e-05 [-2, 2]
# JK rc 0.039175999999999996 [-2, 2]
# QJ rc 0.0 [2, -2]
# QK rc 0.0 [-2, 2]
# KJ rc 0.00011833333333333331 [2, -2]
# KQ rc 0.00023666666666666663 [2, -2]




# JQ cc 0.08049999999999999 [-1, 1]
# JK cc 0.0 [-1, 1]
# QJ cc 0.04958033333333334 [1, -1]
# QK cc 0.0 [-1, 1]
# KJ cc 0.049358 [1, -1]
# KQ cc 0.074 [1, -1]

# JQ crf 0.0 [-1.0, 1.0]
# JK crf 0.08025849999999998 [-1.0, 1.0]
# QJ crf 0.024455964 [-1.0, 1.0]
# QK crf 0.07344133333333333 [-1.0, 1.0]
# KJ crf 0.00022177799999999998 [-1.0, 1.0]
# KQ crf 0.0 [-1.0, 1.0]

# JQ crc 0.0 [-2, 2]
# JK crc 0.00024149999999999996 [-2, 2]
# QJ crc 0.00029703600000000003 [2, -2]
# QK crc 0.000892 [-2, 2]
# KJ crc 0.024420222000000002 [2, -2]
# KQ crc 0.0 [2, -2]

# JQ rf 0.05747316666666667 [1.0, -1.0]
# JK rf 0.0 [1.0, -1.0]
# QJ rf 0.09233333333333334 [1.0, -1.0]
# QK rf 0.0 [1.0, -1.0]
# KJ rf 0.09266666666666667 [1.0, -1.0]
# KQ rf 0.06180866666666668 [1.0, -1.0]

# JQ rc 0.028693500000000004 [-2, 2]
# JK rc 0.08616666666666667 [-2, 2]
# QJ rc 0.0 [2, -2]
# QK rc 0.09233333333333334 [-2, 2]
# KJ rc 0.0 [2, -2]
# KQ rc 0.030858000000000003 [2, -2]

# 0.7716333333333325 4.358044633333333