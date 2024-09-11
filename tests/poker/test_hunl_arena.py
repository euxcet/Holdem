import pytest
from alphaholdem.arena.hunl_arena import HunlArena
from alphaholdem.arena.policy.hunl.ppo_hunl_policy import PPOHunlPolicy
from alphaholdem.arena.policy.hunl.tf_texas_policy import TFTexasPolicy
from alphaholdem.arena.policy.random_policy import RandomPolicy

class TestHunlArena():
    SKIP = False

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_cfr(self):
        ppos = PPOHunlPolicy.load_policies_from_run('/home/clouduser/ray_results/PPO_2024-09-01_12-07-04')
        # tf = TFTexasPolicy('/home/clouduser/zcc/checkpoint/38000_model/model.ckpt')
        random = RandomPolicy()
        arena = HunlArena('./checkpoint/supervise/small_v1/supervise.pt')
        mean, var = arena.policy_vs_policy(
            policy0=arena.nash,
            # policy0=random,
            # policy0=tf,
            policy1=random,
            # policy1=ppos[0],
            runs=4096,
        )
        print('Nash vs Nash:', mean)
        # 6.298828124999994