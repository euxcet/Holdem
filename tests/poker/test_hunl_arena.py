import pytest
from alphaholdem.arena.hunl_arena import HunlArena
from alphaholdem.arena.policy.hunl.ppo_hunl_policy import PPOHunlPolicy

class TestHunlArena():
    SKIP = False

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_cfr(self):
        ppos = PPOHunlPolicy.load_policies_from_run('/home/clouduser/ray_results/PPO_2024-09-01_12-07-04')
        arena = HunlArena()
        mean, var = arena.policy_vs_policy(
            policy0=arena.nash,
            policy1=ppos[0],
            runs=1024,
        )
        print('Nash vs Nash:', mean)
        # 6.298828124999994