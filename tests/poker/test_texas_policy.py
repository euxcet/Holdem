import pytest
from alphaholdem.arena.hunl_arena import HunlArena
from alphaholdem.arena.policy.hunl.ppo_hunl_policy import PPOHunlPolicy
from alphaholdem.arena.policy.hunl.tf_texas_policy import TFTexasPolicy

class TestTexasPolicy():
    SKIP = True
    # ppo = PPOTexasPolicy(model_path='/home/clouduser/zcc/AlphaHoldem/torch/checkpoint/hunl/model.pt')

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_ppo_vs_tf(self):
        mean, var = HunlArena().ppo_vs_tf(
            ppo=self.ppo,
            runs=1024,
        )
        print(mean, var)