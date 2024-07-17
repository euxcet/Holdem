import pytest
from alphaholdem.arena.texas_arena import TexasArena
from alphaholdem.arena.policy.hunl.ppo_texas_policy import PPOTexasPolicy
from alphaholdem.arena.policy.hunl.tf_texas_policy import TFTexasPolicy

class TestTexasPolicy():
    SKIP = True
    # ppo = PPOTexasPolicy(model_path='/home/clouduser/zcc/AlphaHoldem/torch/checkpoint/hunl/model.pt')

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_ppo_vs_tf(self):
        mean, var = TexasArena().ppo_vs_tf(
            ppo=self.ppo,
            runs=1024,
        )
        print(mean, var)