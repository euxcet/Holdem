from alphaholdem.arena.texas_arena import TexasArena
from alphaholdem.arena.policy.ppo_texas_policy import PPOTexasPolicy
from alphaholdem.arena.policy.tf_texas_policy import TFTexasPolicy

class TestTexasPolicy():
    SKIP = False
    ppo = PPOTexasPolicy(model_path='/home/clouduser/zcc/AlphaHoldem/torch/checkpoint/hunl/model.pt')

    def test_ppo_vs_tf(self):
        mean, var = TexasArena().ppo_vs_tf(
            ppo=self.ppo,
            runs=1024,
        )
        print(mean, var)