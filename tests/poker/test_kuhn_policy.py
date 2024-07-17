import pytest
from alphaholdem.arena.kuhn_arena import KuhnArena
from alphaholdem.arena.policy.kuhn.lookup_kuhn_policy import LookupKuhnPolicy
from alphaholdem.arena.policy.kuhn.ppo_kuhn_policy import PPOKuhnPolicy
from alphaholdem.arena.leduc_arena import LeducArena
from alphaholdem.arena.policy.leduc.lookup_leduc_policy import LookupLeducPolicy

class TestTexasPolicy():
    SKIP = True

    run_folder = '/home/clouduser/ray_results/PPO_2024-07-09_05-51-05'
    ppos = PPOKuhnPolicy.load_policies_from_run(run_folder)

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_cfr(self):
        mean, var = KuhnArena('/home/clouduser/zcc/Holdem/strategy/kuhn_nash.txt').policy_vs_policy(
            policy0=LookupLeducPolicy('/home/clouduser/zcc/Holdem/strategy/kuhn_nash.txt'),
            policy1=LookupLeducPolicy('/home/clouduser/zcc/Holdem/strategy/kuhn_nash.txt'),
            runs=16384,
        )
        print(mean, var)