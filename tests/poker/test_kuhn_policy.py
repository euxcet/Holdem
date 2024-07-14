import pytest
from alphaholdem.arena.kuhn_arena import KuhnArena
from alphaholdem.arena.policy.lookup_kuhn_policy import LookupKuhnPolicy
from alphaholdem.arena.policy.ppo_kuhn_policy import PPOKuhnPolicy
from alphaholdem.arena.leduc_arena import LeducArena
from alphaholdem.arena.policy.lookup_leduc_policy import LookupLeducPolicy

class TestTexasPolicy():
    SKIP = False

    run_folder = '/home/clouduser/ray_results/PPO_2024-07-09_05-51-05'
    ppos = PPOKuhnPolicy.load_policies_from_run(run_folder)

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_cfr(self):
        ...

        # mean, var = KuhnArena('/home/clouduser/zcc/Holdem/strategy/kuhn.txt').cfr_self_play(
        #     cfr1=CFRKuhnPolicy('/home/clouduser/zcc/Holdem/strategy/kuhn.txt'),
        #     cfr2=CFRKuhnPolicy('/home/clouduser/zcc/Holdem/strategy/kuhn_.txt'),
        #     runs=65536,
        # )
        # print(mean, var)
        # mean, var = KuhnArena('/home/clouduser/zcc/Holdem/strategy/kuhn.txt').cfr_self_play(
        #     cfr1=CFRKuhnPolicy('/home/clouduser/zcc/Holdem/strategy/kuhn.txt'),
        #     cfr2=CFRKuhnPolicy('/home/clouduser/zcc/Holdem/strategy/kuhn__.txt'),
        #     runs=65536,
        # )
        # print(mean, var)
        # mean, var = KuhnArena('/home/clouduser/zcc/Holdem/strategy/kuhn.txt').cfr_self_play(
        #     cfr1=CFRKuhnPolicy('/home/clouduser/zcc/Holdem/strategy/kuhn.txt'),
        #     cfr2=CFRKuhnPolicy('/home/clouduser/zcc/Holdem/strategy/kuhn___.txt'),
        #     runs=65536,
        # )
        # print(mean, var)
        # mean, var = KuhnArena('/home/clouduser/zcc/Holdem/strategy/kuhn.txt').cfr_self_play(
        #     cfr1=CFRKuhnPolicy('/home/clouduser/zcc/Holdem/strategy/kuhn.txt'),
        #     cfr2=CFRKuhnPolicy('/home/clouduser/zcc/Holdem/strategy/kuhn____.txt'),
        #     runs=65536,
        # )
        # print(mean, var)