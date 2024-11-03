import pytest
from alphaholdem.arena.kuhn_arena import KuhnArena
from alphaholdem.arena.policy.kuhn.lookup_kuhn_policy import LookupKuhnPolicy
from alphaholdem.arena.policy.kuhn.ppo_kuhn_policy import PPOKuhnPolicy
from alphaholdem.arena.leduc_arena import LeducArena
from alphaholdem.arena.policy.leduc.lookup_leduc_policy import LookupLeducPolicy

class TestKuhnArena():

    def test_cfr(self):
        mean, var = KuhnArena('./strategy/kuhn_nash.txt').policy_vs_policy(
            policy0=LookupKuhnPolicy('./strategy/kuhn_nash.txt'),
            policy1=LookupKuhnPolicy('./strategy/kuhn_best.txt'),
            runs=16384,
        )
        print('Kuhn arena:', mean)