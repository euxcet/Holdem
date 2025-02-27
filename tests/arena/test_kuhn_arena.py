import pytest
from alphaholdem.arena.kuhn_arena import KuhnArena
from alphaholdem.arena.policy.kuhn.lookup_kuhn_policy import LookupKuhnPolicy
from alphaholdem.arena.policy.kuhn.ppo_kuhn_policy import PPOKuhnPolicy

from alphaholdem.arena.cfr.tree import *
from alphaholdem.arena.cfr.game import *
from alphaholdem.arena.cfr.strategy import *

class TestKuhnArena():

    def test_cfr(self):
        mean, var = KuhnArena('./strategy/kuhn_nash.txt').policy_vs_policy(
            policy0=LookupKuhnPolicy('./strategy/kuhn_nash.txt'),
            policy1=LookupKuhnPolicy('./strategy/kuhn_nash.txt'),
            runs=16384,
        )
        print('Kuhn arena:', mean)

        x0, x1 = LookupKuhnPolicy('./strategy/kuhn_nash.txt').to_strategy()
        y0, y1 = LookupKuhnPolicy('./strategy/kuhn_nash.txt').to_strategy()

        rules = kuhn_rules()
        e0 = StrategyProfile(rules, [x0, y1]).expected_value()
        e1 = StrategyProfile(rules, [y0, x1]).expected_value()
        print(e0, e1)
        ev = (e0[0] + e1[1]) * 25
        print(ev * 25)

        profile = StrategyProfile(rules, [y0, y1])
        br = profile.best_response()[1]
        exploitability = (br[0] + br[1]) / 2
        print(exploitability)
