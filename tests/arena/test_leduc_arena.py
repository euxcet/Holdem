import pytest
from alphaholdem.arena.leduc_arena import LeducArena
from alphaholdem.arena.policy.leduc.lookup_leduc_policy import LookupLeducPolicy

from alphaholdem.arena.cfr.tree import *
from alphaholdem.arena.cfr.game import *
from alphaholdem.arena.cfr.strategy import *

class TestLeducArena():

    def test_cfr(self):
        mean, var = LeducArena('./strategy/leduc_nash.txt').policy_vs_policy(
            policy0=LookupLeducPolicy('./strategy/leduc_nash.txt'),
            policy1=LookupLeducPolicy('./strategy/leduc_ppo.txt'),
        )
        print('Leduc:', mean)

        x0, x1 = LookupLeducPolicy('./strategy/leduc_nash.txt').to_strategy()
        y0, y1 = LookupLeducPolicy('./strategy/leduc_ppo.txt').to_strategy()

        rules = leduc_rules()
        e0 = StrategyProfile(rules, [x0, y1]).expected_value()
        e1 = StrategyProfile(rules, [y0, x1]).expected_value()
        ev = (e0[0] + e1[1]) * 25
        print(ev * 25)

        profile = StrategyProfile(rules, [y0, y1])
        br = profile.best_response()[1]
        exploitability = (br[0] + br[1]) / 2
        print(exploitability)
