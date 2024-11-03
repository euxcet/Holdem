from alphaholdem.arena.cfr.tree import *
from alphaholdem.arena.cfr.game import *
from alphaholdem.arena.cfr.strategy import *

class TestCFR:
    def test_cfr(self):
        rules = leduc_rules()
        s0 = Strategy(0)
        s1 = Strategy(1)
        s0.load_from_file('strategy/leduc/0.strat')
        s1.load_from_file('strategy/leduc/1.strat')
        profile = StrategyProfile(rules, [s0, s1])
        brev = profile.best_response()
        print(brev[1])