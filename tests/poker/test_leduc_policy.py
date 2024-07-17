import pytest
from alphaholdem.poker.limit_leduc_holdem_env import LimitLeducHoldemEnv
from alphaholdem.poker.component.card import Card
from alphaholdem.poker.component.street import Street
from alphaholdem.arena.policy.leduc.ppo_leduc_policy import PPOLeducPolicy
from alphaholdem.arena.policy.leduc.ppo_range_leduc_policy import PPORangeLeducPolicy
from alphaholdem.arena.policy.leduc.lookup_leduc_policy import LookupLeducPolicy
from alphaholdem.arena.leduc_arena import LeducArena

class TestLeducPolicy():
    SKIP = True
    # run_folder = '/home/clouduser/ray_results/PPO_2024-05-18_15-09-37'
    cfr = LookupLeducPolicy('strategy/leduc.txt')

    def test_range_policy(self):
        ppos = PPORangeLeducPolicy.load_policies_from_run('/home/clouduser/ray_results/PPO_2024-07-17_06-18-22')
        mean, var = LeducArena('/home/clouduser/zcc/Holdem/strategy/leduc.txt').policy_vs_policy(
            policy0=LookupLeducPolicy('/home/clouduser/zcc/Holdem/strategy/leduc.txt'),
            policy1=ppos[-1],
            runs=16384,
        )
        print(mean, var)

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_tree(self):
        ppos = PPOLeducPolicy.load_policies_from_run('/home/clouduser/ray_results/PPO_2024-07-13_15-04-54')
        mean, var = LeducArena('/home/clouduser/zcc/Holdem/strategy/leduc.txt').policy_vs_policy(
            policy0=LookupLeducPolicy('/home/clouduser/zcc/Holdem/strategy/leduc.txt'),
            policy1=ppos[-1],
            runs=16384,
        )
        print(mean, var)
        
    # Fold Check Call All_in Raise_25% Raise_50% Raise_75% Raise_125%
    # 0    1     2    3      4         5         6         7
    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_range_policy_(self):
        action_history = []
        board_cards = Card.from_str_list([])
        env = LimitLeducHoldemEnv(
            num_players=2,
            initial_chips=100,
            showdown_street=Street.Turn,
            custom_board_cards=Card.from_str_list(board_cards),
        )
        env.reset()
        for action in action_history:
            env.step(action)
        print(self.ppos[-1].get_range_policy(env.observe_current()))

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_cfr_self_play(self):
        mean, var = LeducArena().cfr_self_play(
            cfr=self.cfr,
            runs=1024,
        )
        print(mean, var)

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_latest_win_rate(self):
        mean, var = LeducArena().ppo_vs_cfr(
            ppo=self.ppos[-1],
            cfr=self.cfr,
            runs=1024,
            batch_size=32,
        )
        print(mean, var)

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_win_rate_curve(self):
        for ppo in self.ppos:
            mean, var = LeducArena().ppo_vs_cfr(
                ppo=ppo,
                cfr=self.cfr,
                runs=1024,
                batch_size=32
            )
            print(ppo.model_path)
            print(mean, var)