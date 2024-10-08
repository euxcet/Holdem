from pettingzoo.utils import wrappers
from ..config.train_config import TrainConfig
from .poker_game_env import PokerGameEnv
from .limit_leduc_holdem_env import LimitLeducHoldemEnv
from .kuhn_poker_env import KuhnPokerEnv
from .range_kuhn_poker_env import RangeKuhnPokerEnv
from .range_limit_leduc_holdem_env import RangeLimitLeducHoldemEnv
from .no_limit_texas_holdem_env import NoLimitTexasHoldemEnv


def get_poker_env(cfg: TrainConfig.TrainGameConfig):
    def _wrap(env: PokerGameEnv):
        env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    def _range_wrap(env: PokerGameEnv):
        # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
        # env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    if cfg.type == 'hunl':
        return _wrap(NoLimitTexasHoldemEnv(
            num_players=cfg.num_players,
            num_runs=cfg.num_runs,
            initial_chips=cfg.initial_chips,
            showdown_street=cfg.showdown_street,
            custom_board_cards=cfg.custom_board_cards,
            circular_train=cfg.circular_train,
            raise_pot_size=cfg.legal_raise_pot_size,
            legal_raise_pot_size=cfg.legal_raise_pot_size,
            payoff_max=cfg.payoff_max,
        ))
    elif cfg.type == 'leduc':
        return _wrap(LimitLeducHoldemEnv(
            num_players=cfg.num_players,
            num_runs=cfg.num_runs,
            initial_chips=cfg.initial_chips,
            showdown_street=cfg.showdown_street,
            custom_board_cards=cfg.custom_board_cards,
            circular_train=cfg.circular_train,
            payoff_max=cfg.payoff_max,
        ))
    elif cfg.type == 'kuhn':
        return _wrap(KuhnPokerEnv(
            circular_train=cfg.circular_train,
            payoff_max=cfg.payoff_max,
        ))
    elif cfg.type == 'range_kuhn':
        return _range_wrap(RangeKuhnPokerEnv(
            circular_train=cfg.circular_train,
            payoff_max=cfg.payoff_max,
        ))
    elif cfg.type == 'range_leduc':
        return _range_wrap(RangeLimitLeducHoldemEnv(
            payoff_max=cfg.payoff_max,
        ))
    raise Exception