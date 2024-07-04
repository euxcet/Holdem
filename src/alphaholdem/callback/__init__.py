from ..config.train_config import TrainConfig
from .hunl_self_play_callback import create_hunl_self_play_callback
from .leduc_self_play_callback import create_leduc_self_play_callback
from .range_leduc_self_play_callback import create_range_leduc_self_play_callback

def get_callback(cfg: TrainConfig):
    if cfg.self_play.type == 'hunl':
        return create_hunl_self_play_callback(
            alphaholdem_checkpoint=cfg.policy.alphaholdem,
            opponent_policies=cfg.self_play.opponent_policies,
            num_opponent_limit=cfg.self_play.num_opponent_limit,
            num_update_iter=cfg.self_play.num_update_iter,
            payoff_max=cfg.game.payoff_max,
            win_rate_window_size=cfg.self_play.win_rate_window_size,
            arena_runs=cfg.self_play.arena_runs,
        )
    elif cfg.self_play.type == 'leduc':
        return create_leduc_self_play_callback(
            cfr_strategy_checkpoint=cfg.policy.leduc_cfr,
            opponent_policies=cfg.self_play.opponent_policies,
            num_opponent_limit=cfg.self_play.num_opponent_limit,
            num_update_iter=cfg.self_play.num_update_iter,
            payoff_max=cfg.game.payoff_max,
            win_rate_window_size=cfg.self_play.win_rate_window_size,
            arena_runs=cfg.self_play.arena_runs,
        )
    elif cfg.self_play.type == 'range_leduc':
        return create_range_leduc_self_play_callback(
            cfr_strategy_checkpoint=cfg.policy.leduc_cfr,
            opponent_policies=cfg.self_play.opponent_policies,
            num_opponent_limit=cfg.self_play.num_opponent_limit,
            num_update_iter=cfg.self_play.num_update_iter,
            payoff_max=cfg.game.payoff_max,
            win_rate_window_size=cfg.self_play.win_rate_window_size,
            arena_runs=cfg.self_play.arena_runs,
        )
    raise Exception