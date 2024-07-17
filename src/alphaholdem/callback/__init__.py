from ..config.train_config import TrainConfig
from .kuhn_k_best_callback import create_kuhn_k_best_callback
from .naive_self_play_callback import NaiveSelfPlayCallback
from ..arena.kuhn_arena import KuhnArena
from ..arena.leduc_arena import LeducArena
from ..arena.policy.kuhn.ppo_kuhn_policy import PPOKuhnPolicy
from ..arena.policy.kuhn.ppo_range_kuhn_policy import PPORangeKuhnPolicy
from ..arena.policy.leduc.ppo_leduc_policy import PPOLeducPolicy
from ..arena.policy.leduc.ppo_range_leduc_policy import PPORangeLeducPolicy

def get_callback(cfg: TrainConfig):
    if cfg.self_play.arena == 'kuhn':
        arena = KuhnArena(cfg.policy.kuhn_nash)
    elif cfg.self_play.arena == 'leduc':
        arena = LeducArena(cfg.policy.leduc_nash)
    else:
        raise Exception

    if cfg.self_play.policy_type == 'kuhn':
        policy_type = PPOKuhnPolicy
    elif cfg.self_play.policy_type == 'range_kuhn':
        policy_type = PPORangeKuhnPolicy
    elif cfg.self_play.policy_type == 'leduc':
        policy_type = PPOLeducPolicy
    elif cfg.self_play.policy_type == 'range_leduc':
        policy_type = PPORangeLeducPolicy
    else:
        raise Exception

    if cfg.self_play.type == 'naive':
        class WrapSelfPlayCallback(NaiveSelfPlayCallback):
            def __init__(self):
                super().__init__(
                    num_opponent_limit=cfg.self_play.num_opponent_limit,
                    num_update_iter=cfg.self_play.num_update_iter,
                    arena=arena,
                    arena_runs=cfg.self_play.arena_runs,
                    payoff_max=cfg.game.payoff_max,
                    rule_based_policies=cfg.self_play.rule_based_policies,
                    policy_type=policy_type,
                )
        return WrapSelfPlayCallback
    elif cfg.self_play.type == 'k_best':
        ...

    raise Exception