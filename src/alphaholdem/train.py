import argparse
import random
import ray
from ray import air, train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.tune import CLIReporter, register_env
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from pettingzoo.utils import wrappers

from .poker.component.card import Card
from .poker.component.street import Street
from .model.hunl_conv_model import HUNLConvModel
from .callback.hunl_self_play_callback import create_hunl_self_play_callback
from .policy.random_heuristic import RandomHeuristic
from .poker.poker_game_env import PokerGameEnv
from .poker.no_limit_leduc_holdem_env import NoLimitLeducHoldemEnv
from .utils.logger import log
from .arena.policy.tf_texas_policy import TFTexasPolicy

from .utils.config import load_train_config


def wrap(env: PokerGameEnv):
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def select_policy(agent_id: str, episode: EpisodeV2, **kwargs):
    return ("learned" if episode.episode_id % 2 == int(agent_id[-1:])
            else random.choice(["random"]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/hunl.yaml')
    args = parser.parse_args()
    conf = load_train_config(args.config)

    ray.init(num_gpus=conf.resources.num_gpus, num_cpus=conf.resources.num_cpus)
    env_name = conf.game.type

    register_env(env_name, lambda _: PettingZooEnv(
        wrap(env = NoLimitLeducHoldemEnv(
            num_players=conf.game.num_players,
            num_runs=conf.game.num_runs,
            initial_chips=conf.game.initial_chips,
            showdown_street=Street.from_str(conf.game.showdown_street),
            circular_train=conf.game.circular_train,
            legal_raise_pot_size=list(map(float, args.legal_raise_pot_size.strip().split(','))),
            payoff_max=args.payoff_max,
        ))
    ))

#     ModelCatalog.register_custom_model("BaselineModel", HUNLConvModel)

#     learned_policy = 'learned'
#     initial_policies = {
#         learned_policy: PolicySpec(),
#         "random": PolicySpec(policy_class=RandomHeuristic),
#     }
#     opponent_policies = list(initial_policies.keys() - [learned_policy])

#     config = (
#         PPOConfig()
#         .environment(env=env_name, disable_env_checking=True)
#         .rollouts(
#             num_rollout_workers=args.num_rollout_workers,
#             num_envs_per_worker=args.num_envs_per_worker,
#         )
#         .resources(
#             num_gpus_per_worker=args.num_gpus_per_worker,
#             num_cpus_per_worker=args.num_cpus_per_worker,
#             num_gpus=args.num_gpus_algorithm,
#             num_learner_workers=args.num_learner_workers,
#         )
#         .framework(framework="torch")
#         .debugging(log_level="ERROR")
#         .rl_module(_enable_rl_module_api=False)
#         .multi_agent(
#             policies=initial_policies,
#             policy_mapping_fn=select_policy,
#             policies_to_train=[learned_policy],
#         )
#         .callbacks(
#             create_hunl_self_play_callback(
#                 win_rate_threshold=args.win_rate_threshold,
#                 opponent_policies=opponent_policies,
#                 opponent_count=args.opponent_count,
#                 num_update_iter=args.num_update_iter,
#                 payoff_max=args.payoff_max,
#             ),
#         )
#         .evaluation(evaluation_interval=1)
#         .training(
#             _enable_learner_api=False,
#             lr=args.learning_rate,
#             train_batch_size=args.train_batch_size,
#             sgd_minibatch_size=args.sgd_minibatch_size,
#             num_sgd_iter=args.num_sgd_iter,
#             entropy_coeff = args.entropy_coeff,
#             kl_coeff = args.kl_coeff,
#             kl_target = args.kl_target,
#             vf_loss_coeff = args.vf_loss_coeff,
#             vf_clip_param = args.vf_clip_param,
#             clip_param = args.clip_param,
#             model={
#                 "custom_model": "BaselineModel",
#             },
#         )
#     )

#     tuner = tune.Tuner(
#         "PPO",
#         param_space=config,
#         run_config=air.RunConfig(
#             callbacks= [
#                 WandbLoggerCallback(project=env_name)
#             ],
#             stop={
#                 "timesteps_total": 2_000_000_000,
#                 "training_iteration": 400000,
#             },
#             verbose=2,
#             checkpoint_config=air.CheckpointConfig(
#                 num_to_keep=args.checkpoint_num_to_keep,
#                 checkpoint_frequency=args.checkpoint_frequency,
#                 checkpoint_at_end=True,
#                 checkpoint_score_order="max",
#             ),
#         ),
#     )
#     tuner.fit()

if __name__ == '__main__':
    main()