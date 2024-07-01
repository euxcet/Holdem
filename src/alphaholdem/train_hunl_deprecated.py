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
from .model.hunl_conv_model_action_8 import HUNLConvModel
from .callback.hunl_self_play_callback_deprecated import create_hunl_self_play_callback
from .policy.hunl.policy import RandomHeuristic
from .poker.poker_game_env import PokerGameEnv
from .poker.no_limit_texas_holdem_env import NoLimitTexasHoldemEnv
from .utils.logger import log

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
    parser.add_argument('-s', '--showdown-street', type=str, default='showdown', required=True)
    parser.add_argument('-r', '--num-runs', type=int, default=100, required=True)
    parser.add_argument('-p', '--num-players', type=int, default=2, required=True)
    parser.add_argument('--initial-chips', type=int, default=200, required=True)
    parser.add_argument('--win-rate-threshold', type=int, default=100, required=True)
    parser.add_argument('--opponent-count', type=int, default=4, required=True)
    parser.add_argument('--min-update-step-count', type=int, default=20)
    parser.add_argument('--max-update-step-count', type=int, default=200)
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--num-rollout-workers', type=int, default=16)
    parser.add_argument('--num-envs-per-worker', type=int, default=4)
    parser.add_argument('--num-cpus-per-worker', type=float, default=0.125)
    parser.add_argument('--num-gpus-per-worker', type=float, default=0.03125)
    parser.add_argument('--num-learner-workers', type=int, default=0)
    parser.add_argument('--train-batch-size', type=int, default=16384)
    parser.add_argument('--circular-train', type=bool, default=False)
    parser.add_argument('--legal-raise-pot-size', type=str, default="0.75")
    parser.add_argument('--custom-board-cards', type=str, default=None)
    # parser.add_argument('--num-gpus-per-learner-worker', type=float, default=0.0625)
    parser.add_argument('--kl-coeff', type=float, default=0.2)
    parser.add_argument('--kl-target', type=float, default=0.003)
    parser.add_argument('--vf-clip-param', type=float, default=10.0)
    parser.add_argument('--vf-loss-coeff', type=float, default=1.0)
    parser.add_argument('--clip-param', type=float, default=0.2)
    parser.add_argument('--entropy-coeff', type=float, default=0.01)

    args = parser.parse_args()
    log.info(Street.from_str(args.showdown_street))
    log.info(args.num_players)
    log.info(args.num_runs)
    log.info(args.circular_train)
    log.info(list(map(float, args.legal_raise_pot_size.strip().split(','))))

    ray.init(num_gpus=args.num_gpus)

    env_name = "hunl"
    register_env(env_name, lambda _: PettingZooEnv(
        wrap(env = NoLimitTexasHoldemEnv(
            num_players=args.num_players,
            num_runs=args.num_runs,
            initial_chips=args.initial_chips,
            showdown_street=Street.from_str(args.showdown_street),
            circular_train=args.circular_train,
            custom_board_cards=Card.from_str_list(args.custom_board_cards.strip().split(',')) if args.custom_board_cards is not None else None,
            legal_raise_pot_size=list(map(float, args.legal_raise_pot_size.strip().split(','))),
        ))
    ))

    ModelCatalog.register_custom_model("BaselineModel", HUNLConvModel)

    learned_policy = 'learned'
    initial_policies = {
        learned_policy: PolicySpec(),
        "random": PolicySpec(policy_class=RandomHeuristic),
    }
    opponent_policies = list(initial_policies.keys() - [learned_policy])

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(
            num_rollout_workers=args.num_rollout_workers,
            num_envs_per_worker=args.num_envs_per_worker,
        )
        .resources(
            num_gpus_per_worker=args.num_gpus_per_worker,
            num_cpus_per_worker=args.num_cpus_per_worker,
            num_gpus=0.25,
            num_learner_workers=args.num_learner_workers,
            # num_gpus_per_learner_worker=args.num_gpus_per_learner_worker,
        )
        .framework(framework="torch")
        .debugging(log_level="ERROR")
        .rl_module(_enable_rl_module_api=False)
        .multi_agent(
            policies=initial_policies,
            policy_mapping_fn=select_policy,
            policies_to_train=[learned_policy],
        )
        .callbacks(
            create_hunl_self_play_callback(
                win_rate_threshold=args.win_rate_threshold,
                opponent_policies=opponent_policies,
                opponent_count=args.opponent_count,
                min_update_step_count=args.min_update_step_count,
                max_update_step_count=args.max_update_step_count,
            )
        )
        .evaluation(evaluation_interval=1)
        .training(
            _enable_learner_api=False,
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=512,
            num_sgd_iter = 30,
            entropy_coeff = args.entropy_coeff,
            kl_coeff = args.kl_coeff,
            kl_target = args.kl_target,
            vf_loss_coeff = args.vf_loss_coeff,
            vf_clip_param = args.vf_clip_param,
            clip_param = args.clip_param,
            model={
                "custom_model": "BaselineModel",
            },
        )
    )

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=air.RunConfig(
            callbacks= [
                WandbLoggerCallback(project=env_name)
            ],
            stop={
                "timesteps_total": 2_000_000_000,
                "training_iteration": 4000,
            },
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=10,
                checkpoint_at_end=True,
                checkpoint_frequency=10,
                checkpoint_score_order="max",
            ),
        ),
    )
    # tuner.restore('/home/clouduser/ray_results/PPO_2024-05-13_03-43-23', resume_errored=True)
    tuner.fit()

if __name__ == '__main__':
    main()