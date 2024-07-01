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
from .callback.hunl_play_with_tf_callback import create_hunl_play_with_tf_callback
from .policy.hunl.policy import TfHeuristic, BestHeuristic
from .poker.poker_game_env import PokerGameEnv
from .poker.no_limit_leduc_holdem_env import NoLimitLeducHoldemEnv
from .utils.logger import log
from .arena.policy.tf_texas_policy import TFTexasPolicy

def wrap(env: PokerGameEnv):
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def select_policy(agent_id: str, episode: EpisodeV2, **kwargs):
    return ("learned" if episode.episode_id % 2 == int(agent_id[-1:])
            else random.choice(["tf"]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--showdown-street', type=str, default='showdown', required=True)
    parser.add_argument('--num-runs', type=int, default=100, required=True)
    parser.add_argument('--num-players', type=int, default=2, required=True)
    parser.add_argument('--initial-chips', type=int, default=200, required=True)
    parser.add_argument('--win-rate-threshold', type=int, default=100, required=True)
    parser.add_argument('--opponent-count', type=int, default=4, required=True)
    parser.add_argument('--num-update-iter', type=int, default=20)
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--num-rollout-workers', type=int, default=16)
    parser.add_argument('--num-envs-per-worker', type=int, default=4)
    parser.add_argument('--num-cpus-per-worker', type=float, default=0.125)
    parser.add_argument('--num-gpus-per-worker', type=float, default=0.03125)
    parser.add_argument('--num-gpus-algorithm', type=float, default=0.5)
    parser.add_argument('--num-learner-workers', type=int, default=0)
    parser.add_argument('--train-batch-size', type=int, default=16384)
    parser.add_argument('--circular-train', type=bool, default=False)
    parser.add_argument('--legal-raise-pot-size', type=str, default="0.5,0.75,1,1.5,2")
    parser.add_argument('--custom-board-cards', type=str, default=None)
    parser.add_argument('--kl-coeff', type=float, default=0.2)
    parser.add_argument('--kl-target', type=float, default=0.003)
    parser.add_argument('--vf-clip-param', type=float, default=10.0)
    parser.add_argument('--vf-loss-coeff', type=float, default=1.0)
    parser.add_argument('--clip-param', type=float, default=0.2)
    parser.add_argument('--entropy-coeff', type=float, default=0.01)
    parser.add_argument('--payoff-max', type=float, default=200.0)
    parser.add_argument('--learning-rate', type=float, default=0.00005)
    parser.add_argument('--sgd-minibatch-size', type=int, default=256)
    parser.add_argument('--num-sgd-iter', type=int, default=30)
    parser.add_argument('--checkpoint-num-to-keep', type=int, default=10)
    parser.add_argument('--checkpoint-frequency', type=int, default=10)

    args = parser.parse_args()
    log.info(Street.from_str(args.showdown_street))
    log.info(args.num_players)
    log.info(args.num_runs)
    log.info(args.circular_train)
    log.info(list(map(float, args.legal_raise_pot_size.strip().split(','))))

    ray.init(num_gpus=args.num_gpus, num_cpus=64)

    env_name = "hunl"
    register_env(env_name, lambda _: PettingZooEnv(
        wrap(env = NoLimitLeducHoldemEnv(
            num_players=args.num_players,
            num_runs=args.num_runs,
            initial_chips=args.initial_chips,
            showdown_street=Street.from_str(args.showdown_street),
            circular_train=args.circular_train,
            legal_raise_pot_size=list(map(float, args.legal_raise_pot_size.strip().split(','))),
            payoff_max=args.payoff_max,
        ))
    ))

    ModelCatalog.register_custom_model("BaselineModel", HUNLConvModel)

    learned_policy = 'learned'
    initial_policies = {
        learned_policy: PolicySpec(),
        "tf": PolicySpec(policy_class=TfHeuristic),
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
            num_gpus=args.num_gpus_algorithm,
            num_learner_workers=args.num_learner_workers,
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
            create_hunl_play_with_tf_callback(
                payoff_max=args.payoff_max,
            ),
        )
        .evaluation(evaluation_interval=1)
        .training(
            _enable_learner_api=False,
            lr=args.learning_rate,
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
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
                "training_iteration": 400000,
            },
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=args.checkpoint_num_to_keep,
                checkpoint_frequency=args.checkpoint_frequency,
                checkpoint_at_end=True,
                checkpoint_score_order="max",
            ),
        ),
    )
    tuner.fit()

if __name__ == '__main__':
    main()