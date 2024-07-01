import random
import ray
from ray import air, train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune import CLIReporter, register_env
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from pettingzoo.classic import connect_four_v3
from pettingzoo.utils import wrappers

from .model.aof_conv_model import AoFConvModel
from .callback.aof_self_play_callback import create_aof_self_play_callback
from .policy.aof.policy import RandomHeuristic, BroadwayHeuristic, AlwaysFoldHeuristic, AlwaysAllInHeuristic
from .poker.poker_game_env import PokerGameEnv
from .poker.aof_env import AoFEnv

def wrap(env: PokerGameEnv):
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def select_policy(agent_id: str, episode: EpisodeV2, **kwargs):
    return ("learned" if episode.episode_id % 2 == int(agent_id[-1:])
            else random.choice(["random", "broadway", "allin", "fold"]))

def main():
    ray.init(num_gpus=8)

    env_name = "aof"
    register_env(env_name, lambda _: PettingZooEnv(
        wrap(env = AoFEnv(num_players=2))
    ))

    ModelCatalog.register_custom_model("BaselineModel", AoFConvModel)

    learned_policy = 'learned'
    initial_policies = {
        learned_policy: PolicySpec(),
        "random": PolicySpec(policy_class=RandomHeuristic),
        "broadway": PolicySpec(policy_class=BroadwayHeuristic),
        "allin": PolicySpec(policy_class=AlwaysAllInHeuristic),
        "fold": PolicySpec(policy_class=AlwaysFoldHeuristic),
    }
    opponent_policies = list(initial_policies.keys() - [learned_policy])

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(num_rollout_workers=16, rollout_fragment_length=512)
        .resources(num_gpus=8)
        .framework(framework="torch")
        .debugging(log_level="ERROR")
        .rl_module(_enable_rl_module_api=False)
        .multi_agent(
            policies=initial_policies,
            policy_mapping_fn=select_policy,
            policies_to_train=[learned_policy],
        )
        .callbacks(
            create_aof_self_play_callback(
                win_rate_threshold=30,
                opponent_policies=opponent_policies,
                opponent_count=8,
            )
        )
        .evaluation(evaluation_interval=1)
        .training(
            _enable_learner_api=False,
            train_batch_size=8192,
            model={
                "custom_model": "BaselineModel",
            },
        )
    )

    analysis = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=air.RunConfig(
            callbacks= [
                WandbLoggerCallback(project="aof")
            ],
            stop={
                "timesteps_total": 2_000_000_000,
                "training_iteration": 20000,
            },
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=10,
                checkpoint_at_end=True,
                checkpoint_frequency=10,
                checkpoint_score_order="max",
            ),
        ),
    ).fit()

if __name__ == '__main__':
    main()