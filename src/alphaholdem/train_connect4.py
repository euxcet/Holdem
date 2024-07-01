import random
import ray
from ray import air, train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune import register_env
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from pettingzoo.classic import connect_four_v3

from .model.connect4_mask_conv_model import Connect4MaskConvModel
from .callback.connect4_self_play_callback import create_connect4_self_play_callback
from .policy.connect4.policy import RandomHeuristic, AlwaysSameHeuristic, SmartHeuristic, BeatLastHeuristic, LinearHeuristic

def select_policy(agent_id: str, episode: EpisodeV2, **kwargs):
    return ("learned" if episode.episode_id % 2 == int(agent_id[-1:])
            else random.choice(['random', 'always_same', 'beat_last', 'smart', 'linear']))

def main():
    ray.init(num_gpus=8)

    env_name = "connect_four"
    register_env(env_name, lambda _: PettingZooEnv(
        connect_four_v3.env()
    ))

    ModelCatalog.register_custom_model("BaselineModel", Connect4MaskConvModel)

    learned_policy = 'learned'
    initial_policies = {
        learned_policy: PolicySpec(),
        "random": PolicySpec(policy_class=RandomHeuristic),
        "always_same": PolicySpec(policy_class=AlwaysSameHeuristic),
        "beat_last": PolicySpec(policy_class=BeatLastHeuristic),
        "smart": PolicySpec(policy_class=SmartHeuristic),
        "linear": PolicySpec(policy_class=LinearHeuristic),
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
            create_connect4_self_play_callback(
                win_rate_threshold=0.95,
                opponent_policies=opponent_policies,
                opponent_count=7,
            )
        )
        .evaluation(evaluation_interval=1)
        .training(
            _enable_learner_api=False,
            train_batch_size=8192,
            model={
                "custom_model": "BaselineModel",
                "post_fcnet_hiddens": [256, 256],
                "conv_filters": [[32, [4, 4], 1]],
            },
        )
    )

    analysis = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=air.RunConfig(
            callbacks= [
                WandbLoggerCallback(project="connect4")
            ],
            stop={
                "timesteps_total": 2000_000_000,
                "training_iteration": 30000,
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

    # algorithm = Algorithm.from_checkpoint(analysis.get_best_result().checkpoint)
    # ppo_policy = algo

if __name__ == '__main__':
    main()