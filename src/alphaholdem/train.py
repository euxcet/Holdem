import argparse
import random
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.models.torch.torch_action_dist import TorchDeterministic
from ray.tune import register_env
from ray.air.integrations.wandb import WandbLoggerCallback

from .model import get_model
from .callback import get_callback
from .policy import get_policies
from .poker import get_poker_env
from .config.config import load_train_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/hunl.yaml')
    args = parser.parse_args()
    cfg = load_train_config(args.config, args)

    ray.init(num_gpus=cfg.resources.num_gpus, num_cpus=cfg.resources.num_cpus)
    env_name = cfg.game.type

    register_env(env_name, lambda _: PettingZooEnv(env=get_poker_env(cfg.game)))

    ModelCatalog.register_custom_model("Model", get_model(cfg))
    # ModelCatalog.register_custom_action_dist("ActionDist", TorchDeterministic)

    learned_policy = 'learned'
    initial_policies = get_policies(cfg.self_play.rule_based_policies)
    initial_policies.update({learned_policy: PolicySpec()})

    def select_policy(agent_id: str, episode: EpisodeV2, **kwargs):
        if len(cfg.self_play.rule_based_policies) == 0:
            return learned_policy
        return (learned_policy if episode.episode_id % 2 == int(agent_id[-1:])
                else random.choice(cfg.self_play.rule_based_policies))

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(
            num_rollout_workers=cfg.rllib.num_rollout_workers,
            num_envs_per_worker=cfg.rllib.num_envs_per_worker,
        )
        .resources(
            num_gpus_per_worker=cfg.rllib.num_gpus_per_worker,
            num_cpus_per_worker=cfg.rllib.num_cpus_per_worker,
            num_gpus=cfg.rllib.num_gpus_algorithm,
            num_learner_workers=cfg.rllib.num_learner_workers,
        )
        .framework(framework=cfg.rllib.framework)
        .debugging(log_level=cfg.rllib.log_level)
        .rl_module(_enable_rl_module_api=False)
        .multi_agent(
            policies=initial_policies,
            policy_mapping_fn=select_policy,
            policies_to_train=[learned_policy],
        )
        .callbacks(get_callback(cfg))
        .evaluation(evaluation_interval=cfg.rllib.evaluation_interval)
        .training(
            _enable_learner_api=False,
            lr=cfg.hyper.learning_rate,
            train_batch_size=cfg.hyper.train_batch_size,
            sgd_minibatch_size=cfg.hyper.sgd_minibatch_size,
            num_sgd_iter=cfg.hyper.num_sgd_iter,
            entropy_coeff=cfg.hyper.entropy_coeff,
            kl_coeff=cfg.hyper.kl_coeff,
            kl_target=cfg.hyper.kl_target,
            vf_loss_coeff=cfg.hyper.vf_loss_coeff,
            vf_clip_param=cfg.hyper.vf_clip_param,
            clip_param=cfg.hyper.clip_param,
            model={
                "custom_model": "Model",
                # "custom_action_dist": "ActionDist",
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
                "timesteps_total": cfg.hyper.stop_timesteps_total,
                "training_iteration": cfg.hyper.stop_training_iteration,
            },
            verbose=cfg.rllib.verbose,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=cfg.hyper.checkpoint_num_to_keep,
                checkpoint_frequency=cfg.hyper.checkpoint_frequency,
            ),
        ),
    )
    tuner.fit()

if __name__ == '__main__':
    main()