import numpy as np
from .envs.no_limit_texas_holdem_env import create_no_limit_holdem_env
from ..poker.no_limit_texas_holdem_env import NoLimitTexasHoldemEnv
from .policy.ppo_poker_policy import PPOPokerPolicy
from .policy.tf_texas_policy import TFTexasPolicy

# TODO: refactor
class TexasArena():
    def __init__(self, tf_checkpoint: str = './checkpoint/38000_model/model.ckpt') -> None:
        self.tf = TFTexasPolicy(model_path=tf_checkpoint)

    def ppo_vs_tf(
        self,
        ppo: PPOPokerPolicy,
        runs: int = 1024,
    ) -> tuple[float, float]:
        env: NoLimitTexasHoldemEnv = create_no_limit_holdem_env()
        result = []
        for run in range(runs):
            env.reset()
            while not env.is_over():
                env_obs = env.observe_current()
                game_obs = env.game.observe_current()
                if env.current_agent_id() == run % 2:
                    env.step(ppo.sample_action(env_obs, game_obs))
                else:
                    env.step(self.tf.sample_action(env_obs, game_obs))
            result.append(env.agent_payoff()[run % 2] * env.payoff_max * 50)
        return np.mean(result), np.std(result)