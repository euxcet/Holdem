from __future__ import annotations

import random
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from .component.action import Action
from .component.observation import Observation
from .poker_game import PokerGame

class PokerGameEnv(AECEnv):
    def __init__(
        self,
        num_players: int,
        game: PokerGame,
        circular_train: bool = False,
        payoff_max: float = 200,
    ) -> None:
        super().__init__()
        self.game = game
        self.num_players = num_players
        self.agents = [self._agent_id_to_name(i) for i in range(num_players)]
        self.possible_agents = self.agents
        self.circular_train = circular_train
        self.payoff_max = payoff_max

    def is_over(self) -> bool:
        return self.terminations[self._agent_id_to_name(0)]

    def agent_payoff(self) -> list[float]:
        return list(self._cumulative_rewards.values())

    def current_agent_id(self) -> int:
        return self._agent_name_to_id(self.agent_selection)

    def observation_space(self, agent: int) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: int) -> spaces.Space:
        return self.action_spaces[agent]

    def _agent_id_to_name(self, id: int) -> str:
        return 'agent_' + str(id)

    def _agent_name_to_id(self, s: str) -> id:
        return int(s.split('_')[1])

    def _game_id_to_agent_id(self, game_id: int) -> int:
        return (game_id + self.circular_offset) % self.num_agents

    def _agent_id_to_game_id(self, agent_id: int) -> int:
        return (agent_id - self.circular_offset) % self.num_agents
        
    def _to_dict(self, l: list) -> dict:
        return { x: y for x, y in zip(self.agents, l)}
    
    def step(self, action: Action) -> PokerGameEnv:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)
        observation = self.game.step(action)
        self.last_observation = observation
        self.step_count += 1
        self.rewards = self._to_dict([0] * self.num_agents)
        if self.game.is_over() and self.step_count >= self.num_players:
            for i in range(self.num_agents):
                if self.circular_train:
                    self.rewards_sum[i] += observation.player_payoff[self._agent_id_to_game_id(i)] / self.num_agents / self.payoff_max
                else:
                    self.rewards_sum[i] += observation.player_payoff[self._agent_id_to_game_id(i)] / self.payoff_max
            self.circular_offset += 1
            if self.circular_train:
                if self.circular_offset == self.num_agents:
                    self.rewards = self._to_dict(self.rewards_sum)
                    self.terminations = self._to_dict([True] * self.num_agents)
                    self.truncations = self._to_dict([False] * self.num_agents)
                else:
                    observation = self.game.reset(rng=self.game.game_rng_bak)
                    self.last_observation = observation
                    self.step_count = 0
            else:
                self.rewards = self._to_dict(self.rewards_sum)
                self.terminations = self._to_dict([True] * self.num_agents)
                self.truncations = self._to_dict([False] * self.num_agents)
                
        self.agent_selection = self._agent_id_to_name(self._game_id_to_agent_id(observation.current_player))
        self._accumulate_rewards()
        return self

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> PokerGameEnv:
        self.step_count = 0
        self.circular_offset = 0
        self.agents = [self._agent_id_to_name(i) for i in range(self.num_players)]
        self.possible_agents = self.agents
        observation = self.game.reset(seed = seed)
        self.last_observation = observation
        self.agent_selection = self._agent_id_to_name(self._game_id_to_agent_id(observation.current_player))
        self.rewards = self._to_dict([0] * self.num_agents)
        self.rewards_sum = [0] * self.num_agents
        self._cumulative_rewards = self._to_dict([0] * self.num_agents)
        self.terminations = self._to_dict([False] * self.num_agents)
        self.truncations = self._to_dict([False] * self.num_agents)
        self.infos = self._to_dict([{}] * self.num_agents)
        return self

    def observe(self, agent: str) -> Observation:
        return self.game.observe(agent)

    def render(self) -> None | np.ndarray | str | list:
        raise NotImplementedError

    def state(self) -> np.ndarray:
        raise NotImplementedError(
            "state() method has not been implemented in the environment {}.".format(
                self.metadata.get("name", self.__class__.__name__)
            )
        )

    def close(self):
        pass