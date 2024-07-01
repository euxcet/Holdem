import numpy as np
from gymnasium import spaces

from .aof import AoF
from .poker_game_env import PokerGameEnv
from .component.action import Action, ActionType

# [4 * 13]
# [num_agents * 1]

class AoFEnv(PokerGameEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "AoF",
        "is_parallelizable": True,
        "render_fps": 1,
    }

    def __init__(
        self,
        num_players: int = 2,
    ) -> None:
        super().__init__(
            num_players=num_players,
            game=AoF(num_players=num_players, num_runs=10)
        )
        self.max_num_actions_street = 6
        self.observation_spaces = self._to_dict([
            spaces.Dict({
                # 4(hole, flop, turn, river) * 4 * 13
                'observation': spaces.Box(
                    low=0.0, high=1.0, shape=(4, 4, 13), dtype=np.float32
                ),
                # 4(preflop, flop, turn, river) * num_players * max_num_actions_street, 5(fold, check, call, raise, all_in)
                'action_history': spaces.Box( 
                    low=0.0, high=5.0, shape=(4, num_players * self.max_num_actions_street, 5), dtype=np.float32
                ),
                'action_mask': spaces.Box(
                    low=0, high=1, shape=(2,), dtype=np.int8
                ),
            }) for _ in range(self.num_agents)
        ])
        # Fold Check Call All_in Raise_25% Raise_50% Raise_75% Raise_125%
        # Fold All_in
        self.action_spaces = self._to_dict([spaces.Discrete(2) for _ in range(self.num_agents)])

    def step(self, action: int) -> None:
        # self.last_observation.simple_legal_actions[action]
        if action is None:
            super().step(None)
        elif action == 0:
            super().step(self.last_observation.legal_actions[0])
        else:
            super().step(self.last_observation.legal_actions[3])

    def observe(self, agent: str) -> dict:
        observation = self.game.observe(self._agent_name_to_id(agent))
        cards = np.zeros((4, 4, 13), np.float32)
        for hole_card in observation.hole_cards:
            cards[0][hole_card.suit][hole_card.rank] = 1.0
        for id, board_card in enumerate(observation.board_cards):
            if id < 3: # Flop
                cards[1][board_card.suit][board_card.rank] = 1.0
            elif id == 3:
                cards[2][board_card.suit][board_card.rank] = 1.0
            elif id == 4:
                cards[3][board_card.suit][board_card.rank] = 1.0
        # street, player, num_actions_street, action
        action_history = np.zeros((4, self.num_players * self.max_num_actions_street, 5), np.float32)
        action_street_count = [[0, 0, 0, 0]] * self.num_players
        for action in observation.log_action:
            street = action.street.value
            if street >= 4:
                continue
            num_action = action_street_count[action.player][street]
            if num_action >= self.max_num_actions_street:
                continue
            if action.type == ActionType.Raise:
                action_history[street][action.player * self.max_num_actions_street + num_action][action.type.value] = action.raise_pot
            else:
                action_history[street][action.player * self.max_num_actions_street + num_action][action.type.value] = 1
            action_street_count[action.player][street] += 1
        action_mask = np.ones(2, np.int8)
        return {"observation": cards, "action_history": action_history, "action_mask": action_mask}
        