import numpy as np
from gymnasium import spaces

from .component.street import Street
from .range_limit_leduc_holdem import RangeLimitLeducHoldem
from .range_poker_game_env import RangePokerGameEnv
from .component.action import Action, ActionType
from .component.card import Card

# [4 * 13]
# [num_agents * 1]

class RangeLimitLeducHoldemEnv(RangePokerGameEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "AoF",
        "is_parallelizable": True,
        "render_fps": 1,
    }

    def __init__(
        self,
        num_players: int = 2,
        num_runs: int = 100,
        initial_chips: int = 100,
        showdown_street: Street = Street.Turn,
        custom_board_cards: list[Card] = None,
        circular_train: bool = False,
        payoff_max: float = 15,
    ) -> None:
        game = RangeLimitLeducHoldem(
            num_players=num_players,
            initial_chips=[initial_chips] * num_players,
            num_runs=num_runs,
            showdown_street=showdown_street,
            custom_board_cards=custom_board_cards,
        )
        super().__init__(
            num_players=num_players,
            game=game,
            circular_train=circular_train,
            payoff_max=payoff_max,
        )
        self.max_num_actions_street = 2
        self.observation_spaces = self._to_dict([
            spaces.Dict({
                # 2(preflop, flop) * num_players * max_num_actions_street, 5(fold, check, call, raise, all_in)
                'action_history': spaces.Box( 
                    low=0.0, high=1.0, shape=(2, num_players * self.max_num_actions_street, 5), dtype=np.float32
                ),
                'board_card': spaces.Box(
                    low=0.0, high=1.0, shape=(3,), dtype=np.float32
                ),
                'ranges': spaces.Box( 
                    low=0.0, high=1.0, shape=(2, 3), dtype=np.float32
                ),
                'action_mask': spaces.Box(
                    low=0, high=1, shape=(self.game.action_shape,), dtype=np.int8
                ),
            }) for _ in range(self.num_agents)
        ])
        # Fold Check Call All_in Raise
        self.action_spaces = self._to_dict([
            spaces.Box(low=-1, high=1, shape=(3 * self.game.action_shape,), dtype=np.float32)
             for _ in range(self.num_agents)
        ])

    def step(self, action: np.ndarray | None) -> None:
        # obs = self.observe_current()
        # print(action)
        # print(obs['board_card'], obs['action_mask'], obs['action_history'])
        super().step(action)

    def observe_current(self) -> dict:
        return self.observe(self.agent_selection)

    def observe(self, agent: str) -> dict:
        observation = self.game.observe(self._agent_id_to_game_id(self._agent_name_to_id(agent)))
        board_card = np.zeros((3,)).astype(np.float32)
        if len(observation.board_cards) > 0:
            board_card[observation.board_cards[0].rank - 9] = 1.0
        ranges = np.array(observation.player_range).astype(np.float32)
        # street, player, num_actions_street, action
        action_history = np.zeros((2, self.num_players * self.max_num_actions_street, 5), np.float32)
        action_street_count = [[0 for i in range(4)] for j in range(self.num_players)]
        for action in observation.log_action:
            street = action.street.value
            if street >= 2:
                continue
            num_action = action_street_count[action.player][street]
            if num_action >= self.max_num_actions_street:
                continue
            action_history[street][action.player * self.max_num_actions_street + num_action][action.type.value] = 1
            action_street_count[action.player][street] += 1
        action_mask = np.zeros(self.game.action_shape, np.int8)
        for i in range(self.game.action_shape):
            action_mask[i] = 0 if observation.legal_actions[i] is None else 1
        return {
            'ranges': ranges,
            'action_history': action_history,
            'action_mask': action_mask,
            'board_card': board_card,
        }
        