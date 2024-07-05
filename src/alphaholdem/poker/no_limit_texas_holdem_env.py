import numpy as np
from gymnasium import spaces

from .component.street import Street
from .no_limit_texas_holdem import NoLimitTexasHoldem
from .poker_game_env import PokerGameEnv
from .component.action import Action, ActionType
from .component.card import Card

# [4 * 13]
# [num_agents * 1]

class NoLimitTexasHoldemEnv(PokerGameEnv):
    metadata = {
        "name": "NoLimitTexas",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_players: int = 2,
        num_runs: int = 100,
        initial_chips: int = 200,
        showdown_street: Street = Street.Showdown,
        custom_board_cards: list[Card] = None,
        circular_train: bool = False,
        legal_raise_pot_size: list[float] = [0.5, 0.75, 1, 1.5, 2],
        payoff_max: float = 200,
    ) -> None:
        game = NoLimitTexasHoldem(
            num_players=num_players,
            initial_chips=[initial_chips] * num_players,
            num_runs=num_runs,
            showdown_street=showdown_street,
            custom_board_cards=custom_board_cards,
            legal_raise_pot_size=legal_raise_pot_size,
        )
        super().__init__(
            num_players=num_players,
            game=game,
            circular_train=circular_train,
            payoff_max=payoff_max,
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
                    low=0, high=1, shape=(self.game.action_shape,), dtype=np.int8
                ),
            }) for _ in range(self.num_agents)
        ])
        # Fold Check Call All_in Raise_25% Raise_50% Raise_75% Raise_125%
        self.action_spaces = self._to_dict([spaces.Discrete(self.game.action_shape) for _ in range(self.num_agents)])

    def step(self, action: int) -> None:
        if action is None:
            super().step(None)
        else:
            super().step(self.last_observation.legal_actions[action])

    def observe_current(self) -> dict:
        return self.observe(self.agent_selection)
        # return self.observe(self._agent_id_to_name(self._game_id_to_agent_id(self.game.current_player)))

    def observe(self, agent: str) -> dict:
        observation = self.game.observe(self._agent_id_to_game_id(self._agent_name_to_id(agent)))
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
        action_street_count = [[0 for i in range(4)] for j in range(self.num_players)]
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
        action_mask = np.zeros(self.game.action_shape, np.int8)
        for i in range(self.game.action_shape):
            action_mask[i] = 0 if observation.legal_actions[i] is None else 1
        return {"observation": cards, "action_history": action_history, "action_mask": action_mask}
        