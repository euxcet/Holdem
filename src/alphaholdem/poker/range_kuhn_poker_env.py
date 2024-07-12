import numpy as np
from gymnasium import spaces

from .component.street import Street
from .range_kuhn_poker import RangeKuhnPoker
from .range_poker_game_env import RangePokerGameEnv
from .component.action import Action, ActionType
from .component.card import Card

# [4 * 13]
# [num_agents * 1]

class RangeKuhnPokerEnv(RangePokerGameEnv):
    metadata = {
        "name": "Kuhn",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        circular_train: bool = False,
        payoff_max: float = 2,
    ) -> None:
        game = RangeKuhnPoker()
        super().__init__(
            num_players=2,
            game=game,
            circular_train=circular_train,
            payoff_max=payoff_max,
        )
        self.max_num_actions_street = 6
        self.observation_spaces = self._to_dict([
            spaces.Dict({
                'observation': spaces.Box(
                    low=0.0, high=1.0, shape=(3,), dtype=np.float32
                ),
                'action_history': spaces.Box( 
                    low=0.0, high=5.0, shape=(2, self.game.action_shape), dtype=np.float32
                ),
                'action_mask': spaces.Box(
                    low=0, high=1, shape=(self.game.action_shape,), dtype=np.int8
                ),
            }) for _ in range(self.num_agents)
        ])
        # Fold Check Call Raise
        self.action_spaces = self._to_dict([
            spaces.Box(low=0.0, high=1, shape=(1,), dtype=np.float32)
                for _ in range(self.num_agents)
        ])

    def step(self, action: list[float]) -> None:
        if action is None:
            super().step(None)
        else:
            actions = list(filter(lambda x: x is not None, self.last_observation.legal_actions))
            if np.random.random() < action[0]:
                super().step(actions[0])
            else:
                super().step(actions[1])

    def observe_current(self) -> dict:
        return self.observe(self.agent_selection)

    def observe(self, agent: str) -> dict:
        observation = self.game.observe(self._agent_id_to_game_id(self._agent_name_to_id(agent)))
        cards = np.zeros((3), np.float32)
        cards[observation.hole_cards[0].rank - 9] = 1.0
        action_history = np.zeros((2, self.game.action_shape), np.float32)
        for action_id, action in enumerate(observation.log_action):
            if action_id < 2:
                action_history[action_id][action.type.value] = 1
        action_mask = np.zeros(self.game.action_shape, np.int8)
        for i in range(self.game.action_shape):
            action_mask[i] = 0 if observation.legal_actions[i] is None else 1
        return {"observation": cards, "action_history": action_history, "action_mask": action_mask}
        