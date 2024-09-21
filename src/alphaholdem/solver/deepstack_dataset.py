import os
import numpy as np
from ..poker.component.card import Card
from ..poker.utils.format_utils import acpc_to_tensor
from ..poker.component.observation import Observation
from ..poker.no_limit_texas_holdem_env import NoLimitTexasHoldemEnv
from ..poker.component.card import Card
from ..poker.component.observation import Observation
from ..poker.component.street import Street

class DeepStackGame():
    def __init__(self, observations: list[np.ndarray], policies: list[np.ndarray]) -> None:
        self.observations = observations
        self.policies = policies
    
    def num_steps(self) -> int:
        return len(self.observations)

    def format_action(self, action: int) -> int:
        # fold check call raise all_in
        # ->
        # fold check call all_in raise
        return [0, 1, 2, 4, 3][action]

    def get_board_cards(self, step: int) -> list[Card]:
        cards: np.ndarray = self.observations[step][:156].reshape(3, 4, 13)
        board_cards = []
        for i in range(3):
            for j in range(4):
                for k in range(13):
                    if cards[i][j][k] > 0.5:
                        board_cards.append(Card(rank = k, suit = j))
        return board_cards

    def get_actions(self, step: int) -> tuple[list, list]:
        actions: np.ndarray = self.observations[step][156:].reshape(4, 12, 5)
        player0_actions = []
        player1_actions = []
        for i in range(4):
            for j in range(12):
                for k in range(5):
                    if actions[i][j][k] > 0.5:
                        if j < 6:
                            player0_actions.append(k)
                        else:
                            player1_actions.append(k)
        return player0_actions, player1_actions

    def get_observation(self, step: int) -> Observation:
        step = min(step, self.num_steps() - 1)

        board_cards = self.get_board_cards(step)
        player0_actions, player1_actions = self.get_actions(step)

        player0_pointer = 0
        player1_pointer = 0

        env = NoLimitTexasHoldemEnv(
            num_players=2,
            initial_chips=200,
            showdown_street=Street.Showdown,
            custom_board_cards=board_cards,
            raise_pot_size=[1],
            legal_raise_pot_size=[1],
        )
        env.reset()
        for i in range(len(player0_actions) + len(player1_actions)):
            game_obs = env.game.observe_current()

            if env.current_agent_id() == 0:
                env.step(self.format_action(player0_actions[player0_pointer]))
                player0_pointer += 1
            else:
                env.step(self.format_action(player1_actions[player1_pointer]))
                player1_pointer += 1

        game_obs = env.game.observe_current()
        if step < self.num_steps() - 1:
            next_p0_actions, next_p1_actions = self.get_actions(step + 1)
            if env.current_agent_id() == 0:
                next_action = self.format_action(next_p0_actions[-1])
            else:
                next_action = self.format_action(next_p1_actions[-1])
            for i in range(len(game_obs.legal_actions)):
                if i != next_action:
                    game_obs.legal_actions[i] = None
            
        return game_obs

    def get_policy(self, step: int) -> np.ndarray:
        step = min(step, self.num_steps() - 1)
        return self.policies[step].transpose((1, 0))

class DeepStackDataset():
    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.xs, self.ys = self.load(folder)
        self.games = self.split()

    def load(self, folder: str = None) -> tuple[np.ndarray, np.ndarray]:
        folder = self.folder if folder is None else folder
        xs = []
        ys = []
        for name in os.listdir(folder):
            if name.endswith('.txt'):
                print('Load', name)
                history = None
                strategy = []
                with open(os.path.join(folder, name), 'r') as f:
                    while True:
                        line = f.readline().strip()
                        if len(line) < 3:
                            break
                        line = ''.join(x for x in line if x.isprintable())
                        if line.startswith('[0m'):
                            if history is not None:
                                xs.append(acpc_to_tensor(history))
                                if len(strategy) == 2: # fold call
                                    strategy.append(np.zeros((1326,), dtype=np.float32)) # raise
                                    strategy.append(np.zeros((1326,), dtype=np.float32)) # all_in
                                elif len(strategy) == 3: # fold check all_in
                                    strategy.insert(2, np.zeros((1326,), dtype=np.float32)) # raise
                                elif len(strategy) != 4:
                                    assert False
                                ys.append(strategy)
                                strategy = []
                                if len(xs) == 10:
                                    break
                            history = line[3:-3]
                        else:
                            strategy.append(list(map(lambda x: float(x), line.strip().split(' '))))
        xs = np.array(xs).astype(np.float32)
        ys = np.array(ys).astype(np.float32)
        return xs, ys

    # Tensor: [cards(3 * 4 * 13), actions(4 * 12 * 5)]
    def split(self) -> list[DeepStackGame]:
        games = []
        observations = []
        policies = []
        for i in range(self.xs.shape[0]):
            actions: np.ndarray = self.xs[i][156:].reshape(4, 12, 5)
            # print(np.sum(actions), np.all(np.equal(self.xs[i], self.xs[i - 1])))
            if np.sum(actions) == 0 and len(observations) > 0:
                games.append(DeepStackGame(observations.copy(), policies.copy()))
                observations.clear()
                policies.clear()
            observations.append(self.xs[i])
            policies.append(self.ys[i])
        print('games', len(games))
        return games



# 0.0 False
# 1.0 False
# 1.0 False
# 1.0 True
# 2.0 False
# 3.0 False
# 4.0 False
# 5.0 False
# 0.0 False
# 1.0 False
# 1.0 False