import os
import time
import numpy as np
from poker_dataset import PokerDataset
from alphaholdem.poker.component.card import Card

class Exporter():
    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.start_game()

    def start_game(self):
        self.ranges = [np.ones(1326) for player in range(2)]
        self.last_history = ""
        self.last_strategy = []

    def to_observation(self, history: str) -> np.ndarray:
        # MATCHSTATE:1:5:r300c/:|Js7s/Ac5d9d
        states = history.split(':')
        player = int(states[1])
        board = states[-1].split('/')[1:]
        in_cards = np.zeros((3, 4, 13), dtype=np.float32)
        if len(board) >= 1:
            for card in Card.from_str_list([board[0][:2], board[0][2:4], board[0][4:6]]):
                in_cards[0][card.suit][card.rank] = 1.0
        if len(board) >= 2:
            card = Card.from_str(board[1])
            in_cards[1][card.suit][card.rank] = 1.0
        if len(board) >= 3:
            card = Card.from_str(board[2])
            in_cards[2][card.suit][card.rank] = 1.0
        action_history = states[3]
        current_player = 0
        street = 0
        max_num_actions_street = 6
        in_action_history = np.zeros((4, 12, 5), dtype=np.float32)
        is_all_in = False
        for street_actions in action_history.split('/'):
            num_action = [0, 0]
            if street == 0:
                is_check = False
            else:
                is_check = True
            for i in range(len(street_actions)):
                action = 0
                if street_actions[i] == 'c':
                    if is_all_in:
                        action = 4 # all in
                    elif is_check:
                        action = 1 # check
                    else:
                        if street == 0:
                            is_check = True
                        action = 2 # call
                if street_actions[i] == 'r':
                    is_check = False
                    if street_actions[i + 1:].startswith('20000'):
                        action = 4 # all in
                        is_all_in = True
                    else:
                        action = 3 # raise
                if action != 0:
                    in_action_history[street][current_player * max_num_actions_street + num_action[current_player]][action] = 1
                    num_action[current_player] += 1
                    current_player = 1 - current_player
            current_player = 1
            street += 1

        in_tensor = np.concatenate((in_cards.flatten(), in_action_history.flatten()))
        return in_tensor

    def is_new_game(self, history: str) -> bool:
        states = history.split(':')
        return states[3] == ""

    def get_last_action(self, history: str) -> int:
        actions = history.split(':')[3].replace('/', '')
        if len(actions) == 0:
            return -1
        if actions[-1] == 'c': # check / call
            return 1
        # WARNING: hardcode 20000
        if len(actions) > 5 and actions[-5:] == "20000": # all in
            return 3
        return 2 # raise

    def update_range(self, history: str) -> None:
        if self.last_history != "":
            last_states = self.last_history.split(':')
            last_player = int(last_states[1])
            last_action = self.get_last_action(history)
            self.ranges[last_player] *= self.last_strategy[last_action]

    # fold check/call raise all_in
    def export(self, folder: str = None) -> None:
        folder = self.folder if folder is None else folder
        t = time.time()
        xs_f = os.path.join(folder, str(int(t)) + '_x.npy')
        ys_f = os.path.join(folder, str(int(t)) + '_y.npy')
        rs_f = os.path.join(folder, str(int(t)) + '_r.npy')
        for name in os.listdir(folder):
            if name == "1724729466679.txt":
            # if name.endswith('.txt'):
                print('Export', name)
                xs = []
                ys = []
                rs = []
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
                                if self.is_new_game(history):
                                    self.start_game()
                                self.update_range(history)
                                player = int(history.split(':')[1])
                                x = self.to_observation(history)
                                # if np.sum(x[:156]) == 0:
                                #     actions = x[156:].reshape(4, 12, 5)
                                #     if actions[0][0][2] == 1:
                                #         print(actions[0])
                                xs.append(x)
                                if len(strategy) == 2:
                                    strategy.append(np.zeros((1326,), dtype=np.float32))
                                    strategy.append(np.zeros((1326,), dtype=np.float32))
                                elif len(strategy) == 3:
                                    strategy.insert(2, np.zeros((1326,), dtype=np.float32))
                                elif len(strategy) != 4:
                                    assert False
                                self.last_strategy = np.array(strategy)
                                self.last_history = history
                                ys.append(strategy)
                                rs.append(self.ranges[player].copy())
                                strategy = []
                            history = line[3:-3]
                        else:
                            strategy.append(list(map(lambda x: float(x), line.strip().split(' '))))
        xs = np.array(xs).astype(np.float32)
        ys = np.array(ys).astype(np.float32)
        rs = np.array(rs).astype(np.float32)
        xs, index = np.unique(xs, axis=0, return_index=True)
        ys = ys[index]
        ys = ys.transpose((0, 2, 1))
        rs = rs[index]

        np.save(xs_f, xs)
        np.save(ys_f, ys)
        np.save(rs_f, rs)

def export():
    poker_dataset = Exporter('/home/clouduser/zcc/Agent')
    poker_dataset.export()

def transpose():
    folder = '/home/clouduser/zcc/Agent'
    for f in os.listdir(folder):
        if f.endswith('y.npy'):
            ys = np.load(os.path.join(folder, f))
            ys = ys.transpose((0, 2, 1))
            np.save(os.path.join(folder, f), ys)

export()
            
# transpose()