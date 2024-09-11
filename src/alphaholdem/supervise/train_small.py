import random
import os
import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.regression import MeanSquaredError
from ..poker.component.card import Card
from ..model.hunl_supervise_model import HUNLSuperviseModel
from ..model.hunl_supervise_small_model import HUNLSuperviseSmallModel
from ..model.hunl_supervise_resnet import HUNLSuperviseResnet
from ..model.hunl_supervise_resnet import HUNLSuperviseResnet50

class PokerDataset(Dataset):
    def __init__(self, folder: str) -> None:
        # self._export(folder)
        # exit(0)
        self.hole_cards_mapping: list[tuple[Card, Card]] = []
        for i in range(52):
            for j in range(i):
                self.hole_cards_mapping.append((Card(rank_first_id=i), Card(rank_first_id=j)))

        self.raw_xs, self.raw_ys = self._load(folder)
        self.raw_ys = self.raw_ys.transpose((0, 2, 1))
        self.lines = self.raw_xs.shape[0]
        # self.xs, self.ys = self._to_hole_cards_input(self.raw_xs, self.raw_ys)
        print(self.raw_xs.shape, self.raw_ys.shape)
        # print(self.xs.shape, self.ys.shape)
        self.length = self.raw_xs.shape[0] * self.raw_ys.shape[1]
        print(self.length)

    def _add_hole_cards(self, line_id: int, hole_id: int) -> tuple[np.ndarray, np.ndarray]:
        hole = self.hole_cards_mapping[hole_id]
        hole_cards = np.zeros((1, 4, 13), dtype=np.float32)
        hole_cards[0][hole[0].suit][hole[0].rank] = 1
        hole_cards[0][hole[1].suit][hole[1].rank] = 1
        data: np.ndarray = self.raw_xs[line_id]
        cards = data[:156].reshape(3, 4, 13)
        action_history = data[156:].reshape(4, 12, 5)
        cards = np.concatenate((hole_cards, cards))
        return np.concatenate((cards.flatten(), action_history.flatten())), np.argmax(self.raw_ys[line_id][hole_id])
        # return np.concatenate((cards.flatten(), action_history.flatten())), self.raw_ys[line_id][hole_id]

    def _to_hole_cards_input(self, raw_xs: np.ndarray, raw_ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for line_id in range(raw_xs.shape[0]):
            for hole_id in range(raw_ys.shape[1]):
                x, y = self._add_hole_cards(line_id, hole_id)
                xs.append(x)
                ys.append(y)
        return np.array(xs), np.array(ys)

    def _load(self, folder: str) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = None, None
        for name in sorted(os.listdir(folder)):
            if name.endswith('x.npy'):
                if xs is None:
                    xs = np.load(os.path.join(folder, name))
                else:
                    xs = np.concatenate((xs, np.load(os.path.join(folder, name))))
            elif name.endswith('y.npy'):
                if ys is None:
                    ys = np.load(os.path.join(folder, name))
                else:
                    ys = np.concatenate((ys, np.load(os.path.join(folder, name))))
            if xs is not None and ys is not None:
                break
        xs, index = np.unique(xs, axis=0, return_index=True)
        ys = ys[index]
        return xs.astype(np.float32), ys.astype(np.float32)

    def _export(self, folder: str) -> None:
        for name in os.listdir(folder):
            if name.endswith('.txt'):
                print('Export', name)
                xs = []
                ys = []
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
                                xs.append(self.to_observation(history))
                                if len(strategy) == 2:
                                    strategy.append(np.zeros((1326,), dtype=np.float32))
                                    strategy.append(np.zeros((1326,), dtype=np.float32))
                                elif len(strategy) == 3:
                                    strategy.insert(2, np.zeros((1326,), dtype=np.float32))
                                elif len(strategy) != 4:
                                    assert False
                                ys.append(strategy)
                                strategy = []
                            history = line[3:-3]
                        else:
                            strategy.append(list(map(lambda x: float(x), line.strip().split(' '))))
                xs_f = os.path.join(folder, name[:-4] + '_x.npy')
                ys_f = os.path.join(folder, name[:-4] + '_y.npy')
                xs = np.array(xs).astype(np.float32)
                ys = np.array(ys).astype(np.float32)
                np.save(xs_f, xs)
                np.save(ys_f, ys)

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
                if street_actions[i] == 'c':
                    if is_all_in:
                        in_action_history[street][current_player * max_num_actions_street + num_action[current_player]][4] = 1
                    elif is_check:
                        in_action_history[street][current_player * max_num_actions_street + num_action[current_player]][1] = 1
                    else:
                        if street == 0:
                            is_check = True
                        in_action_history[street][current_player * max_num_actions_street + num_action[current_player]][2] = 1
                    num_action[current_player] += 1
                    current_player = 1 - current_player
                if street_actions[i] == 'r':
                    is_check = False
                    if street_actions[i + 1:].startswith('20000'): # all in
                        in_action_history[street][current_player * max_num_actions_street + num_action[current_player]][4] = 1
                        is_all_in = True
                    else:
                        in_action_history[street][current_player * max_num_actions_street + num_action[current_player]][3] = 1
                    num_action[current_player] += 1
                    current_player = 1 - current_player
            current_player = 1
            street += 1

        in_tensor = np.concatenate((in_cards.flatten(), in_action_history.flatten()))
        return in_tensor

    def __len__(self):
        return self.length
        # return self.xs.shape[0]

    def __getitem__(self, index: int):
        return self._add_hole_cards(index // 1326, index % 1326)
        # return self._add_hole_cards(random.randint(0, self.lines - 1), random.randint(0, 1325))
        # return self.xs[index], self.ys[index]

def validate(model, loader):
    model.eval()
    mean_square_error = MeanSquaredError().to('cuda')
    mse = 0
    for data, target in tqdm(loader):
        cards = data[:, :208].reshape(-1, 4, 4, 13).to('cuda')
        action_history = data[:, 208:].reshape(-1, 4, 12, 5).to('cuda')
        target = target.flatten(1).to('cuda')
        output = model(cards, action_history)
        mse += mean_square_error(output, target)
    return mse / len(loader)

def train():
    batch_size = 4096 * 16
    train_dataset = PokerDataset('/home/clouduser/zcc/Agent')
    # dataset = PokerDataset('/home/clouduser/zcc/Agent')
    # train_dataset, valid_dataset = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = nn.DataParallel(HUNLSuperviseResnet()).to('cuda')

    # model = HUNLSuperviseResnet50()
    # model.load_state_dict(torch.load('./checkpoint/supervise/small/supervise.pt'))
    # model = nn.DataParallel(model).to('cuda')

    optimizer = optim.Adam(model.parameters(), lr = 0.0005)
    # optimizer = optim.Adam(model.parameters(), lr = 0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, threshold=0.01)
    # criterion = nn.MSELoss().to('cuda')
    criterion = nn.CrossEntropyLoss().to('cuda')
    min_loss = 100000.0

    # print('Origin validation:', validate(model, valid_loader))

    # validate(model, train_loader)

    save_dir = './checkpoint/supervise/small_v1/'
    os.makedirs(save_dir, exist_ok=True)
    wandb.init(project='supervise_small')

    for epoch in range(10000000):
        model.train()
        train_loss = 0
        for data, target in tqdm(train_loader):
            cards = data[:, :208].reshape(-1, 4, 4, 13).to('cuda')
            action_history = data[:,208:].reshape(-1, 4, 12, 5).to('cuda')
            # target = target.flatten(1).to('cuda')
            target = target.to('cuda')
            output = model(cards, action_history)
            loss = criterion(output, target)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss = train_loss / len(train_loader) * 100
        print('Epoch:', epoch, 'Loss:', train_loss)
        wandb.log({
            'epoch': epoch,
            'loss': train_loss,
            'lr': optimizer.param_groups[0]['lr'],
        })
        if train_loss < min_loss:
            min_loss = train_loss
            torch.save(model.module.state_dict(), save_dir + '/supervise.pt')
        if epoch % 100 == 0:
            torch.save(model.module.state_dict(), save_dir + '/supervise_c_' + str(epoch) + '.pt')
    wandb.finish()