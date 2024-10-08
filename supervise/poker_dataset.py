import os
import time
import numpy as np
import random
from alphaholdem.poker.component.card import Card
from torch.utils.data import Dataset, IterableDataset, get_worker_info

class PokerDataset(Dataset):
    def __init__(self, folder: str, do_load: bool = True) -> None:
        self.folder = folder
        self.hole_cards_mapping: list[tuple[Card, Card]] = []
        for i in range(52):
            for j in range(i):
                self.hole_cards_mapping.append((Card(rank_first_id=i), Card(rank_first_id=j)))
        if do_load:
            self.raw_xs, self.raw_ys = self._load(folder)
            self.raw_ys = self.raw_ys.transpose((0, 2, 1))
            self.lines = self.raw_xs.shape[0]
            self.length = self.raw_xs.shape[0] * self.raw_ys.shape[1]
            print('Length of the dataset:', self.length)

    def _add_hole_cards(self, line_id: int, hole_id: int) -> tuple[np.ndarray, np.ndarray]:
        hole = self.hole_cards_mapping[hole_id]
        hole_cards = np.zeros((1, 4, 13), dtype=np.float32)
        hole_cards[0][hole[0].suit][hole[0].rank] = 1
        hole_cards[0][hole[1].suit][hole[1].rank] = 1
        data: np.ndarray = self.raw_xs[line_id]
        cards = data[:156].reshape(3, 4, 13)
        action_history = data[156:].reshape(4, 12, 5)
        cards = np.concatenate((hole_cards, cards))
        # return np.concatenate((cards.flatten(), action_history.flatten())), np.argmax(self.raw_ys[line_id][hole_id])
        return np.concatenate((cards.flatten(), action_history.flatten())), self.raw_ys[line_id][hole_id]

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
            print('Load', name)
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
            # if xs is not None and ys is not None:
            #     break
        xs, index = np.unique(xs, axis=0, return_index=True)
        ys = ys[index]
        print('Dataset', xs.shape, ys.shape)
        return xs.astype(np.float32), ys.astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        return self._add_hole_cards(index // 1326, index % 1326)

class IterablePokerDataset(IterableDataset):
    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.hole_cards_mapping: list[tuple[Card, Card]] = []
        for i in range(52):
            for j in range(i):
                self.hole_cards_mapping.append((Card(rank_first_id=i), Card(rank_first_id=j)))

    def _add_hole_cards(self, cards: np.ndarray, actions: np.ndarray, y: np.ndarray, hole_id: int) -> tuple[np.ndarray, np.ndarray]:
        hole = self.hole_cards_mapping[hole_id]
        hole_cards = np.zeros((1, 4, 13), dtype=np.float32)
        hole_cards[0][hole[0].suit][hole[0].rank] = 1
        hole_cards[0][hole[1].suit][hole[1].rank] = 1
        return np.concatenate((np.concatenate((hole_cards, cards)).flatten(), actions.flatten())), y[hole_id]

    def _to_hole_cards_input(self, raw_xs: np.ndarray, raw_ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for line_id in range(raw_xs.shape[0]):
            for hole_id in range(raw_ys.shape[1]):
                x, y = self._add_hole_cards(line_id, hole_id)
                xs.append(x)
                ys.append(y)
        return np.array(xs), np.array(ys)

    def __iter__(self):
        info = get_worker_info()
        for name in sorted(os.listdir(self.folder)):
            if name.endswith('x.npy'):
                xs = np.load(os.path.join(self.folder, name), mmap_mode='r')
                ys = np.load(os.path.join(self.folder, name[:-5] + 'y.npy'), mmap_mode='r')
                if info is None:
                    start = 0
                    end = xs.shape[0]
                else:
                    per_worker = int(np.ceil(xs.shape[0] / float(info.num_workers)))
                    start = info.id * per_worker
                    end = min(start + per_worker, xs.shape[0])
                l = [i for i in range(start, end)]
                random.shuffle(l)
                for i in l:
                    # if np.sum(xs[i][:156]) > 3:
                    # if np.sum(xs[i]) > 0:
                        # continue
                    cards = xs[i][:156].reshape(3, 4, 13).copy()
                    actions = xs[i][156:].reshape(4, 12, 5).copy()
                    y = ys[i].copy()
                    # for j in range(1326):
                    #     yield self._add_hole_cards(cards, actions, y, j)
                    for j in range(100):
                        yield self._add_hole_cards(cards, actions, y, np.random.randint(0, 1326))

class IterableRangePokerDataset(IterableDataset):
    def __init__(self, folder: str) -> None:
        self.folder = folder

    def __iter__(self):
        info = get_worker_info()
        for name in sorted(os.listdir(self.folder)):
            if name.endswith('r.npy'):
                xs = np.load(os.path.join(self.folder, name[:-5] + 'x.npy'), mmap_mode='r')
                ys = np.load(os.path.join(self.folder, name[:-5] + 'y.npy'), mmap_mode='r')
                rs = np.load(os.path.join(self.folder, name[:-5] + 'r.npy'), mmap_mode='r')
                if info is None:
                    start = 0
                    end = xs.shape[0]
                else:
                    per_worker = int(np.ceil(xs.shape[0] / float(info.num_workers)))
                    start = info.id * per_worker
                    end = min(start + per_worker, xs.shape[0])
                l = [i for i in range(start, end)]
                random.shuffle(l)
                for i in l:
                    if np.sum(xs[i][:156]) != 5:
                        continue
                    yield xs[i].copy(), ys[i].copy(), rs[i].copy()