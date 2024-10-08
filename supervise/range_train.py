import os
api_key = os.getenv('WANDB_API_KEY')
if api_key is None:
    raise ValueError("Please set variable: WANDB_API_KEY")

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import argparse
import wandb

from lightning import Trainer, LightningModule, LightningDataModule, seed_everything
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from alphaholdem.poker.component.card import Card
from alphaholdem.model.hunl_supervise_range_model import HUNLSuperviseRangeModel
from poker_dataset import IterableRangePokerDataset

# torch.set_float32_matmul_precision("medium")

class DeepStackDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
    
    def setup(self, stage: str):
        self.dataset = IterableRangePokerDataset(self.data_dir)
        self.train_dataset = self.dataset
        self.valid_dataset = self.dataset
        # self.dataset = PokerDataset(self.data_dir)
        # self.train_dataset, self.valid_dataset = random_split(self.dataset, [0.95, 0.05])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=2, shuffle=False, pin_memory=True)

class DeepStackModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = HUNLSuperviseRangeModel()
        self.criterion = nn.MSELoss()
        self.lr = 1e-4

    def training_step(self, batch):
        # 1326 * 4
        x, y, r = batch
        cards = x[:, :156].reshape(-1, 3, 4, 13)
        actions = x[:, 156:].reshape(-1, 4, 12, 5)
        output = self.model(cards, actions)
        # loss = self.criterion(output * r.view(-1, 1326, 1), y * r.view(-1, 1326, 1))
        loss = self.criterion(output, y)
        # print(y.shape, r.shape)
        # print((y * r.view(-1, 1326, 1)).shape)
        # exit(0)
        self.log(f'train_loss', loss, sync_dist=True, prog_bar=True)
        return loss

    # def validation_step(self, batch):
    #     x, y = batch
    #     cards = x[:, :208].reshape(-1, 4, 4, 13)
    #     actions = x[:,208:].reshape(-1, 4, 12, 5)
    #     output = self.model(cards, actions)
    #     loss = self.criterion(output, y)
    #     self.log(f'valid_loss', loss, prog_bar=True, sync_dist=True)
    #     return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def get_checkpoint(run: str) -> str:
    folder = './wandb/supervise/' + run + '/checkpoints/'
    max_step = 0
    result = ""
    for c in os.listdir(folder):
        if c.endswith('ckpt'):
            v = int(c.split('.')[0].split('=')[-1])
            if v > max_step:
                max_step = v
                result = os.path.join(folder, c)
    return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    seed_everything(1)
    batch_size = 16
    data = DeepStackDataModule('/home/clouduser/zcc/Agent', batch_size=batch_size)
    if args.checkpoint is not None:
        deepstack = DeepStackModule.load_from_checkpoint(get_checkpoint(args.checkpoint))
    else:
        deepstack = DeepStackModule()
    deepstack.lr = args.lr
    # /home/clouduser/zcc/Holdem/supervise/wandb/supervise/0pgny90u/checkpoints/
    wandb.login(key=api_key)
    wandb_logger = WandbLogger(
        log_model="all",
        project="supervise",
        save_dir="./wandb",
        name=f"bs-{batch_size}-lr-{deepstack.lr}-{time.strftime('%Y-%m-%d--%H:%M:%S', time.localtime())}",
    )
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=-1,
        logger=wandb_logger,
        callbacks=[ ModelCheckpoint(every_n_epochs=1, save_top_k=1, monitor='train_loss'), ],
        default_root_dir='./log',
    )
    tuner = Tuner(trainer)
    trainer.fit(model=deepstack, datamodule=data)
