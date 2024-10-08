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
from alphaholdem.model.hunl_supervise_model import HUNLSuperviseModel, HUNLSuperviseSimpleModel
from alphaholdem.model.hunl_supervise_resnet import HUNLSuperviseResnet, HUNLSuperviseResnet50
from poker_dataset import PokerDataset, IterablePokerDataset

# torch.set_float32_matmul_precision("medium")

class DeepStackDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
    
    def setup(self, stage: str):
        self.dataset = IterablePokerDataset(self.data_dir)
        self.train_dataset = self.dataset
        self.valid_dataset = self.dataset
        # self.dataset = PokerDataset(self.data_dir)
        # self.train_dataset, self.valid_dataset = random_split(self.dataset, [0.95, 0.05])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=32, shuffle=False, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=32, shuffle=False, pin_memory=True)

class DeepStackModule(LightningModule):
    def __init__(self):
        super().__init__()
        # self.model = HUNLSuperviseSimpleModel()
        self.model = HUNLSuperviseResnet()
        # self.model = HUNLSuperviseResnet50()
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()
        self.lr = 1e-3
        # self.save_hyperparameters()

    def training_step(self, batch):
        x, y, r = batch
        # output = self.model(x)
        cards = x[:, :208].reshape(-1, 4, 4, 13)
        actions = x[:, 208:].reshape(-1, 4, 12, 5)
        output = self.model(cards, actions)
        loss = self.criterion(output, y)
        # print(output[:3], y[:3], loss)
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
    
if __name__ == '__main__':
    seed_everything(1)
    wandb.login(key=api_key)
    # batch_size = 1024
    batch_size = 8192
    # batch_size = 8192 * 2
    data = DeepStackDataModule('/home/clouduser/zcc/Agent', batch_size=batch_size)
    deepstack = DeepStackModule()
    wandb_logger = WandbLogger(
        log_model="all",
        project="supervise",
        save_dir="./wandb",
        name=f"bs-{batch_size}-lr-{deepstack.lr}-{time.strftime('%Y-%m-%d--%H:%M:%S', time.localtime())}",
    )
    trainer = Trainer(
        accelerator='gpu',
        devices=4,
        max_epochs=-1,
        logger=wandb_logger,
        callbacks=[ ModelCheckpoint(every_n_epochs=1, save_top_k=1, monitor='train_loss'), ],
        default_root_dir='./log',
    )
    tuner = Tuner(trainer)
    # print('origin lr', deepstack.lr)
    # tuner.lr_find(model=deepstack, datamodule=data)
    # print('lr', deepstack.lr)
    trainer.fit(model=deepstack, datamodule=data)
