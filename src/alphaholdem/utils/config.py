from omegaconf import OmegaConf
from .train_config import TrainConfig

def train_add_arguments():
    ...

def load_config(path: str) -> dict:
    conf = OmegaConf.load(path)
    return conf

def load_train_config(path: str) -> TrainConfig:
    return TrainConfig.load_from_dict(load_config(path))