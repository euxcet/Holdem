from ..config.train_config import TrainConfig
from .hunl_conv_model import create_hunl_conv_model
from .hunl_resnet_model import create_hunl_resnet_model
from .range_hunl_conv_model import create_range_hunl_conv_model
from .kuhn_model import create_kuhn_model

def get_model(cfg: TrainConfig):
    if cfg.hyper.model == 'hunl_conv':
        return create_hunl_conv_model(cfg.game.num_action)
    elif cfg.hyper.model == 'range_hunl_conv':
        return create_range_hunl_conv_model(cfg.game.num_action)
    elif cfg.hyper.model == 'hunl_resnet':
        return create_hunl_resnet_model(cfg.game.num_action)
    elif cfg.hyper.model == 'kuhn':
        return create_kuhn_model(cfg.game.num_action)
    raise Exception
