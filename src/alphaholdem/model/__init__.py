from ..config.train_config import TrainConfig
from .hunl_conv_model import HUNLConvModel
from .hunl_resnet_model import HUNLResnetModel
from .range_hunl_conv_model import create_range_hunl_conv_model
from .kuhn_model import KuhnModel

def get_model(cfg: TrainConfig):
    if cfg.hyper.model == 'hunl_conv':
        return HUNLConvModel
    elif cfg.hyper.model == 'range_hunl_conv':
        return create_range_hunl_conv_model(cfg.game.num_action)
    elif cfg.hyper.model == 'hunl_resnet':
        return HUNLResnetModel
    elif cfg.hyper.model == 'kuhn':
        return KuhnModel
    raise Exception
