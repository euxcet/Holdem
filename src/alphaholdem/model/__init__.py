from ..config.train_config import TrainConfig
from .hunl_conv_model import HUNLConvModel
from .hunl_resnet_model import HUNLResnetModel
from .range_hunl_conv_model import create_range_hunl_conv_model
from .kuhn_model import KuhnModel
from .range_kuhn_model import RangeKuhnModel
from .range_leduc_model import RangeLeducModel

def get_model(cfg: TrainConfig):
    if cfg.hyper.model == 'hunl_conv':
        return HUNLConvModel
    elif cfg.hyper.model == 'range_hunl_conv':
        return create_range_hunl_conv_model(cfg.game.num_action)
    elif cfg.hyper.model == 'hunl_resnet':
        return HUNLResnetModel
    elif cfg.hyper.model == 'kuhn':
        return KuhnModel
    elif cfg.hyper.model == 'range_kuhn':
        return RangeKuhnModel
    elif cfg.hyper.model == 'range_leduc':
        return RangeLeducModel
    raise Exception
