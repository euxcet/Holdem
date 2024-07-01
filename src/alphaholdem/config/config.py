from argparse import Namespace, ArgumentParser
from omegaconf import OmegaConf
from .train_config import TrainConfig

def train_add_arguments(parser: ArgumentParser):
    parser.add_argument('-s', '--showdown-street', type=str, default='showdown')
    parser.add_argument('-r', '--num-runs', type=int, default=100)
    parser.add_argument('-p', '--num-players', type=int, default=2)
    parser.add_argument('--initial-chips', type=int, default=200)
    parser.add_argument('--win-rate-threshold', type=int, default=100)
    parser.add_argument('--opponent-count', type=int, default=4)
    parser.add_argument('--num-update-iter', type=int, default=20)
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--num-rollout-workers', type=int, default=16)
    parser.add_argument('--num-envs-per-worker', type=int, default=4)
    parser.add_argument('--num-cpus-per-worker', type=float, default=0.125)
    parser.add_argument('--num-gpus-per-worker', type=float, default=0.03125)
    parser.add_argument('--num-gpus-algorithm', type=float, default=0.5)
    parser.add_argument('--num-learner-workers', type=int, default=0)
    parser.add_argument('--train-batch-size', type=int, default=16384)
    parser.add_argument('--circular-train', type=bool, default=False)
    parser.add_argument('--legal-raise-pot-size', type=str, default="0.75")
    parser.add_argument('--custom-board-cards', type=str, default=None)
    parser.add_argument('--kl-coeff', type=float, default=0.2)
    parser.add_argument('--kl-target', type=float, default=0.003)
    parser.add_argument('--vf-clip-param', type=float, default=10.0)
    parser.add_argument('--vf-loss-coeff', type=float, default=1.0)
    parser.add_argument('--clip-param', type=float, default=0.2)
    parser.add_argument('--entropy-coeff', type=float, default=0.01)
    parser.add_argument('--payoff-max', type=float, default=200.0)
    parser.add_argument('--learning-rate', type=float, default=0.00005)
    parser.add_argument('--sgd-minibatch-size', type=int, default=256)
    parser.add_argument('--num-sgd-iter', type=int, default=30)
    parser.add_argument('--checkpoint-num-to-keep', type=int, default=10)
    parser.add_argument('--checkpoint-frequency', type=int, default=10)

def load_config(path: str) -> dict:
    conf = OmegaConf.load(path)
    return conf

def load_train_config(path: str, args: Namespace) -> TrainConfig:
    return TrainConfig.load_from_dict(load_config(path))