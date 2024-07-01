import random
import numpy as np
import time
from pettingzoo.classic import texas_holdem_no_limit_v6
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env import PettingZooEnv

from .poker.no_limit_texas_holdem import NoLimitTexasHoldem
from .poker.aof import AoF
from .poker.component.street import Street
from .poker.component.hand import Hand
from .poker.component.card import Card
from .utils.logger import log

def speed_test():
    """ Result
    [4.18] 6max 1000 runs cost 3.37s
           2max 1000 runs cost 1.68s
    """
    count = 1000
    for num_players in [6, 2]:
        log.info(f"[red]{num_players}max speed test[/red]", extra=dict(markup=True))
        t0 = time.time()
        for _ in range(count):
            seed = random.randint(0, 99999999)
            game = NoLimitTexasHoldem(num_players = num_players)
            game.run(seed = seed)
        log.info(f"{num_players}max {count} runs cost {round(time.time() - t0, 2)}s")

def aof():
    seed = random.randint(0, 99999999)
    log.info(seed)
    game = AoF(num_players = 2, verbose = True)
    history = game.run(seed = seed)

def hunl():
    seed = random.randint(0, 99999999)
    seed = 1758204
    log.info(seed)
    game = NoLimitTexasHoldem(num_players = 2, verbose = True, showdown_street=Street.Showdown)
    history = game.run(seed = seed)

def main():
    seed = random.randint(0, 99999999)
    log.info(seed)
    game = NoLimitTexasHoldem(num_players = 2, verbose = True, showdown_street=Street.Flop)
    history = game.run(seed = seed)
