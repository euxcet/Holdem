from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from ..component.action import Action
from ..component.observation import Observation

class Agent(ABC):
    def __init__(self):
        self.rng = np.random.default_rng()

    def set_rng(self, rng: Generator) -> None:
        self.rng = rng

    @abstractmethod
    def step(self, observation: Observation) -> Action:
        ...