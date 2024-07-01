import random
from typing_extensions import override

from ..component.action import Action
from ..component.observation import Observation
from .agent import Agent

class RandomAgent(Agent):
    def __init__(self):
        ...

    @override
    def step(self, observation: Observation) -> Action:
        # print("Legal actions:")
        # print('\n'.join(map(str, observation.simple_legal_actions)))
        # print()
        return self.rng.choice(list(filter(lambda x: x != None, observation.legal_actions)))