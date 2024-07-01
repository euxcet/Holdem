from __future__ import annotations

import copy
import numpy as np
from numpy.random import Generator
from .deck import Deck
from .card import Card

class Dealer():
    def __init__(self, deck: Deck) -> None:
        self.origin_deck = deck
        self.deck = copy.deepcopy(deck)
        self.rng = np.random.default_rng()

    def reset(self, rng: Generator) -> Dealer:
        self.deck = copy.deepcopy(self.origin_deck)
        self.rng = rng
        return self

    def shuffle(self) -> Dealer:
        self.deck.shuffle(self.rng)
        return self

    def burn(self):
        self.deck.pop()
    
    def burn(self, cards: list[Card]):
        self.deck.burn(cards)

    def deal_one(self) -> Card:
        return self.deck.pop()

    def deal_without_pop(self, count: int) -> list[Card]:
        if count <= 0:
            return []
        self.deck.shuffle(self.rng)
        return self.deck.get(count)

    def deal(self, count: int) -> list[Card]:
        return [self.deck.pop() for _ in range(count)]