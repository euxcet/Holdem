from __future__ import annotations

from numpy.random import Generator
from .card import Card
import random

class Deck():
    def __init__(self, cards: list[Card]) -> None:
        self.cards = cards

    def shuffle(self, rng: Generator) -> Deck:
        rng.shuffle(self.cards)
        return self

    def get(self, count: int) -> list[Card]:
        return self.cards[:count]

    def pop(self) -> Card:
        return self.cards.pop()

    def burn(self, cards: list[Card]):
        for card in cards:
            self.cards.remove(card)

    @staticmethod
    def sub_deck(begin: int, end: int) -> Deck:
        return Deck([Card(suit_first_id = i) for i in range(begin, end)])
        
    @staticmethod
    def deck_52() -> Deck:
        return Deck.sub_deck(0, 52)

    @staticmethod
    def deck_54() -> Deck:
        return Deck.sub_deck(0, 54)

    @staticmethod
    def deck_leduc() -> Deck:
        return Deck(Card.from_str_list(['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']))