from __future__ import annotations

from enum import Enum
import functools
from .card import Card

class HandType(Enum):
    HighCard = 0
    OnePair = 1
    TwoPair = 2
    ThreeOfAKind = 3
    Straight = 4
    Flush = 5
    FullHouse = 6
    FourOfAKind = 7
    StraightFlush = 8

    @staticmethod
    def order() -> list[HandType]:
        return [
            HandType.HighCard, HandType.OnePair, HandType.TwoPair,
            HandType.ThreeOfAKind, HandType.Straight, HandType.Flush,
            HandType.FullHouse, HandType.FourOfAKind, HandType.StraightFlush
        ]
        
    # In short deck poker, a flush beats a full house.
    @staticmethod
    def short_deck_order() -> list[HandType]:
        return [
            HandType.HighCard, HandType.OnePair, HandType.TwoPair,
            HandType.ThreeOfAKind, HandType.Straight, HandType.FullHouse,
            HandType.Flush, HandType.FourOfAKind, HandType.StraightFlush
        ]

@functools.total_ordering
class Hand():
    def __init__(
        self,
        cards: list[Card|str],
        order: list[HandType] = HandType.order(),
    ) -> None:
        if type(cards[0]) is str:
            cards = Card.from_str_list(cards)
        tmp_cards = [(cards.count(x) * 100 + x.rank_first_id, x) for x in cards]
        tmp_cards.sort(reverse=True)
        self.cards = [x[1] for x in tmp_cards]
        self.hand_strength = [0] * len(HandType)
        for hand in HandType:
            self.hand_strength[hand.value] = order.index(hand)

        if self._contains_straight() and self._contains_flush():
            self.hand = HandType.StraightFlush
        elif self._contains_four_of_a_kind():
            self.hand = HandType.FourOfAKind
        elif self._contains_full_house():
            self.hand = HandType.FullHouse
        elif self._contains_flush():
            self.hand = HandType.Flush
        elif self._contains_straight():
            self.hand = HandType.Straight
        elif self._contains_three_of_a_kind():
            self.hand = HandType.ThreeOfAKind
        elif self._contains_two_pair():
            self.hand = HandType.TwoPair
        elif self._contains_pair():
            self.hand = HandType.OnePair
        else:
            self.hand = HandType.HighCard
        
        if self.cards[0].rank == 12 and (self.hand == HandType.Straight or self.hand == HandType.StraightFlush):
            self.cards = self.cards[1:] + self.cards[0:1]
    
    @staticmethod
    def worst_hand() -> Hand:
        # 2c 3c 4c 5c 7d
        return Hand([
            Card(rank = 0, suit = 0),
            Card(rank = 1, suit = 0),
            Card(rank = 2, suit = 0),
            Card(rank = 3, suit = 0),
            Card(rank = 5, suit = 1),
        ])

    def _contains_pair(self) -> bool:
        return len(self.cards) >= 2 and self.cards[0] == self.cards[1]

    def _contains_two_pair(self) -> bool:
        return len(self.cards) >= 4 and (self.cards[0] == self.cards[1] and self.cards[2] == self.cards[3] or \
               self.cards[0] == self.cards[1] and self.cards[3] == self.cards[4])
    
    def _contains_three_of_a_kind(self) -> bool:
        return len(self.cards) >= 3 and (self.cards[0] == self.cards[1] and self.cards[1] == self.cards[2])

    def _contains_straight(self) -> bool:
        return len(self.cards) >= 5 and ((self.cards[0].is_next_rank(self.cards[1]) and self.cards[1].is_next_rank(self.cards[2]) and \
               self.cards[2].is_next_rank(self.cards[3]) and self.cards[3].is_next_rank(self.cards[4])) or \
               (self.cards[4].is_next_rank(self.cards[0]) and self.cards[1].is_next_rank(self.cards[2]) and \
               self.cards[2].is_next_rank(self.cards[3]) and self.cards[3].is_next_rank(self.cards[4])))
    
    def _contains_flush(self) -> bool:
        return len(self.cards) >= 5 and (self.cards[0].suit == self.cards[1].suit and self.cards[1].suit == self.cards[2].suit and \
               self.cards[2].suit == self.cards[3].suit and self.cards[3].suit == self.cards[4].suit)

    def _contains_full_house(self) -> bool:
        return len(self.cards) >= 5 and (self.cards[0] == self.cards[1] and self.cards[1] == self.cards[2] and \
               self.cards[3] == self.cards[4])

    def _contains_four_of_a_kind(self) -> bool:
        return len(self.cards) >= 4 and (self.cards[0] == self.cards[1] and self.cards[1] == self.cards[2] and \
               self.cards[2] == self.cards[3])

    def __lt__(self, other: Hand):
        if self.hand_strength[self.hand.value] < other.hand_strength[other.hand.value]:
            return True
        if self.hand_strength[self.hand.value] > other.hand_strength[other.hand.value]:
            return False
        for x, y in zip(self.cards, other.cards):
            if x < y:
                return True
            if x > y:
                return False
        return False

    def __eq__(self, other: Hand):
        for x, y in zip(self.cards, other.cards):
            if x != y:
                return False
        return self.hand_strength[self.hand.value] == other.hand_strength[other.hand.value]

    def __str__(self) -> str:
        return f"{self.hand} {self.cards}"

    def __repr__(self) -> str:
        return self.__str__()