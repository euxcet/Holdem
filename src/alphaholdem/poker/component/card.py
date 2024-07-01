from __future__ import annotations
import functools

@functools.total_ordering
class Card:
    """
    id (0 ~ 51)
        rank = id % 13
        suit = id // 13
    rank(0 ~ 12)
        0 -> 2
        8  -> T
        9 -> J
        10 -> Q
        11 -> K
        12  -> A

    suit(0 ~ 3)
        0 -> club
        1 -> diamond
        2 -> heart
        3 -> spade
    """
    rank_num2str_dict = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8',
                         7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}
    rank_str2num_dict = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
                         '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    suit_num2str_dict = {0: 'c', 1: 'd', 2: 'h', 3: 's'}
    suit_str2num_dict = {'c': 0, 'd': 1, 'h': 2, 's': 3}
    def __init__(
        self,
        rank: int = 0,
        suit: int = 0,
        suit_first_id = None,
        rank_first_id = None
    ) -> None:
        if suit_first_id != None:
            self.rank = suit_first_id % 13
            self.suit = suit_first_id // 13
        elif rank_first_id != None:
            self.rank = rank_first_id // 4
            self.suit = rank_first_id % 4
        else:
            self.rank = rank
            self.suit = suit
        self.suit_first_id = self.suit * 13 + self.rank
        self.rank_first_id = self.rank * 4 + self.suit

    def is_prev_rank(self, other: Card) -> bool:
        return self.suit_first_id < 52 and other.suit_first_id < 52 and self.rank == (other.rank + 12) % 13

    def is_next_rank(self, other: Card) -> bool:
        return self.suit_first_id < 52 and other.suit_first_id < 52 and self.rank == (other.rank + 1) % 13
    
    @staticmethod
    def from_str(s: str) -> Card:
        assert len(s) == 2
        return Card(
            rank=Card.rank_from_str(s[0]),
            suit=Card.suit_from_str(s[1]),
        )

    @staticmethod
    def rank_str(rank: int) -> str:
        return Card.rank_num2str_dict[rank]

    @staticmethod
    def rank_from_str(s: str) -> int:
        return Card.rank_str2num_dict[s]

    @staticmethod
    def suit_str(suit: int) -> str:
        return Card.suit_num2str_dict[suit]

    @staticmethod
    def suit_from_str(suit: int) -> str:
        return Card.suit_str2num_dict[suit]

    @staticmethod
    def from_str_list(cards: list[str]) -> list[Card]:
        return [Card.from_str(c) for c in cards]
 
    # Suits don't affect the strength of cards in Texas Hold'em
    def __lt__(self, other: Card) -> bool:
        return self.rank < other.rank

    def __eq__(self, other: Card) -> bool:
        return self.rank == other.rank

    def __str__(self) -> str:
        if self.suit_first_id == 52:
            return 'BJ'
        elif self.suit_first_id == 53:
            return 'CJ'
        return Card.rank_str(self.rank) + Card.suit_str(self.suit)

    def __repr__(self) -> str:
        return self.__str__()