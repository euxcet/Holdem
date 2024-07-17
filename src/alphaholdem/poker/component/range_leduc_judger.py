from .card import Card
from .hand import Hand, HandType
from .judger import Judger

class RangeLeducJudger(Judger):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def judge(
        self,
        pot: int,
        board_cards: list[Card],
        player_fold: list[bool],
    ) -> list[int]:
        payoff: list[int] = [0, 0]
        if player_fold.count(False) == 1:
            winner = player_fold.index(False)
            payoff[winner] += pot
            return payoff

        # JQ 1/10 # JK 1/10 # QJ 1/10 # QQ 1/10
        # QK 1/5 # KJ 1/10 # KQ 1/5 # KK 1/10
        for player0_card in Card.from_str_list(['Jc', 'Qc', 'Kc']):
            for player1_card in Card.from_str_list(['Jc', 'Qc', 'Kc']):
                if player0_card == player1_card and player0_card == board_cards[0]:
                    prob = 0
                elif player0_card == player1_card or player0_card == board_cards[0] or player1_card == board_cards[0]:
                    prob = 1 / 10
                else:
                    prob = 1 / 5

                player_best_hand =[
                    self.get_best_hand([player0_card], board_cards),
                    self.get_best_hand([player1_card], board_cards),
                ]
                if player_best_hand[0] > player_best_hand[1]:
                    this_payoff = [pot // 2, -pot // 2]
                elif player_best_hand[0] < player_best_hand[1]:
                    this_payoff = [-pot // 2, pot // 2]
                else:
                    this_payoff = [0, 0]
                payoff[0] += this_payoff[0] * prob
                payoff[1] += this_payoff[1] * prob
        return payoff

    # TODO: to be optimized
    def choose(
        self,
        cards: list[Card],
        num: int,
    ) -> list[list[Card]]:
        result: list[list[Card]] = []
        for bit in range(1 << len(cards)):
            count_1 = (bit & 0x55) + ((bit >> 1) & 0x55)
            count_1 = (count_1 & 0x33) + ((count_1 >> 2) & 0x33)
            count_1 = (count_1 & 0x0f) + ((count_1 >> 4) & 0x0f)
            if count_1 == num:
                sub: list[Card] = []
                for i in range(len(cards)):
                    if (bit & (1 << i)) > 0:
                        sub.append(cards[i])
                result.append(sub)
        return result

    def get_best_hand(
        self,
        hole_cards: list[Card],
        board_cards: list[Card],
    ) -> Hand:
        # for kuhn and leduc
        if len(hole_cards) + len(board_cards) <= 2:
            return Hand(hole_cards + board_cards)

        best_hand: Hand = None
        for num_hole_cards in self.allow_num_hole_cards:
            hole_cards_parts = self.choose(hole_cards, num_hole_cards)
            board_cards_parts = self.choose(board_cards, self.HAND_CARD_SIZE - num_hole_cards)
            for hole_cards_part in hole_cards_parts:
                for board_cards_part in board_cards_parts:
                    hand = Hand(hole_cards_part + board_cards_part, order = self.hand_order)
                    if best_hand is None or hand > best_hand:
                        best_hand = hand
        return best_hand