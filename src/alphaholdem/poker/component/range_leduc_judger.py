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
        player_bet: list[int],
        player_fold: list[bool],
        player_range: list[list[float]],
    ) -> list[int]:
        payoff: list[int] = [0, 0]
        # print('Judge')
        # print(pot, board_cards, player_bet, player_range)

        # JQ 1/10 # JK 1/10 # QJ 1/10 # QQ 1/10
        # QK 1/5 # KJ 1/10 # KQ 1/5 # KK 1/10
        # optimize
        # for player0_card in Card.from_str_list(['Jc', 'Qc', 'Kc']):
        #     for player1_card in Card.from_str_list(['Jc', 'Qc', 'Kc']):
        #         if len(board_cards) > 0:
        #             if player0_card == player1_card and player0_card == board_cards[0]:
        #                 continue
        #             elif player0_card == player1_card or player0_card == board_cards[0] or player1_card == board_cards[0]:
        #                 prob = 1 / 10
        #             else:
        #                 prob = 1 / 5
        #         else:
        #             if player0_card == player1_card:
        #                 prob = 1 / 15
        #             else:
        #                 prob = 2 / 15

        #         player_best_hand =[
        #             self.get_best_hand([player0_card], board_cards),
        #             self.get_best_hand([player1_card], board_cards),
        #         ]
        #         if player_fold[0]: # player0 fold
        #             this_payoff = [-player_bet[0], player_bet[0]]
        #         elif player_fold[1]: # player1 fold
        #             this_payoff = [player_bet[1], -player_bet[1]]
        #         elif player_best_hand[0] > player_best_hand[1]: # player0 win
        #             this_payoff = [pot // 2, -pot // 2]
        #         elif player_best_hand[0] < player_best_hand[1]: # player1 win
        #             this_payoff = [-pot // 2, pot // 2]
        #         else: # chop
        #             this_payoff = [0, 0]
        #         range_prob = player_range[0][player0_card.rank - 9] * player_range[1][player1_card.rank - 9]
        #         payoff[0] += this_payoff[0] * prob * range_prob
        #         payoff[1] += this_payoff[1] * prob * range_prob

        sum_p0 = sum(player_range[0])
        sum_p1 = sum(player_range[1])
        if sum_p0 < 1e-5 or sum_p1 < 1e-5:
            return payoff

        for player0_card in Card.from_str_list(['Jc', 'Qc', 'Kc']):
            for player1_card in Card.from_str_list(['Jc', 'Qc', 'Kc']):
                player_best_hand =[
                    self.get_best_hand([player0_card], board_cards),
                    self.get_best_hand([player1_card], board_cards),
                ]
                if player_fold[0]: # player0 fold
                    this_payoff = [-player_bet[0], player_bet[0]]
                elif player_fold[1]: # player1 fold
                    this_payoff = [player_bet[1], -player_bet[1]]
                elif player_best_hand[0] > player_best_hand[1]: # player0 win
                    this_payoff = [pot // 2, -pot // 2]
                elif player_best_hand[0] < player_best_hand[1]: # player1 win
                    this_payoff = [-pot // 2, pot // 2]
                else: # chop
                    this_payoff = [0, 0]

                sum_p1 = sum(player_range[1]) - player_range[1][player0_card.rank - 9] / 2
                prob = player_range[0][player0_card.rank - 9] / sum_p0
                if player0_card == player1_card:
                    prob *= player_range[1][player1_card.rank - 9] / 2 / sum_p1
                else:
                    prob *= player_range[1][player1_card.rank - 9] / sum_p1

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