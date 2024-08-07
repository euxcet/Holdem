from .card import Card
from .hand import Hand, HandType
from .judger import Judger

class RangeKuhnJudger(Judger):
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
        factor: float,
    ) -> list[int]:
        payoff: list[int] = [0, 0]

        # sum_p0 = sum(player_range[0])
        # sum_p1 = sum(player_range[1])
        # if sum_p0 < 1e-5 or sum_p1 < 1e-5:
        #     return payoff

        # print('player range', player_range)

        # tot = 0
        for player0_card in Card.from_str_list(['Js', 'Qs', 'Ks']):
            for player1_card in Card.from_str_list(['Js', 'Qs', 'Ks']):
                if player0_card == player1_card:
                    continue
                player_best_hand =[
                    Hand([player0_card]),
                    Hand([player1_card]),
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

                prob = player_range[0][player0_card.rank - 9] * player_range[1][player1_card.rank - 9] / 6
                # tot += prob

                # print(player0_card, player1_card, prob, this_payoff[0], this_payoff[1], factor)

                payoff[0] += this_payoff[0] * prob * factor
                payoff[1] += this_payoff[1] * prob * factor
        return payoff
        print([payoff[0] / tot, payoff[1] / tot])
        return [payoff[0] / tot, payoff[1] / tot]
