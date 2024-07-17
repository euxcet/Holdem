from .card import Card
from .hand import Hand, HandType

class Judger():

    HAND_CARD_SIZE = 5

    """
    hand_order: If the game is short deck poker, it should be HandType.short_deck_order()
    allow_num_hole_cards: How many hole cards can be included in the hand.
                          If the game is Omaha Hold'em, it should be [2].
    """
    def __init__(
        self,
        hand_order: list[HandType] = HandType.order(),
        allow_num_hole_cards: list[int] = [0, 1, 2],
    ) -> None:
        self.hand_order = hand_order
        self.allow_num_hole_cards = allow_num_hole_cards

    def judge(
        self,
        pot: int,
        board_cards: list[Card],
        num_players: int,
        player_bet: list[int],
        player_hole_cards: list[list[Card]],
        player_fold: list[bool],
    ) -> list[int]:
        payoff: list[int] = [-player_bet[i] for i in range(num_players)]
        if player_fold.count(False) == 1:
            winner = player_fold.index(False)
            payoff[winner] += pot
            return payoff
        player_best_hand: list[Hand] = [None] * num_players
        pot_player: list[tuple[int, Hand, int]] = []
        for player in range(num_players):
            if not player_fold[player]:
                player_best_hand[player] = self.get_best_hand(player_hole_cards[player], board_cards)
            else:
                player_best_hand[player] = Hand.worst_hand()
            pot_player.append((player_bet[player], player_best_hand[player], player))
        pot_player.sort()
        last_bet = 0
        for i in range(len(pot_player)):
            bet, hand, player = pot_player[i]
            if bet > last_bet:
                win_hand = hand
                win_players = [pot_player[i][2]]
                for j in range(i + 1, len(pot_player)):
                    if pot_player[j][1] > win_hand:
                        win_hand = hand
                        win_players = [pot_player[j][2]]
                    elif pot_player[j][1] == win_hand:
                        win_players.append(pot_player[j][2])
                side_pot = (len(pot_player) - i) * (bet - last_bet)
                for win_player in win_players:
                    payoff[win_player] += side_pot // len(win_players)
                payoff[win_players[0]] += side_pot % len(win_players)
                last_bet = bet
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