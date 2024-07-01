from alphaholdem.poker.component.hand import Hand, HandType
from alphaholdem.poker.component.card import Card
from alphaholdem.poker.component.judger import Judger

class TestJudger():
    judger = Judger()
    # straight_flush_hand  = Hand(['5s', '6s', '7s', '8s', '9s'])
    # four_of_a_kind_hand  = Hand(['Qs', 'Qd', 'Qh', 'Qc', '6c'])
    # full_house_hand      = Hand(['5h', '5d', 'Ad', 'Ac', '5s'])
    # flush_hand           = Hand(['Th', '7h', '6h', '5h', '8h'])
    # straight_hand        = Hand(['6s', '5h', '2c', '3d', '4h'])
    # straight_hand_A      = Hand(['As', '5h', '2c', '3d', '4h'])
    # three_of_a_kind_hand = Hand(['Ks', '4h', 'Kh', 'Kc', 'Jd'])
    # two_pair_hand        = Hand(['Ks', '4h', 'Kh', '4s', 'Jd'])
    # one_pair_hand        = Hand(['Ks', '4h', '7h', '4s', 'Jd'])
    # high_card_hand       = Hand(['Ks', '2h', '7h', '4s', 'Jd'])

    def get_best_hand(self, cards: list[str]):
        cards = Card.from_str_list(cards)
        return self.judger.get_best_hand(cards[:2], cards[2:])

    def test_best_hand(self):
        assert str(self.get_best_hand(['5s', '5d', '5h', '6s', '7s', '8s', '9s'])) == 'HandType.StraightFlush [9s, 8s, 7s, 6s, 5s]'
        assert str(self.get_best_hand(['Qs', 'Qd', '4h', '4c', 'Qh', 'Qc', '6c'])) == 'HandType.FourOfAKind [Qs, Qh, Qd, Qc, 6c]'
        assert str(self.get_best_hand(['5h', '5d', 'Ad', 'Ac', '5s', 'Kd', 'Kc'])) == 'HandType.FullHouse [5s, 5h, 5d, Ad, Ac]'
        assert str(self.get_best_hand(['9d', 'As', 'Th', '7h', '6h', '5h', '8h'])) == 'HandType.Flush [Th, 8h, 7h, 6h, 5h]'
        assert str(self.get_best_hand(['6s', '5d', '5h', '2c', '5c', '3d', '4h'])) == 'HandType.Straight [6s, 5h, 4h, 3d, 2c]'
        assert str(self.get_best_hand(['7h', 'As', '5h', '7h', '2c', '3d', '4h'])) == 'HandType.Straight [5h, 4h, 3d, 2c, As]'
        assert str(self.get_best_hand(['3h', 'Ks', '2d', '4h', 'Kh', 'Kc', 'Jd'])) == 'HandType.ThreeOfAKind [Ks, Kh, Kc, Jd, 4h]'
        assert str(self.get_best_hand(['Ks', 'Th', '9h', '4h', 'Kh', '4s', 'Jd'])) == 'HandType.TwoPair [Ks, Kh, 4s, 4h, Jd]'
        assert str(self.get_best_hand(['Ks', '4h', '5d', '7h', '4s', 'Jd', '3c'])) == 'HandType.OnePair [4s, 4h, Ks, Jd, 7h]'
        assert str(self.get_best_hand(['Ks', 'Ad', '2h', '7h', '5s','4s', 'Jd'])) == 'HandType.HighCard [Ad, Ks, Jd, 7h, 5s]'

    def test_judge(self):
        assert self.judger.judge(
            pot=200,
            board_cards=Card.from_str_list(['5s', '6s', '7s', '8s', '9s']),
            num_players=2,
            player_bet=[100, 100],
            player_hole_cards=[Card.from_str_list(['2c', 'Kh']), Card.from_str_list(['6c', '3h'])],
            player_fold=[False, False],
        ) == [0, 0]
        assert self.judger.judge(
            pot=200,
            board_cards=Card.from_str_list(['5s', '6s', '7s', '8s', '9c']),
            num_players=2,
            player_bet=[100, 100],
            player_hole_cards=[Card.from_str_list(['As', 'Kh']), Card.from_str_list(['Ks', 'Qh'])],
            player_fold=[False, False],
        ) == [100, -100]
        assert self.judger.judge(
            pot=200,
            board_cards=Card.from_str_list(['5s', '6h', '7s', '8s', '9c']),
            num_players=2,
            player_bet=[100, 100],
            player_hole_cards=[Card.from_str_list(['As', 'Kh']), Card.from_str_list(['Ks', 'Th'])],
            player_fold=[False, False],
        ) == [-100, 100]
        assert self.judger.judge(
            pot=200,
            board_cards=Card.from_str_list(['5s', '6h', '7s', '8s', '9c']),
            num_players=2,
            player_bet=[100, 100],
            player_hole_cards=[Card.from_str_list(['As', 'Ks']), Card.from_str_list(['6c', '6d'])],
            player_fold=[False, False],
        ) == [100, -100]
        assert self.judger.judge(
            pot=200,
            board_cards=Card.from_str_list(['5s', '6h', '7s', '9s', '9c']),
            num_players=2,
            player_bet=[100, 100],
            player_hole_cards=[Card.from_str_list(['As', 'Ks']), Card.from_str_list(['6c', '6d'])],
            player_fold=[False, False],
        ) == [-100, 100]