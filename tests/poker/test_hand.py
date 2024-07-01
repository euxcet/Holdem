from alphaholdem.poker.component.hand import Hand, HandType
from alphaholdem.poker.component.card import Card

class TestHand():
    straight_flush_hand  = Hand(['5s', '6s', '7s', '8s', '9s'])
    four_of_a_kind_hand  = Hand(['Qs', 'Qd', 'Qh', 'Qc', '6c'])
    full_house_hand      = Hand(['5h', '5d', 'Ad', 'Ac', '5s'])
    flush_hand           = Hand(['Th', '7h', '6h', '5h', '8h'])
    straight_hand        = Hand(['6s', '5h', '2c', '3d', '4h'])
    straight_hand_A      = Hand(['As', '5h', '2c', '3d', '4h'])
    three_of_a_kind_hand = Hand(['Ks', '4h', 'Kh', 'Kc', 'Jd'])
    two_pair_hand        = Hand(['Ks', '4h', 'Kh', '4s', 'Jd'])
    one_pair_hand        = Hand(['Ks', '4h', '7h', '4s', 'Jd'])
    high_card_hand       = Hand(['Ks', '2h', '7h', '4s', 'Jd'])

    def test_hand_type(self):
        # HighCard = 0
        # OnePair = 1
        # TwoPair = 2
        # ThreeOfAKind = 3
        # Straight = 4
        # Flush = 5
        # FullHouse = 6
        # FourOfAKind = 7
        # StraightFlush = 8
        assert str(self.straight_flush_hand) == 'HandType.StraightFlush [9s, 8s, 7s, 6s, 5s]'
        assert str(self.four_of_a_kind_hand) == 'HandType.FourOfAKind [Qs, Qh, Qd, Qc, 6c]'
        assert str(self.full_house_hand) == 'HandType.FullHouse [5s, 5h, 5d, Ad, Ac]'
        assert str(self.flush_hand) == 'HandType.Flush [Th, 8h, 7h, 6h, 5h]'
        assert str(self.straight_hand) == 'HandType.Straight [6s, 5h, 4h, 3d, 2c]'
        assert str(self.straight_hand_A) == 'HandType.Straight [5h, 4h, 3d, 2c, As]'
        assert str(self.three_of_a_kind_hand) == 'HandType.ThreeOfAKind [Ks, Kh, Kc, Jd, 4h]'
        assert str(self.two_pair_hand) == 'HandType.TwoPair [Ks, Kh, 4s, 4h, Jd]'
        assert str(self.one_pair_hand) == 'HandType.OnePair [4s, 4h, Ks, Jd, 7h]'
        assert str(self.high_card_hand) == 'HandType.HighCard [Ks, Jd, 7h, 4s, 2h]'

    def test_compare_hand_type(self):
        assert self.straight_flush_hand > self.four_of_a_kind_hand
        assert self.four_of_a_kind_hand > self.full_house_hand
        assert self.full_house_hand > self.flush_hand
        assert self.flush_hand > self.straight_hand
        assert self.straight_hand > self.straight_hand_A
        assert self.straight_hand_A > self.three_of_a_kind_hand
        assert self.three_of_a_kind_hand > self.two_pair_hand
        assert self.two_pair_hand > self.one_pair_hand
        assert self.one_pair_hand > self.high_card_hand

    def test_same_hand_type(self):
        assert self.straight_hand == Hand(['6h', '5h', '2c', '3d', '4h'])
        assert self.one_pair_hand == Hand(['Kh', '4h', '7h', '4d', 'Jh'])