import pytest
from alphaholdem.poker.component.hand import Hand, HandType
from alphaholdem.poker.component.card import Card

class TestSupervise():
    SKIP = True

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_supervise(self):
        self.hole_cards_mapping: list[tuple[Card, Card]] = []
        for i in range(52):
            for j in range(i):
                self.hole_cards_mapping.append((Card(rank_first_id=i), Card(rank_first_id=j)))
        print()
        # with open('/home/clouduser/zcc/Agent/1724729416219.txt', 'r') as f:
        with open('/home/clouduser/zcc/Holdem/strategy/bug.txt', 'r') as f:
            line = f.readline()
            while True:
                print(line)
                strategy = []
                while True:
                    line = f.readline()
                    if 'MATCH' in line[:10]:
                        break
                    strategy.append(list(map(float, line.strip().split(' '))))
                table = [[ [0 for k in range(len(strategy))] for i in range(13) ]  for j in range(13)]
                for a_id in range(len(strategy)):
                    s = strategy[a_id]
                    for i in range(len(s)):
                        c0, c1 = self.hole_cards_mapping[i]
                        if c0.rank == c1.rank:
                            table[12 - c0.rank][12 - c1.rank][a_id] += s[i] / 6.0
                        elif c0.suit == c1.suit:
                            table[12 - c0.rank][12 - c1.rank][a_id] += s[i] / 4.0
                        else:
                            table[12 - c1.rank][12 - c0.rank][a_id] += s[i] / 12.0
                card_s = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
                for i in range(13):
                    for j in range(13):
                        if i == j:
                            hand = card_s[i] + card_s[j] + 'p'
                        elif i < j:
                            hand = card_s[i] + card_s[j] + 's'
                        else:
                            hand = card_s[j] + card_s[i] + 'o'
                        print(hand  + "|" + ",".join(map(lambda x: '%.2f'%x, table[i][j])), end = ' ')
                    print()
                _ = input()



# BUG!
# [[Td, 3d], [Jh, 2c]]
# player=BB pot=181.0BB chips=181.0BB current_player=BB
# Hole cards: [Jh, 2c]   Board cards: [Qh, 3h, As, 3s, Qs]
# Action Log:
# [Action] player=SB street=Street.Preflop action=ActionType.Call
#          raise_chip=0.0BB        raise_pot=0%    raise_to=0.0BB
# [Action] player=BB street=Street.Preflop action=ActionType.Raise
#          raise_chip=2.0BB        raise_pot=100%          raise_to=3.0BB
# [Action] player=SB street=Street.Preflop action=ActionType.Call
#          raise_chip=0.0BB        raise_pot=0%    raise_to=0.0BB
# [Action] player=BB street=Street.Flop action=ActionType.Raise
#          raise_chip=6.0BB        raise_pot=100%          raise_to=6.0BB
# [Action] player=SB street=Street.Flop action=ActionType.Call
#          raise_chip=0.0BB        raise_pot=0%    raise_to=0.0BB
# [Action] player=BB street=Street.Turn action=ActionType.Raise
#          raise_chip=18.0BB       raise_pot=100%          raise_to=18.0BB
# [Action] player=SB street=Street.Turn action=ActionType.Raise
#          raise_chip=72.0BB       raise_pot=100%          raise_to=72.0BB
# [Action] player=BB street=Street.Turn action=ActionType.Call
#          raise_chip=0.0BB        raise_pot=0%    raise_to=0.0BB
# [Action] player=BB street=Street.River action=ActionType.All_in
#          raise_chip=0.0BB        raise_pot=0%    raise_to=0.0BB
# [Action] player=SB street=Street.River action=ActionType.Fold
#          raise_chip=0.0BB        raise_pot=0%    raise_to=0.0BB
# payoff = [-81.0, 81.0]
#  [-0.81, 0.81]