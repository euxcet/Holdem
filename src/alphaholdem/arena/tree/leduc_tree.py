from __future__ import annotations

class LeducNode():
    def __init__(
        self,
        player: int,
        son: list[LeducNode],
        action_history: str,
        action_prob: list[float],
        is_terminal: bool,
        hole_cards: str,
        board_card: str,
        payoff: list[float] = [0, 0],
    ) -> None:
        self.player = player
        self.son = son
        self.action_history = action_history
        self.action_prob = action_prob
        self.is_terminal = is_terminal
        self.payoff = payoff
        self.hole_cards = hole_cards
        self.board_card = board_card
    
    def dfs_ev(self, reach_prob: float) -> float:
        if self.is_terminal:
            # print('Terminal', self.action_history, self.hole_cards, self.board_card, self.payoff, reach_prob)
            return reach_prob * self.payoff[0]
        payoff = 0.0
        for i, child in enumerate(self.son):
            payoff += child.dfs_ev(reach_prob * self.action_prob[i]) 
        return payoff

class LeducTree():
    def __init__(self, strategy: list[dict]):
        self.t = set()
        self.root = self.create_root(strategy)

    def dfs_ev(self) -> float:
        return self.root.dfs_ev(reach_prob=1.0)

    def create_root(self, strategy: list[dict]) -> LeducNode:
        combos = [
            # 1/15
            'JQK', 'JKQ', 'QJK', 'QKJ', 'KJQ', 'KQJ',
            # 1 / 30
            'JJQ', 'JJK', 'JQJ', 'JQQ', 'JKJ', 'JKK',
            'QJJ', 'QJQ', 'QQJ', 'QQK', 'QKQ', 'QKK',
            'KJJ', 'KJK', 'KQQ', 'KQK', 'KKJ', 'KKQ',
        ]
        son = [
            self.create_node(
                strategy=strategy,
                player=0,
                hole_cards=combo[:2],
                board_card=combo[-1],
                action_history='/',
                street=0,
                num_street_raise=0,
                pot=[1, 1],
            )
            for combo in combos
        ]
        return LeducNode(
            player=2,
            son=son,
            action_history="",
            action_prob=[1 / 15] * 6 + [1 / 30] * 18,
            hole_cards="",
            board_card="",
            is_terminal=False,
            payoff=[],
        )

    #  1 win  0 tie  -1 lose
    def compare(self, p0: str, p1: str) -> int:
        score = {
            'KK': 0, 'QQ': 1, 'JJ': 2,
            'KQ': 3, 'QK': 3,
            'KJ': 4, 'JK': 4,
            'QJ': 5, 'JQ': 5,
        }
        if score[p0] < score[p1]:
            return 1
        elif score[p0] == score[p1]:
            return 0
        else:
            return -1

    def create_node(
        self,
        strategy: list[dict],
        player: int,
        hole_cards: str,
        board_card: str,
        action_history: str,
        street: int,
        num_street_raise: int,
        pot: list[int],
    ) -> LeducNode:
        if action_history[-1] == 'f':
            if player == 0:
                payoff = [pot[1], -pot[1]]
            else:
                payoff = [-pot[0], pot[0]]
            return LeducNode(
                player=2,
                son=[],
                action_history=action_history,
                action_prob=[],
                is_terminal=True,
                hole_cards=hole_cards,
                board_card=board_card,
                payoff=payoff,
            )

        if street == 2:
            c = self.compare(hole_cards[0] + board_card, hole_cards[1] + board_card)
            return LeducNode(
                player=2,
                son=[],
                action_history=action_history,
                action_prob=[],
                is_terminal=True,
                hole_cards=hole_cards,
                board_card=board_card,
                payoff=[c * pot[0], -c * pot[1]],
            )
        
        if street == 0:
            key = hole_cards[player] + ':' + action_history + ':'
        else:
            key = hole_cards[player] + board_card + ':' + action_history + ':'
        self.t.add(key)
        # raise call fold
        son = []
        action_prob = []
        # raise
        if num_street_raise < 2:
            pot[player] += 2 * (street + 1) * (num_street_raise + 1)
            son.append(self.create_node(
                strategy=strategy,
                player = 1 - player,
                hole_cards=hole_cards,
                board_card=board_card,
                action_history=action_history + 'r',
                street=street,
                num_street_raise=num_street_raise + 1,
                pot=pot,
            ))
            pot[player] -= 2 * (street + 1) * (num_street_raise + 1)
            action_prob.append(strategy[player][key][0])
        # check
        if action_history[-1] == '/':
            son.append(self.create_node(
                strategy=strategy,
                player = 1 - player,
                hole_cards=hole_cards,
                board_card=board_card,
                action_history=action_history + 'c',
                street=street,
                num_street_raise=num_street_raise,
                pot=pot,
            ))
            action_prob.append(strategy[player][key][1])
        # call or (check check)
        if action_history[-1] != '/':
            if action_history[-1] == 'r':
                pot[player] += 2 * (street + 1)
            son.append(self.create_node(
                strategy=strategy,
                player = 0,
                hole_cards=hole_cards,
                board_card=board_card,
                action_history=action_history + 'c/',
                street=street + 1,
                num_street_raise=0,
                pot=pot,
            ))
            if action_history[-1] == 'r':
                pot[player] -= 2 * (street + 1)
            action_prob.append(strategy[player][key][1])
        # fold
        if action_history[-1] == 'r':
            son.append(self.create_node(
                strategy=strategy,
                player = 1 - player,
                hole_cards=hole_cards,
                board_card=board_card,
                action_history=action_history + 'f',
                street=street,
                num_street_raise=num_street_raise,
                pot=pot,
            ))
            action_prob.append(strategy[player][key][2])

        # print(player, action_history, hole_cards, board_card, sum(action_prob))
        return LeducNode(
            player=player,
            son=son,
            action_history=action_history,
            action_prob=action_prob,
            is_terminal=False,
            hole_cards=hole_cards,
            board_card=board_card,
            payoff=[],
        )
