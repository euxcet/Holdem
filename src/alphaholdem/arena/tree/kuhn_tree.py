from __future__ import annotations

class KuhnNode():
    def __init__(
        self,
        player: int,
        son: list[KuhnNode],
        action_history: str,
        action_prob: list[float],
        hole_cards: str,
        is_terminal: bool,
        payoff: list[float] = [0, 0],
    ) -> None:
        self.player = player
        self.son = son
        self.action_history = action_history
        self.action_prob = action_prob
        self.is_terminal = is_terminal
        self.hole_cards = hole_cards
        self.payoff = payoff
    
    def dfs_ev(self, reach_prob: float) -> float:
        if self.is_terminal:
            # print(self.hole_cards, self.action_history, reach_prob, self.payoff)
            return reach_prob * self.payoff[0]
        payoff = 0.0
        for i, child in enumerate(self.son):
            payoff += child.dfs_ev(reach_prob * self.action_prob[i]) 
        return payoff

class KuhnTree():
    def __init__(self, strategy: list[dict]):
        self.mapping = {'J': 0, 'Q': 1, 'K': 2}
        self.root = self.create_root(strategy)

    def dfs_ev(self) -> float:
        return self.root.dfs_ev(reach_prob=1.0)

    def create_root(self, strategy: list[dict]) -> KuhnNode:
        hole_cards_combo = ['JQ', 'JK', 'QJ', 'QK', 'KJ', 'KQ']
        son = [
            self.create_node(
                strategy=strategy,
                player=0,
                hole_cards=h,
                action_history='',
            )
            for h in hole_cards_combo
        ]
        return KuhnNode(
            player=2,
            son=son,
            action_history="",
            action_prob=[1 / 6] * 6,
            is_terminal=False,
            hole_cards="",
            payoff=[],
        )

    def create_node(
        self,
        strategy: list[dict],
        player: int,
        hole_cards: str,
        action_history: str,
    ) -> KuhnNode:
        if action_history in ['cc', 'crf', 'crc', 'rf', 'rc']:
            showdown = 1 if self.mapping[hole_cards[0]] > self.mapping[hole_cards[1]] else -1
            payoff = {
                'cc': [showdown, -showdown],
                'crf': [-1.0, 1.0],
                'crc': [2 * showdown, -2 * showdown],
                'rf': [1.0, -1.0],
                'rc': [2 * showdown, -2 * showdown],
            }[action_history]
            return KuhnNode(
                player=2,
                son=[],
                action_history=action_history,
                action_prob=[],
                is_terminal=True,
                hole_cards=hole_cards,
                payoff=payoff,
            )

        key = hole_cards[player] + ':' + action_history
        action_prob = strategy[player][key]
        if action_history.endswith('r'):
            # f / c
            son=[
                self.create_node(strategy, player=1 - player, hole_cards=hole_cards, action_history=action_history + 'f'),
                self.create_node(strategy, player=1 - player, hole_cards=hole_cards, action_history=action_history + 'c'),
            ]
        else:
            # c / r
            son=[
                self.create_node(strategy, player=1 - player, hole_cards=hole_cards, action_history=action_history + 'c'),
                self.create_node(strategy, player=1 - player, hole_cards=hole_cards, action_history=action_history + 'r'),
            ]
        return KuhnNode(
            player=player,
            son=son,
            action_history=action_history,
            action_prob=action_prob,
            is_terminal=False,
            hole_cards=hole_cards,
            payoff=[],
        )
