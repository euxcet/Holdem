import numpy as np
from ..component.card import Card

'''
    Tensor: [cards(4 * 4 * 13), actions(4 * 12 * 5)]
    Actions: fold check call raise all_in
    Example: MATCHSTATE:1:5:r300c/:|Js7s/Ac5d9d
'''
def acpc_to_tensor(history: str) -> np.ndarray:
    states = history.split(':')
    player = int(states[1])
    board = states[-1].split('/')[1:]
    in_cards = np.zeros((3, 4, 13), dtype=np.float32)
    if len(board) >= 1:
        for card in Card.from_str_list([board[0][:2], board[0][2:4], board[0][4:6]]):
            in_cards[0][card.suit][card.rank] = 1.0
    if len(board) >= 2:
        card = Card.from_str(board[1])
        in_cards[1][card.suit][card.rank] = 1.0
    if len(board) >= 3:
        card = Card.from_str(board[2])
        in_cards[2][card.suit][card.rank] = 1.0
    action_history = states[3]
    current_player = 0
    street = 0
    max_num_actions_street = 6
    in_action_history = np.zeros((4, 12, 5), dtype=np.float32)
    is_all_in = False
    for street_actions in action_history.split('/'):
        num_action = [0, 0]
        if street == 0:
            is_check = False
        else:
            is_check = True
        action = 0
        for i in range(len(street_actions)):
            action = -1
            if street_actions[i] == 'c':
                if is_all_in:
                    action = 4 # all in
                elif is_check:
                    action = 1 # check
                else:
                    if street == 0:
                        is_check = True
                    action = 2 # call
            if street_actions[i] == 'r':
                is_check = False
                if street_actions[i + 1:].startswith('20000'):
                    action = 4 # all in
                    is_all_in = True
                else:
                    action = 3 # raise
            if action >= 0:
                in_action_history[street][current_player * max_num_actions_street + num_action[current_player]][action] = 1
                num_action[current_player] += 1
                current_player = 1 - current_player
        current_player = 1
        street += 1

    in_tensor = np.concatenate((in_cards.flatten(), in_action_history.flatten()))
    return in_tensor