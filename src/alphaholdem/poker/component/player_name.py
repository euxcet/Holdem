def get_players_name(num_players: int) -> list[str]:
    names = [''] * num_players
    names[0] = 'SB'
    names[1] = 'BB'
    if num_players > 2:
        names[-1] = 'BTN'
    if num_players > 3:
        names[-2] = 'CO'
    if num_players > 4:
        names[-3] = 'HJ'
    if num_players > 5:
        names[-4] = 'LJ'
    if num_players > 3:
        names[2] = 'UTG'
    if num_players > 7:
        for i in range(num_players - 7):
            names[i + 3] = 'UTG+' + str(i + 1)
    return names
