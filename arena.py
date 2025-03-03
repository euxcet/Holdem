import os

result = dict()
count = dict()

lines = []
root = 'arena_log/hunl'
for fname in os.listdir(root):
    if fname.endswith('txt'):
        f = open(os.path.join(root, fname), 'r')
        lines.extend(f.readlines())


players = []

for line in lines:
    l = line.strip().split(' ')
    p0 = int(l[0])
    p1 = int(l[1])
    if p0 not in players:
        players.append(p0)
    if p1 not in players:
        players.append(p1)

players.sort()
players = {players[i]: i for i in range(len(players))}

v_sum = [[0 for i in range(len(players))] for j in range(len(players))]
v_cnt = [[0 for i in range(len(players))] for j in range(len(players))]

for line in lines:
    l = line.strip().split(' ')
    p0 = int(l[0])
    p1 = int(l[1])
    runs = int(l[2])
    mean = float(l[3])
    v_sum[players[p0]][players[p1]] += runs * mean
    v_cnt[players[p0]][players[p1]] += runs

print(list(players.keys()))
for i in range(len(players)):
    for j in range(len(players)):
        if v_cnt[i][j] == 0:
            print(0, end=' ')
        else:
            print(v_cnt[i][j], end=' ')
    print()
print()
for i in range(len(players)):
    for j in range(len(players)):
        if v_cnt[i][j] == 0:
            print(0, end=' ')
        else:
            print(v_sum[i][j] / v_cnt[i][j], end=' ')
    print()