import os
import pytest
import torch
import time
from alphaholdem.arena.hunl_arena import HunlArena
from alphaholdem.arena.policy.ppo_poker_policy import PPOPokerPolicy
from alphaholdem.arena.policy.hunl.ppo_hunl_policy import PPOHunlPolicy
from multiprocessing import Process

class TestHunlPPO():

    def pvp(self, i):
        root = './checkpoint/test/'
        cids = [0, 10, 50]
        for checkpoint in os.listdir(root):
            if checkpoint.endswith('.pt'):
                try:
                    cid = int(checkpoint[:-3].split('_')[-1])
                    if cid % 100 == 0 and cid not in cids:
                        cids.append(cid)
                except:
                    pass
        cids = cids[-10:]
        cids.sort()
        policies = []
        runs = 1024
        for cid in cids:
            policies.append(PPOPokerPolicy('./checkpoint/test/model_' + str(cid) + '.pt'))
        f = open(os.path.join('arena_log/hunl/', str(i) + '_' + str(time.time()) + '.txt'), 'w')
        arena = HunlArena()
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                mean, var = arena.policy_vs_policy(
                    policy0=policies[i],
                    policy1=policies[j],
                    runs=runs,
                )
                f.write(f"{cids[i]} {cids[j]} {runs} {mean}\n")
                f.flush()
                print(cids[i], cids[j], mean)

    def test_hunl_ppo(self):
        while True:
            ps: list[Process] = []
            for i in range(40):
                ps.append(Process(target=self.pvp, args=(i,)))
            for p in ps:
                p.start()
            for p in ps:
                p.join()