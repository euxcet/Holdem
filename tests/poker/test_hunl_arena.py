import os
import pytest
import torch
from alphaholdem.arena.hunl_arena import HunlArena
from alphaholdem.arena.policy.hunl.ppo_hunl_policy import PPOHunlPolicy
from alphaholdem.arena.policy.hunl.tf_texas_policy import TFTexasPolicy
from alphaholdem.arena.policy.random_policy import RandomPolicy

class TestHunlArena():
    SKIP = False

    def extract_pt(self, folder: str) -> str:
        for ckpt in os.listdir(folder):
            if ckpt.endswith('ckpt'):
                checkpoint = torch.load(os.path.join(folder, ckpt))
                state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    new_key = key.replace('model.', '')
                    state_dict[new_key] = value
                save_path = "/home/clouduser/zcc/Holdem/checkpoint/showdown/range_model.pt"
                # save_path = os.path.join(folder, ckpt[:-4] + 'pt')
                torch.save(state_dict, save_path)
                return save_path

    @pytest.mark.skipif(SKIP, reason="SKIP == True")
    def test_cfr(self):
        # ppos = PPOHunlPolicy.load_policies_from_run('/home/clouduser/ray_results/PPO_2024-09-01_12-07-04')
        # tf = TFTexasPolicy('/home/clouduser/zcc/checkpoint/38000_model/model.ckpt')
        # arena = HunlArena('./checkpoint/supervise/small_v1/supervise.pt')
        random = RandomPolicy()

        root = '/home/clouduser/zcc/Holdem/supervise/wandb/supervise/'
        folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
        newest_folder = max(folders, key=lambda f: os.path.getctime(os.path.join(root, f)))
        print(newest_folder)
        pt = self.extract_pt(os.path.join(root, newest_folder, 'checkpoints'))

        # arena = HunlArena(pt)
        # mean, var = arena.policy_vs_policy(
        #     policy0=arena.nash,
        #     policy1=random,
        #     runs=1024,
        # )
        # print('Result:', mean)
        # 6.298828124999994

        # preflop: efxeejpq