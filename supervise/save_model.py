import os
import torch
import argparse
from alphaholdem.arena.policy.random_policy import RandomPolicy

def extract_pt(folder: str, street: str) -> str:
    for ckpt in os.listdir(folder):
        if ckpt.endswith('ckpt'):
            checkpoint = torch.load(os.path.join(folder, ckpt))
            state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                new_key = key.replace('model.', '')
                state_dict[new_key] = value
            save_path = "/home/clouduser/zcc/Holdem/checkpoint/showdown/range_model.pt"
            torch.save(state_dict, save_path)
            save_path = "/home/clouduser/zcc/Holdem/checkpoint/showdown/range_" + street + ".pt"
            torch.save(state_dict, save_path)
            return save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--street', type=str)
    args = parser.parse_args()

    root = '/home/clouduser/zcc/Holdem/supervise/wandb/supervise/'
    folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    newest_folder = max(folders, key=lambda f: os.path.getctime(os.path.join(root, f)))
    extract_pt(os.path.join(root, newest_folder, 'checkpoints'), args.street)