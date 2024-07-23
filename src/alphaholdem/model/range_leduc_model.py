import numpy as np
import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class RangeLeducModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.policy_fn = nn.Sequential(
            nn.Linear(53, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, num_outputs),
        )
        self.value_fn = nn.Sequential(
            nn.Linear(53, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, input_dict, state, seq_lens):
        # 2 * 3
        ranges = input_dict['obs']['ranges'].flatten(start_dim=1)
        # 2 * 4 * 5
        action_history = input_dict['obs']['action_history'].flatten(start_dim=1)
        # 4 * 1
        action_mask = input_dict['obs']['action_mask'].flatten(start_dim=1)
        # 3 * 1
        board_card = input_dict['obs']['board_card'].flatten(start_dim=1)
        out = torch.cat((ranges, action_history, action_mask, board_card), dim=1)
        policy = self.policy_fn(out)
        self._value_out = self.value_fn(out)
        policy = torch.clamp(policy, min=1e-3, max=1)
        policy[:, :12] = torch.softmax(policy[:, :12], dim=1)

        # test = torch.tensor(np.array([
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,]
        #     for i in range(policy.shape[0])
        # ]).astype(np.float32)).to('cuda')
        # print("shape", test.shape, policy.shape)
        return policy, state

    def value_function(self):
        return self._value_out.flatten()