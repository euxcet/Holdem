import numpy as np
import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class RangeLeducModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.policy_fn = nn.Sequential(
            nn.Linear(47, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, num_outputs),
        )
        self.value_fn = nn.Sequential(
            nn.Linear(47, 256),
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
        # out = torch.cat((ranges, action_history, action_mask, board_card), dim=1)
        out = torch.cat((action_history, action_mask, board_card), dim=1)
        # print('INPUT', out)
        policy = self.policy_fn(out)
        self._value_out = self.value_fn(out)
        # print('policy', policy)

        policy[:, :12] = torch.sigmoid(policy[:, :12])

        # inf_mask = torch.clamp(torch.log(input_dict['obs']['action_mask']), -1e10, 1e10)
        # print('POLICY', policy)
        # policy[:, :4] = torch.softmax(policy[:, :4] + inf_mask, dim=1)
        # policy[:, 4:8] = torch.softmax(policy[:, 4:8] + inf_mask, dim=1)
        # policy[:, 8:12] = torch.softmax(policy[:, 8:12] + inf_mask, dim=1)
        # print(policy)

        # print('policy', policy)


        # inf_mask = torch.clamp(torch.log(input_dict['obs']['action_mask']), -1e10, 1e10)
        return policy, state

    def value_function(self):
        return self._value_out.flatten()

# J / Q / K

# check / raise
# fold / call / raise
