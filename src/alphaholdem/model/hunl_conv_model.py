import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class HUNLConvModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.card_net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(16, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
            nn.Linear(3328, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.action_net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(16, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
            nn.Linear(3840, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.policy_fn = nn.Sequential(
            nn.Linear(265, 64),
            nn.ReLU(),
            nn.Linear(64, 9),
        )
        self.value_fn = nn.Sequential(
            nn.Linear(265, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_dict, state, seq_lens):
        card_out = self.card_net(input_dict['obs']['observation'])
        action_out = self.action_net(input_dict['obs']['action_history'])
        out = torch.cat((card_out, action_out, input_dict['obs']['action_mask']), dim=1)
        policy = self.policy_fn(out)
        self._value_out = self.value_fn(out)
        # return policy, state
        inf_mask = torch.clamp(torch.log(input_dict['obs']['action_mask']), -1e10, 1e10)
        return policy + inf_mask, state

    def value_function(self):
        return self._value_out.flatten()
