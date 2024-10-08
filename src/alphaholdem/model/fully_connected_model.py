import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class FullyConnectedModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.card_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(208, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.action_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(64, 2)
        self.value_fn = nn.Linear(64, 1)

    def forward(self, input_dict, state, seq_lens):
        card_out = self.card_net(input_dict["obs"]['observation'])
        action_out = self.action_net(input_dict["obs"]['action_history'])
        out = torch.cat((card_out, action_out), dim=1)
        policy = self.policy_fn(out)
        self._value_out = self.value_fn(out)
        return policy, state

    def value_function(self):
        return self._value_out.flatten()