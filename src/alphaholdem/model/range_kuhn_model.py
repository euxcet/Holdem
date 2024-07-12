import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class RangeKuhnModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.policy_fn = nn.Sequential(
            nn.Linear(11, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.value_fn = nn.Sequential(
            nn.Linear(11, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, input_dict, state, seq_lens):
        # 3
        card_out = input_dict['obs']['observation'].flatten(start_dim=1)
        # 2 * 4
        action_out = input_dict['obs']['action_history'].flatten(start_dim=1)
        out = torch.cat((card_out, action_out), dim=1)
        policy = self.policy_fn(out)
        self._value_out = self.value_fn(out)
        return policy, state

    def value_function(self):
        return self._value_out.flatten()
