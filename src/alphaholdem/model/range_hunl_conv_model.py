import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

def create_range_hunl_conv_model(num_action: int):
    class RangeHUNLConvModel(TorchModelV2, nn.Module):
        def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
            TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
            nn.Module.__init__(self)
            self.num_combo = num_outputs // 2 // num_action
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

            # num_combo * num_action
            self.policy_fn = nn.Sequential(
                nn.Linear(256 + num_action, 64),
                nn.ReLU(),
                # nn.Linear(64, num_outputs // 2),
                nn.Linear(64, num_outputs),
            )
            self.value_fn = nn.Sequential(
                nn.Linear(256 + num_action, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, input_dict, state, seq_lens):
            card_out = self.card_net(input_dict['obs']['observation'])
            action_out = self.action_net(input_dict['obs']['action_history'])
            out = torch.cat((card_out, action_out, input_dict['obs']['action_mask']), dim=1)
            policy = self.policy_fn(out)
            self._value_out = self.value_fn(out)
            # batch_size = policy.shape[0]
            # policy = torch.softmax(policy.view((batch_size, 2, self.num_combo, num_action)), axis=3)
            # policy = policy.view((batch_size, self.num_combo * num_action * 2))
            return policy, state

        def value_function(self):
            return self._value_out.flatten()
    return RangeHUNLConvModel
