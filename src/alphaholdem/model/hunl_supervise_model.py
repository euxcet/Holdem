import torch
from torch import nn

class HUNLSuperviseModel(nn.Module):
    def __init__(self):
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
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(),
        )

    def forward(self, cards, action_history):
        card_out = self.card_net(cards)
        action_out = self.action_net(action_history)
        out = torch.cat((card_out, action_out), dim=1)
        out = self.policy_fn(out)
        return out


# class HUNLSuperviseModel(nn.Module):
#     def __init__(self):
#         nn.Module.__init__(self)
#         self.card_net = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=2, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=1),
#             nn.Conv2d(16, 64, kernel_size=2, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=1),
#             nn.Flatten(),
#             nn.Linear(3328, 128),
#             nn.ReLU(),
#         )
#         self.action_net = nn.Sequential(
#             nn.Conv2d(4, 16, kernel_size=2, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=1),
#             nn.Conv2d(16, 64, kernel_size=2, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=1),
#             nn.Flatten(),
#             nn.Linear(3840, 128),
#             nn.ReLU(),
#         )
#         self.policy_fn = nn.Sequential(
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             # nn.Linear(128, 64),
#             # nn.ReLU(),
#             nn.Linear(64, 1326 * 4),
#         )

#     def forward(self, cards, action_history):
#         card_out = self.card_net(cards)
#         action_out = self.action_net(action_history)
#         out = torch.cat((card_out, action_out), dim=1)
#         out = self.policy_fn(out)
#         out = torch.sigmoid(out)
#         return out
