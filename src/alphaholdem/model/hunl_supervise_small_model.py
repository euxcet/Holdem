import torch
from torch import nn

class HUNLSuperviseSmallModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.card_net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

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

            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

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
        self.fn = nn.Sequential(
            nn.Linear(448, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, cards: torch.Tensor, action_history: torch.Tensor):
        # bs * 4 * 4 * 13 bs * 4 * 12 * 5
        # card_out = self.card_net(cards)
        # action_out = self.action_net(action_history)
        out = torch.cat((cards.flatten(start_dim=1), action_history.flatten(start_dim=1)), dim=1)
        out = self.fn(out)
        return out

