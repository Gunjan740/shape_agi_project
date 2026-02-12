import torch
import torch.nn as nn

class PolicyHead(nn.Module):
    """
    Takes CNN features and outputs action logits.
    Output: (B, 5)
    """

    def __init__(self, input_dim, num_actions=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.net(x)
