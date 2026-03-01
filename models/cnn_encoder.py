import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """
    Deeper CNN with Global Average Pooling.
    Input: (B, 3, H, W)
    Output: (B, 64)

    4 conv layers → AdaptiveAvgPool2d(1) → 64 features.
    GAP replaces Flatten to avoid drowning out the goal vector
    (64 features vs 18,496 from Flatten on 144×144 input).
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),   # 144 → 71
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # 71 → 35
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # 35 → 17
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 17 → 17 (refine)
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)
        x = self.flatten(x)
        return x
