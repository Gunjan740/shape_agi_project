import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Projects goal vector to per-channel (gamma, beta) and applies:
        output = gamma * x + beta
    This lets the goal modulate CNN feature maps at each layer.
    """

    def __init__(self, goal_dim, num_channels):
        super().__init__()
        self.proj = nn.Linear(goal_dim, num_channels * 2)

    def forward(self, x, goal):
        # x: (B, C, H, W)  goal: (B, goal_dim)
        params = self.proj(goal)                      # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)          # each (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)     # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class CNNEncoder(nn.Module):
    """
    Deeper CNN with Global Average Pooling + FiLM goal conditioning.
    Input: (B, 3, H, W), goal: (B, goal_dim)
    Output: (B, 64)

    Goal is injected at every conv layer via FiLM, making the encoder
    goal-aware from the start rather than only at the policy head.
    """

    def __init__(self, goal_dim=0):
        super().__init__()
        self.goal_dim = goal_dim

        # Conv layers kept separate so FiLM can be applied between them
        self.conv1 = nn.Sequential(nn.Conv2d(3,  16, kernel_size=3, stride=2), nn.ReLU())  # 144 → 71
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU())  # 71  → 35
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU())  # 35  → 17
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())  # 17 → 17

        if goal_dim > 0:
            self.film1 = FiLMLayer(goal_dim, 16)
            self.film2 = FiLMLayer(goal_dim, 32)
            self.film3 = FiLMLayer(goal_dim, 64)
            self.film4 = FiLMLayer(goal_dim, 64)

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x, goal=None):
        x = self.conv1(x)
        if self.goal_dim > 0 and goal is not None:
            x = self.film1(x, goal)

        x = self.conv2(x)
        if self.goal_dim > 0 and goal is not None:
            x = self.film2(x, goal)

        x = self.conv3(x)
        if self.goal_dim > 0 and goal is not None:
            x = self.film3(x, goal)

        x = self.conv4(x)
        if self.goal_dim > 0 and goal is not None:
            x = self.film4(x, goal)

        x = self.gap(x)
        x = self.flatten(x)
        return x
