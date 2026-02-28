import time
import torch
import torch.nn as nn

from models.cnn_encoder import CNNEncoder
from models.policy_head import PolicyHead


class Policy(nn.Module):
    """
    Image-based policy using CNN encoder + MLP head + value head (for PPO).
    Used for both training and evaluation.
    """

    def __init__(self, delay_ms: int = 0, goal_dim: int = 0):
        super().__init__()

        self.delay_ms = delay_ms
        self.goal_dim = int(goal_dim)

        self.encoder = CNNEncoder()

        dummy_input = torch.zeros(1, 3, 144, 144)
        dummy_features = self.encoder(dummy_input)
        feature_dim = dummy_features.shape[1]

        self.head = PolicyHead(input_dim=feature_dim + self.goal_dim)

        # Value head: shares encoder, separate MLP
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim + self.goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _encode(self, obs: torch.Tensor, goal: torch.Tensor = None) -> torch.Tensor:
        features = self.encoder(obs)
        if self.goal_dim > 0:
            if goal is None:
                raise ValueError("goal must be provided when goal_dim > 0")
            if goal.dim() != 2 or goal.size(0) != features.size(0) or goal.size(1) != self.goal_dim:
                raise ValueError(f"goal must have shape (B, {self.goal_dim}), got {tuple(goal.shape)}")
            return torch.cat([features, goal.to(obs.device)], dim=1)
        return features

    def forward(self, obs: torch.Tensor, goal: torch.Tensor = None):
        """Returns (logits, value) — used during PPO training."""
        x = self._encode(obs, goal)
        return self.head(x), self.value_head(x).squeeze(-1)

    def value(self, obs: torch.Tensor, goal: torch.Tensor = None) -> torch.Tensor:
        """Scalar value estimate — used for GAE bootstrapping."""
        with torch.no_grad():
            return self.value_head(self._encode(obs, goal)).squeeze(-1)

    def act(self, obs: torch.Tensor, goal: torch.Tensor = None) -> torch.Tensor:
        """
        obs → CNN → concat goal → head → argmax → one-hot action
        """
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        device = obs.device

        with torch.no_grad():
            logits = self.head(self._encode(obs, goal))
            action_index = torch.argmax(logits, dim=1)
            action = torch.zeros(5, device=device)
            action[action_index.item()] = 1.0

        return action