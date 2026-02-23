import time
import torch
import torch.nn as nn

from models.cnn_encoder import CNNEncoder
from models.policy_head import PolicyHead


class Policy(nn.Module):
    """
    Image-based policy using CNN encoder + MLP head.
    Used for both training and evaluation.
    """

    def __init__(self, delay_ms: int = 0):
        super().__init__()

        self.delay_ms = delay_ms

        # Build perception model
        self.encoder = CNNEncoder()

        # Dynamically determine feature dimension
        dummy_input = torch.zeros(1, 3, 144, 144)
        dummy_features = self.encoder(dummy_input)
        feature_dim = dummy_features.shape[1]

        self.head = PolicyHead(input_dim=feature_dim)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        obs → CNN → head → argmax → one-hot action
        """

        # Artificial delay (System 2 slower)
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        device = obs.device

        with torch.no_grad():
            features = self.encoder(obs)
            logits = self.head(features)

            action_index = torch.argmax(logits, dim=1)

            # One-hot action (size 5)
            action = torch.zeros(5, device=device)
            action[action_index.item()] = 1.0

        return action