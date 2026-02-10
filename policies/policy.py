import time
import torch

class Policy:
    """
    Minimal policy to test real-time coupling.
    This policy has NO learning and NO perception.
    """

    def __init__(self, delay_ms: int = 0):
        """
        Args:
            delay_ms: artificial computation delay (System 2 > System 1)
        """
        self.delay_ms = delay_ms

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Takes an observation and returns a valid action.
        """
        # Artificial delay to simulate computation time
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        # Always return NO-OP action (index 4)
        action = torch.zeros(5)
        action[4] = 1.0
        return action
