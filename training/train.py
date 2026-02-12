"""
import time
from envs import ShapeEnv
from policies.policy import Policy

def run_step(env, policy, name):
    start = time.perf_counter()
    observations, info = env.step(policy.act)
    end = time.perf_counter()

    print(f"\n{name}")
    print(f"Policy time (ms): {info['actual_policy_time_ms']:.2f}")
    print(f"Env steps taken: {info['num_environment_steps']}")
    print(f"Total wall time (ms): {(end - start)*1000:.2f}")

def main():
    # Create ONE env and set time_scaling
    env = ShapeEnv(time_scaling=5.0)

    # Define policies
    system1 = Policy(delay_ms=0)
    system2 = Policy(delay_ms=20)

    # Benchmark ONLY System 1 (single baseline)
    env.benchmark_policy(system1.act)

    # Evaluate both against the SAME baseline
    env.reset()
    run_step(env, system1, "System 1 (baseline)")

    env.reset()
    run_step(env, system2, "System 2 (slower vs baseline)")

if __name__ == "__main__":
    main()
"""

"""
import torch
from envs import ShapeEnv
from models.cnn_encoder import CNNEncoder

def main():
    env = ShapeEnv()
    obs = env.reset()  # shape: (1, 3, H, W)

    print("Observation shape:", obs.shape)

    model = CNNEncoder()
    features = model(obs)

    print("Feature shape:", features.shape)

if __name__ == "__main__":
    main()
"""
"""
import torch
from envs import ShapeEnv
from models.cnn_encoder import CNNEncoder
from models.policy_head import PolicyHead

def main():
    env = ShapeEnv()
    obs = env.reset()

    encoder = CNNEncoder()
    features = encoder(obs)

    print("Feature shape:", features.shape)

    input_dim = features.shape[1]
    head = PolicyHead(input_dim=input_dim)

    logits = head(features)

    print("Logits shape:", logits.shape)

    action = torch.argmax(logits, dim=1)

    print("Chosen action index:", action.item())

if __name__ == "__main__":
    main()
"""

from envs import ShapeEnv
from policies.policy import Policy

def main():
    env = ShapeEnv()
    policy = Policy(delay_ms=0)

    obs = env.reset()
    action = policy.act(obs)

    print("Action:", action)
    print("Action sum (should be 1):", action.sum())

if __name__ == "__main__":
    main()
