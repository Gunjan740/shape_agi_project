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
'''
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
'''


import argparse
import shutil
import json
import random
import numpy as np
import torch
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Shape AGI Training")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    print("Config path:", args.config)
    print("Seed:", args.seed)
    print("Output directory:", args.output_dir)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy config
    shutil.copy(args.config, output_path / "config.yaml")

    # Set seed
    set_seed(args.seed)
    print("Seed set.")

    # Load config
    config = load_config(args.config)
    print("Loaded config:", config)

    # -------------------------
    # Initialize ENV
    # -------------------------
    from envs import ShapeEnv

    env_cfg = config.get("env", {})
    env = ShapeEnv(
        time_scaling=float(env_cfg.get("time_scaling", 1.0)),
        noisy=bool(env_cfg.get("noisy", False)),
        random_initial=True   # ✅ ADDED
    )

    print("Env initialized:",
          {"time_scaling": env.time_scaling, "noisy": env.noisy})

    # -------------------------
    # Initialize POLICY
    # -------------------------
    from policies.policy import Policy

    policy_cfg = config.get("policy", {})
    policy = Policy(delay_ms=int(policy_cfg.get("delay_ms", 0)))

    print("Policy initialized:",
          {"delay_ms": policy.delay_ms})

    # -------------------------
    # Minimal Training Loop
    # -------------------------
    print("\nStarting minimal training loop...")

    learning_rate = config["training"]["learning_rate"]
    num_steps = config["training"]["num_steps"]

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    total_reward = 0.0
    obs = env.reset()

    for step in range(num_steps):

        # Forward pass
        features = policy.encoder(obs)
        logits = policy.head(features)

        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)

        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)

        # One-hot action
        action = torch.zeros(5)
        action[action_index.item()] = 1.0

        # Environment step (no real-time mode)
        next_obs = env._step_simulation(action)

        # Reward: +1 if shape == triangle  ✅ CHANGED
        state = env._get_state()
        reward = 1.0 if state["shape"] == "triangle" else 0.0

        total_reward += reward

        # REINFORCE loss
        loss = -log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update observation
        obs = next_obs.unsqueeze(0)

    print("Training finished.")
    print("Total reward:", total_reward)
    print("Setup complete.")


if __name__ == "__main__":
    main()
