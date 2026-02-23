import argparse
import shutil
import json
import random
import numpy as np
import torch
import yaml
from pathlib import Path
import os
import sys

print("[train.py] module loaded", flush=True)


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
    print("[train.py] ENTER main()", flush=True)
    print("Python:", sys.executable, flush=True)
    print("CWD:", os.getcwd(), flush=True)

    args = parse_args()

    print("Config path:", args.config, flush=True)
    print("Seed:", args.seed, flush=True)
    print("Output directory:", args.output_dir, flush=True)

    # -------------------------
    # Device setup (CUDA support)
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy config
    shutil.copy(args.config, output_path / "config.yaml")

    # Set seed
    set_seed(args.seed)
    print("Seed set.", flush=True)

    # Load config
    config = load_config(args.config)
    print("Loaded config:", config, flush=True)

    # -------------------------
    # Initialize ENV
    # -------------------------
    from envs import ShapeEnv

    env_cfg = config.get("env", {})
    env = ShapeEnv(
        time_scaling=float(env_cfg.get("time_scaling", 1.0)),
        noisy=bool(env_cfg.get("noisy", False)),
        random_initial=True,
        device=str(device)
    )

    print("Env initialized:",
          {"time_scaling": env.time_scaling, "noisy": env.noisy}, flush=True)

    # -------------------------
    # Initialize POLICY
    # -------------------------
    from policies.policy import Policy

    policy_cfg = config.get("policy", {})
    policy = Policy(delay_ms=int(policy_cfg.get("delay_ms", 0)))
    policy = policy.to(device)

    print("Policy initialized:",
          {"delay_ms": policy.delay_ms}, flush=True)

    # -------------------------
    # Minimal Training Loop
    # -------------------------
    print("\nStarting minimal training loop...", flush=True)

    learning_rate = config["training"]["learning_rate"]
    num_steps = config["training"]["num_steps"]

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    total_reward = 0.0
    reward_history = []  # ✅ ADDED

    obs = env.reset().to(device)

    for step in range(num_steps):
        features = policy.encoder(obs)
        logits = policy.head(features)

        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)

        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)

        action = torch.zeros(5, device=device)
        action[action_index.item()] = 1.0

        next_obs = env._step_simulation(action)
        next_obs = next_obs.unsqueeze(0).to(device)

        state = env._get_state()
        reward = 1.0 if state["shape"] == "triangle" else 0.0
        total_reward += reward
        reward_history.append(total_reward)  # ✅ ADDED

        loss = -log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs

        if (step + 1) % 100 == 0 or step == 0:
            print(f"Step {step+1}/{num_steps} | total_reward={total_reward}", flush=True)

    print("Training finished.", flush=True)
    print("Total reward:", total_reward, flush=True)

    # -------------------------
    # Save Model & Metrics
    # -------------------------

    # Save model
    torch.save(policy.state_dict(), output_path / "model.pt")

    # Save metrics
    metrics = {
        "total_reward": total_reward,
        "num_steps": num_steps,
        "seed": args.seed
    }

    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save reward curve
    np.save(output_path / "reward_curve.npy", np.array(reward_history))

    print("Model and metrics saved.", flush=True)
    print("Setup complete.", flush=True)


if __name__ == "__main__":
    main()