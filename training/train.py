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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, output_path / "config.yaml")

    set_seed(args.seed)
    print("Seed set.", flush=True)

    config = load_config(args.config)
    print("Loaded config:", config, flush=True)

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

    from policies.policy import Policy

    policy_cfg = config.get("policy", {})
    policy = Policy(delay_ms=int(policy_cfg.get("delay_ms", 0)))
    policy = policy.to(device)

    print("Policy initialized:",
          {"delay_ms": policy.delay_ms}, flush=True)

    print("\nStarting minimal training loop...", flush=True)

    learning_rate = config["training"]["learning_rate"]
    num_steps = config["training"]["num_steps"]

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    total_reward = 0.0
    reward_history = []

    obs = env.reset().to(device)

    # -------------------------
    # NEW: Random target shape per episode
    # -------------------------
    possible_shapes = env.shapes_list  # assumes your env exposes this
    target_shape = random.choice(possible_shapes)
    print("Target shape for this episode:", target_shape, flush=True)

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

        # -------------------------
        # NEW: reward based on matching target
        # -------------------------
        reward = 1.0 if state["shape"] == target_shape else 0.0

        total_reward += reward
        reward_history.append(total_reward)

        loss = -log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs

        if (step + 1) % 100 == 0 or step == 0:
            print(f"Step {step+1}/{num_steps} | total_reward={total_reward}", flush=True)

    print("Training finished.", flush=True)
    print("Total reward:", total_reward, flush=True)

    torch.save(policy.state_dict(), output_path / "model.pt")

    metrics = {
        "total_reward": total_reward,
        "num_steps": num_steps,
        "seed": args.seed,
        "target_shape": target_shape
    }

    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(output_path / "reward_curve.npy", np.array(reward_history))

    print("Model and metrics saved.", flush=True)
    print("Setup complete.", flush=True)


if __name__ == "__main__":
    main()