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
    # Episodic Training Loop
    # -------------------------
    print("\nStarting episodic training loop...", flush=True)

    learning_rate = config["training"]["learning_rate"]
    num_steps = config["training"]["num_steps"]

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    steps_per_episode = 10
    num_episodes = num_steps // steps_per_episode

    total_reward = 0.0
    reward_history = []

    possible_shapes = env.shapes_list
    possible_colors = env.colors_list
    possible_sizes = env.sizes_list

    for episode in range(num_episodes):

        obs = env.reset().to(device)

        # FULL LATENT TARGET
        target_shape = random.choice(possible_shapes)
        target_color = random.choice(possible_colors)
        target_size  = random.choice(possible_sizes)

        episode_reward = 0.0

        for step in range(steps_per_episode):

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

            # FULL LATENT REWARD (Minimal Reward Shaping)
            reward = 0.0

            if state["shape"] == target_shape:
                reward += 0.33
            if state["color"] == target_color:
                reward += 0.33
            if state["size"] == target_size:
                reward += 0.34

            

            episode_reward += reward
            total_reward += reward
            reward_history.append(total_reward)

            loss = -log_prob * reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs

        print(f"Episode {episode+1} | episode_reward={episode_reward} | total_reward={total_reward}")

    print("Training finished.", flush=True)
    print("Total reward:", total_reward, flush=True)

    # -------------------------
    # Save Model & Metrics
    # -------------------------
    torch.save(policy.state_dict(), output_path / "model.pt")

    metrics = {
        "total_reward": total_reward,
        "num_steps": num_steps,
        "seed": args.seed
    }

    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(output_path / "reward_curve.npy", np.array(reward_history))

    print("Model and metrics saved.", flush=True)
    print("Setup complete.", flush=True)


if __name__ == "__main__":
    main()