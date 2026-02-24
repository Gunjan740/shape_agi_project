# evaluation/evaluate.py

import argparse
import torch
import yaml
import json
import time
import numpy as np
import random
from pathlib import Path

from envs import ShapeEnv
from policies.policy import Policy


def parse_args():
    parser = argparse.ArgumentParser(description="Shape AGI Evaluation")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--delay_ms", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps_per_episode", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, required=True)

    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # -------------------------
    # Seed
    # -------------------------
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)
    print("Evaluation seed:", args.seed, flush=True)

    config = load_config(args.config)

    # -------------------------
    # Environment (REAL-TIME MODE)
    # -------------------------
    env_cfg = config.get("env", {})

    env = ShapeEnv(
        time_scaling=float(env_cfg.get("time_scaling", 1.0)),
        noisy=bool(env_cfg.get("noisy", False)),
        random_initial=True,
        device=str(device),
    )

    print("Environment ready.", flush=True)

    # -------------------------
    # Load trained model
    # -------------------------
    policy = Policy(delay_ms=args.delay_ms)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy = policy.to(device)
    policy.eval()

    print("Model loaded.", flush=True)

    success_count = 0
    compute_times = []

    possible_shapes = env.shapes_list

    print("\nStarting evaluation...\n", flush=True)

    for ep in range(args.episodes):

        obs = env.reset().to(device)
        target_shape = np.random.choice(possible_shapes)

        episode_success = False

        for step in range(args.steps_per_episode):

            # -------------------------
            # Real-time policy wrapper
            # -------------------------
            def policy_fn(state):
                """
                Accepts state in shapes:
                  (3, H, W)
                  (1, 3, H, W)
                  (T, 3, H, W)
                  (1, T, 3, H, W)

                Always converts to:
                  (1, 3, H, W)
                """
                state = state.to(device)

                # Case 1: (3, H, W)
                if state.dim() == 3:
                    state = state.unsqueeze(0)

                # Case 2: (T, 3, H, W) OR already (1, 3, H, W)
                elif state.dim() == 4:
                    if state.size(0) != 1:
                        # assume time stack -> take latest frame
                        state = state[-1].unsqueeze(0)

                # Case 3: (1, T, 3, H, W)
                elif state.dim() == 5:
                    state = state[:, -1, ...]

                else:
                    raise ValueError(f"Unexpected state shape: {state.shape}")

                start_time = time.time()
                action = policy.act(state)
                compute_times.append(time.time() - start_time)

                return action

            # REAL-TIME COUPLED STEP
            obs = env.step(policy_fn)
            obs = obs.unsqueeze(0).to(device)

            state = env._get_state()

            if state["shape"] == target_shape:
                episode_success = True

        if episode_success:
            success_count += 1

        print(f"Episode {ep+1}/{args.episodes} | success={episode_success}", flush=True)

    # -------------------------
    # Metrics
    # -------------------------
    success_rate = success_count / args.episodes
    avg_compute_time_ms = np.mean(compute_times) * 1000

    print("\nEvaluation finished.")
    print("Success rate:", success_rate)
    print("Avg compute time (ms):", avg_compute_time_ms)

    # -------------------------
    # Save results
    # -------------------------
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "success_rate": success_rate,
        "avg_compute_time_ms": avg_compute_time_ms,
        "delay_ms": args.delay_ms,
        "episodes": args.episodes,
        "seed": args.seed,
    }

    with open(output_path / "eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Metrics saved.")
    print("Done.")


if __name__ == "__main__":
    main()