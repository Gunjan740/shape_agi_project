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


def _to_policy_input(state, device):
    state = state.to(device)

    if state.dim() == 3:
        state = state.unsqueeze(0)
    elif state.dim() == 4:
        if state.size(0) != 1:
            state = state[-1].unsqueeze(0)
    elif state.dim() == 5:
        state = state[:, -1, ...]
    else:
        raise ValueError(f"Unexpected state shape: {state.shape}")

    return state


def make_goal_vec(target_shape, target_color, target_size, shape_to_i, color_to_i, size_to_i, goal_dim, device):
    g = torch.zeros(1, goal_dim, device=device)

    si = shape_to_i[target_shape]
    ci = color_to_i[target_color]
    zi = size_to_i[target_size]

    g[0, si] = 1.0
    g[0, len(shape_to_i) + ci] = 1.0
    g[0, len(shape_to_i) + len(color_to_i) + zi] = 1.0
    return g


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)
    print("Evaluation seed:", args.seed, flush=True)

    config = load_config(args.config)

    env_cfg = config.get("env", {})

    env = ShapeEnv(
        time_scaling=float(env_cfg.get("time_scaling", 1.0)),
        noisy=bool(env_cfg.get("noisy", False)),
        random_initial=True,
        device=str(device),
    )

    print("Environment ready.", flush=True)

    # goal dims from env lists
    possible_shapes = env.shapes_list
    possible_colors = env.colors_list
    possible_sizes  = env.sizes_list

    shape_to_i = {s: i for i, s in enumerate(possible_shapes)}
    color_to_i = {c: i for i, c in enumerate(possible_colors)}
    size_to_i  = {z: i for i, z in enumerate(possible_sizes)}

    goal_dim = len(possible_shapes) + len(possible_colors) + len(possible_sizes)

    policy = Policy(delay_ms=args.delay_ms, goal_dim=goal_dim)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy = policy.to(device)
    policy.eval()

    print("Model loaded.", flush=True)

    print("MODEL_PATH:", args.model_path, flush=True)
    first_param = next(policy.parameters()).detach().float().mean().item()
    print("MODEL_PARAM_MEAN:", first_param, flush=True)
    print("GOAL_DIM:", policy.goal_dim, flush=True)

    success_count = 0
    compute_times = []

    print("\nStarting evaluation...\n", flush=True)

    # ---- Real-time benchmark ----
    # Use a dummy goal for benchmark (shape doesn't matter; it just calibrates runtime)
    dummy_goal = torch.zeros(1, goal_dim, device=device)

    def benchmark_policy_fn(state):
        state = _to_policy_input(state, device)
        with torch.no_grad():
            return policy.act(state, dummy_goal)

    env.benchmark_policy(benchmark_policy_fn)
    print("Real-time benchmark done.", flush=True)

    for ep in range(args.episodes):

        obs = env.reset().to(device)

        target_shape = np.random.choice(possible_shapes)
        target_color = np.random.choice(possible_colors)
        target_size  = np.random.choice(possible_sizes)

        goal_vec = make_goal_vec(
            target_shape, target_color, target_size,
            shape_to_i, color_to_i, size_to_i,
            goal_dim, device
        )

        episode_success = False

        for step in range(args.steps_per_episode):

            def policy_fn(state):
                state = _to_policy_input(state, device)

                start_time = time.time()
                with torch.no_grad():
                    action = policy.act(state, goal_vec)
                compute_times.append(time.time() - start_time)

                return action

            step_out = env.step(policy_fn)

            if isinstance(step_out, tuple):
                obs, info = step_out
            else:
                obs, info = step_out, None

            obs = obs.to(device)

            state = env._get_state()

            if (
                state["shape"] == target_shape and
                state["color"] == target_color and
                state["size"]  == target_size
            ):
                episode_success = True

        if episode_success:
            success_count += 1

        print(
            f"Episode {ep+1}/{args.episodes} | success={episode_success}",
            flush=True,
        )

    success_rate = success_count / args.episodes
    avg_compute_time_ms = np.mean(compute_times) * 1000

    print("\nEvaluation finished.")
    print("Success rate:", success_rate)
    print("Avg compute time (ms):", avg_compute_time_ms)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "success_rate": success_rate,
        "avg_compute_time_ms": avg_compute_time_ms,
        "delay_ms": args.delay_ms,
        "episodes": args.episodes,
        "steps_per_episode": args.steps_per_episode,
        "seed": args.seed,
    }

    with open(output_path / "eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Metrics saved.")
    print("Done.")


if __name__ == "__main__":
    main()