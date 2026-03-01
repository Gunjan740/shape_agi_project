import argparse
import collections
import shutil
import json
import random
import numpy as np
import torch
import yaml
from pathlib import Path
import os
import sys

print("[train_reinforce.py] module loaded", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Shape AGI Training (REINFORCE)")
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


def make_goal_vec(target_shape, target_color, target_size, shape_to_i, color_to_i, size_to_i, goal_dim, device):
    g = torch.zeros(1, goal_dim, device=device)
    si = shape_to_i[target_shape]
    ci = color_to_i[target_color]
    zi = size_to_i[target_size]
    g[0, si] = 1.0
    g[0, len(shape_to_i) + ci] = 1.0
    g[0, len(shape_to_i) + len(color_to_i) + zi] = 1.0
    return g


def compute_reward(state, target_shape, target_color, target_size, phase):
    """
    Compute per-step reward and phase-hit flag based on curriculum phase.

    Phase 1: shape + size (actions 2/3 control shape_size_idx — both change together)
    Phase 2: full task (shape + size + color)

    phase_hit  — used to advance curriculum
    episode_hit — full-tuple match, used for the episodic bonus (evaluation metric)
    """
    shape_match = state["shape"] == target_shape
    color_match = state["color"] == target_color
    size_match  = state["size"]  == target_size

    full_match = shape_match and color_match and size_match
    phase_hit  = False

    if phase == 1:
        if shape_match and size_match:
            reward    = 2.0
            phase_hit = True
        else:
            reward = 0.0
            if shape_match: reward += 0.3
            if size_match:  reward += 0.3

    else:  # phase 2 — full task with partial credit
        if full_match:
            reward    = 3.0
            phase_hit = True
        else:
            reward = 0.0
            if shape_match: reward += 0.2
            if color_match: reward += 0.2
            if size_match:  reward += 0.3

    return reward, phase_hit, full_match


def main():
    print("[train_reinforce.py] ENTER main()", flush=True)
    print("Python:", sys.executable, flush=True)
    print("CWD:", os.getcwd(), flush=True)

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, output_path / "config.yaml")

    set_seed(args.seed)
    config = load_config(args.config)

    from envs import ShapeEnv
    env_cfg = config.get("env", {})
    env = ShapeEnv(
        time_scaling=float(env_cfg.get("time_scaling", 1.0)),
        noisy=bool(env_cfg.get("noisy", False)),
        random_initial=True,
        device=str(device)
    )

    possible_shapes = env.shapes_list
    possible_colors = env.colors_list
    possible_sizes  = env.sizes_list

    shape_to_i = {s: i for i, s in enumerate(possible_shapes)}
    color_to_i = {c: i for i, c in enumerate(possible_colors)}
    size_to_i  = {z: i for i, z in enumerate(possible_sizes)}

    goal_dim = len(possible_shapes) + len(possible_colors) + len(possible_sizes)

    from policies.policy import Policy
    policy_cfg = config.get("policy", {})
    policy = Policy(delay_ms=int(policy_cfg.get("delay_ms", 0)), goal_dim=goal_dim)
    policy = policy.to(device)

    train_cfg          = config["training"]
    learning_rate      = train_cfg["learning_rate"]
    num_steps          = train_cfg["num_steps"]
    entropy_coef       = float(train_cfg.get("entropy_coef", 0.01))
    gamma              = float(train_cfg.get("gamma", 0.99))
    curriculum_window  = int(train_cfg.get("curriculum_window", 200))
    curriculum_thresh  = float(train_cfg.get("curriculum_threshold", 0.3))
    steps_per_episode  = int(train_cfg.get("steps_per_episode", 20))

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    num_episodes      = num_steps // steps_per_episode

    total_reward  = 0.0
    reward_history = []

    # Episode-level running baseline; reset on phase advance
    baseline      = 0.0
    baseline_beta = 0.9

    # Curriculum state
    curriculum_phase = 1
    phase_window     = collections.deque(maxlen=curriculum_window)

    print(f"\nStarting episodic training loop (curriculum phase {curriculum_phase})...", flush=True)

    for episode in range(num_episodes):

        obs = env.reset().to(device)

        target_shape = random.choice(possible_shapes)
        target_color = random.choice(possible_colors)
        target_size  = random.choice(possible_sizes)

        goal_vec = make_goal_vec(
            target_shape, target_color, target_size,
            shape_to_i, color_to_i, size_to_i,
            goal_dim, device
        )

        # Trajectory buffers
        ep_log_probs  = []
        ep_entropies  = []
        ep_rewards    = []
        episode_hit   = False   # full-tuple — for episodic bonus
        ep_phase_hit  = False   # phase-specific — for curriculum advancement

        # ── Collect full episode trajectory ───────────────────────────────────
        for _ in range(steps_per_episode):

            features = policy.encoder(obs)
            x = torch.cat([features, goal_vec], dim=1)
            logits = policy.head(x)

            probs = torch.softmax(logits, dim=1)
            dist  = torch.distributions.Categorical(probs)

            action_index = dist.sample()
            ep_log_probs.append(dist.log_prob(action_index))
            ep_entropies.append(dist.entropy())

            action = torch.zeros(5, device=device)
            action[action_index.item()] = 1.0

            next_obs = env._step_simulation(action)
            next_obs = next_obs.unsqueeze(0).to(device)

            state = env._get_state()

            reward, phase_hit, full_match = compute_reward(
                state, target_shape, target_color, target_size, curriculum_phase
            )

            if phase_hit:
                ep_phase_hit = True
            if full_match:
                episode_hit = True

            ep_rewards.append(reward)
            obs = next_obs

        # ── Episodic strict bonus: only for full-tuple hit (aligns with eval) ─
        if episode_hit:
            ep_rewards[-1] += 5.0

        # ── Compute discounted returns backwards ───────────────────────────────
        returns = []
        G = 0.0
        for r in reversed(ep_rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # ── Episode-level baseline update ──────────────────────────────────────
        episode_return = returns[0].item()
        baseline = baseline_beta * baseline + (1 - baseline_beta) * episode_return

        # ── Advantage ─────────────────────────────────────────────────────────
        advantages = returns - baseline

        # ── Single weight update ───────────────────────────────────────────────
        log_probs_t = torch.stack(ep_log_probs)
        entropies_t = torch.stack(ep_entropies)

        policy_loss  = -(log_probs_t * advantages).sum()
        entropy_loss = -entropy_coef * entropies_t.sum()
        loss = policy_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        # ── Curriculum advancement ─────────────────────────────────────────────
        phase_window.append(float(ep_phase_hit))
        if curriculum_phase < 2 and len(phase_window) == curriculum_window:
            hit_rate = sum(phase_window) / len(phase_window)
            if hit_rate >= curriculum_thresh:
                curriculum_phase += 1
                phase_window.clear()
                baseline = 0.0  # reset baseline — reward scale changes
                print(
                    f"\n>>> Curriculum advanced to phase {curriculum_phase} "
                    f"at episode {episode+1} (hit_rate={hit_rate:.2f})\n",
                    flush=True
                )

        # ── Logging ───────────────────────────────────────────────────────────
        episode_reward = sum(ep_rewards)
        total_reward  += episode_reward
        reward_history.append(total_reward)

        if (episode + 1) % 50 == 0 or episode == 0:
            print(
                f"Episode {episode+1:4d} | phase={curriculum_phase} | "
                f"ep_return={episode_return:6.2f} | baseline={baseline:6.2f} | "
                f"phase_hit={ep_phase_hit} | full_hit={episode_hit} | "
                f"total={total_reward:.1f}",
                flush=True
            )

    print("Training finished.")
    print("Total reward:", total_reward)

    torch.save(policy.state_dict(), output_path / "model.pt")

    metrics = {
        "total_reward": total_reward,
        "num_steps": num_steps,
        "seed": args.seed,
        "final_curriculum_phase": curriculum_phase
    }

    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(output_path / "reward_curve.npy", np.array(reward_history))

    print("Model and metrics saved.")
    print("Setup complete.")


if __name__ == "__main__":
    main()
