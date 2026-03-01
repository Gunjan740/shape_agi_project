import argparse
import collections
import shutil
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
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


def compute_gae(rewards, values, last_value, gamma, gae_lambda):
    """Compute GAE advantages and TD(lambda) returns for one episode."""
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def main():
    print("[train.py] ENTER main()", flush=True)
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

    train_cfg         = config["training"]
    learning_rate     = train_cfg["learning_rate"]
    num_steps         = train_cfg["num_steps"]
    entropy_coef      = float(train_cfg.get("entropy_coef", 0.01))
    gamma             = float(train_cfg.get("gamma", 0.99))
    curriculum_window = int(train_cfg.get("curriculum_window", 200))
    curriculum_thresh = float(train_cfg.get("curriculum_threshold", 0.3))
    steps_per_episode = int(train_cfg.get("steps_per_episode", 20))

    # PPO hyperparameters
    ppo_epochs      = int(train_cfg.get("ppo_epochs", 4))
    ppo_clip        = float(train_cfg.get("ppo_clip", 0.2))
    gae_lambda      = float(train_cfg.get("gae_lambda", 0.95))
    batch_episodes  = int(train_cfg.get("batch_episodes", 20))
    value_loss_coef = float(train_cfg.get("value_loss_coef", 0.5))

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    num_episodes   = num_steps // steps_per_episode
    num_batches    = num_episodes // batch_episodes

    total_reward   = 0.0
    reward_history = []
    global_episode = 0

    # Curriculum state
    curriculum_phase = 1
    phase_window     = collections.deque(maxlen=curriculum_window)

    print(f"\nStarting PPO training (curriculum phase {curriculum_phase})...", flush=True)
    print(f"  Episodes: {num_episodes} | Batches: {num_batches} | batch_episodes: {batch_episodes}", flush=True)

    for batch_idx in range(num_batches):

        # ── Collect batch_episodes rollouts (no_grad) ─────────────────────────
        all_obs        = []
        all_actions    = []
        all_log_probs  = []
        all_advantages = []
        all_returns    = []
        all_goals      = []

        batch_full_hits  = 0
        batch_phase_hits = 0
        batch_return_sum = 0.0

        for ep_in_batch in range(batch_episodes):
            global_episode += 1

            obs = env.reset().to(device)

            target_shape = random.choice(possible_shapes)
            target_color = random.choice(possible_colors)
            target_size  = random.choice(possible_sizes)

            goal_vec = make_goal_vec(
                target_shape, target_color, target_size,
                shape_to_i, color_to_i, size_to_i,
                goal_dim, device
            )

            ep_obs       = []
            ep_actions   = []
            ep_log_probs = []
            ep_rewards   = []
            ep_values    = []
            episode_hit  = False
            ep_phase_hit = False

            with torch.no_grad():
                for _ in range(steps_per_episode):
                    logits, value = policy(obs, goal_vec)
                    probs = torch.softmax(logits, dim=1)
                    dist  = torch.distributions.Categorical(probs)
                    action_idx = dist.sample()

                    ep_obs.append(obs)
                    ep_actions.append(action_idx)
                    ep_log_probs.append(dist.log_prob(action_idx).item())
                    ep_values.append(value.item())

                    action = torch.zeros(5, device=device)
                    action[action_idx.item()] = 1.0

                    next_obs = env._step_simulation(action)
                    next_obs = next_obs.unsqueeze(0).to(device)

                    state = env._get_state()
                    reward, phase_hit, full_match = compute_reward(
                        state, target_shape, target_color, target_size, curriculum_phase
                    )

                    if phase_hit:  ep_phase_hit = True
                    if full_match: episode_hit  = True

                    ep_rewards.append(reward)
                    obs = next_obs

                # Episodic bonus for full-tuple hit (aligns with eval metric)
                if episode_hit:
                    ep_rewards[-1] += 5.0

                # Bootstrap value at end of episode
                _, last_val = policy(obs, goal_vec)
                last_val = last_val.item()

            # GAE
            advantages, returns = compute_gae(ep_rewards, ep_values, last_val, gamma, gae_lambda)

            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)
            all_log_probs.extend(ep_log_probs)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
            all_goals.extend([goal_vec] * steps_per_episode)

            if episode_hit:  batch_full_hits  += 1
            if ep_phase_hit: batch_phase_hits += 1
            batch_return_sum += sum(ep_rewards)

            # Curriculum advancement
            phase_window.append(float(ep_phase_hit))
            if curriculum_phase < 2 and len(phase_window) == curriculum_window:
                hit_rate = sum(phase_window) / len(phase_window)
                if hit_rate >= curriculum_thresh:
                    curriculum_phase += 1
                    phase_window.clear()
                    print(
                        f"\n>>> Curriculum advanced to phase {curriculum_phase} "
                        f"at episode {global_episode} (hit_rate={hit_rate:.2f})\n",
                        flush=True
                    )

        # ── Build tensors ──────────────────────────────────────────────────────
        obs_t        = torch.cat(all_obs, dim=0)                                         # (N, C, H, W)
        actions_t    = torch.stack(all_actions)                                          # (N,)
        old_lp_t     = torch.tensor(all_log_probs,  dtype=torch.float32, device=device) # (N,)
        goals_t      = torch.cat(all_goals, dim=0)                                       # (N, goal_dim)
        returns_t    = torch.tensor(all_returns,    dtype=torch.float32, device=device) # (N,)
        advantages_t = torch.tensor(all_advantages, dtype=torch.float32, device=device) # (N,)

        # Normalize advantages across the full batch
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ── PPO update epochs ──────────────────────────────────────────────────
        for _ in range(ppo_epochs):
            logits, values = policy(obs_t, goals_t)
            probs = torch.softmax(logits, dim=1)
            dist  = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_t)
            entropy       = dist.entropy()

            ratio = torch.exp(new_log_probs - old_lp_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages_t
            policy_loss  = -torch.min(surr1, surr2).mean()

            value_loss   = F.mse_loss(values, returns_t)
            entropy_loss = -entropy_coef * entropy.mean()

            loss = policy_loss + value_loss_coef * value_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────────
        total_reward += batch_return_sum
        reward_history.append(total_reward)

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(
                f"Batch {batch_idx+1:4d}/{num_batches} | ep={global_episode:5d} | "
                f"phase={curriculum_phase} | "
                f"full_hit_rate={batch_full_hits/batch_episodes:.2f} | "
                f"avg_return={batch_return_sum/batch_episodes:.2f} | "
                f"policy_loss={policy_loss.item():.4f} | "
                f"value_loss={value_loss.item():.4f}",
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
