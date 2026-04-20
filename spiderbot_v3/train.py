import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from robot_walk_env import RobotWalkEnv


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mean = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = self.net(x)
        mean = torch.tanh(self.mean(x))
        std = torch.exp(self.log_std.clamp(-2.0, 0.5))
        return mean, std

    def get_action(self, obs, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        mean, std = self.forward(obs_t)
        if deterministic:
            return mean.squeeze(0).detach().numpy()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return (
            action.squeeze(0).detach().numpy(),
            log_prob.squeeze(0).detach().item(),
            mean.squeeze(0).detach().numpy(),
        )


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0.0
            gae = 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = values[t]
    advantages = np.array(advantages, dtype=np.float32)
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def ppo_update(policy, value_net, policy_opt, value_opt,
               obs_batch, act_batch, logp_batch, adv_batch, ret_batch,
               clip_eps=0.2, epochs=10, batch_size=256):
    """PPO clipped objective update."""
    obs_t  = torch.FloatTensor(obs_batch)
    act_t  = torch.FloatTensor(act_batch)
    logp_t = torch.FloatTensor(logp_batch)
    adv_t  = torch.FloatTensor(adv_batch)
    ret_t  = torch.FloatTensor(ret_batch)

    # Normalize advantages
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    dataset_size = len(obs_t)

    for _ in range(epochs):
        indices = np.random.permutation(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            mb_obs  = obs_t[idx]
            mb_act  = act_t[idx]
            mb_logp = logp_t[idx]
            mb_adv  = adv_t[idx]
            mb_ret  = ret_t[idx]

            mean, std = policy(mb_obs)
            dist = torch.distributions.Normal(mean, std)
            new_logp = dist.log_prob(mb_act).sum(-1)
            entropy  = dist.entropy().sum(-1).mean()

            ratio = torch.exp(new_logp - mb_logp)
            clip_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_loss = -torch.min(ratio * mb_adv, clip_ratio * mb_adv).mean()
            policy_loss -= 0.01 * entropy  # entropy bonus

            policy_opt.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy_opt.step()

            value_pred = value_net(mb_obs)
            value_loss = nn.functional.mse_loss(value_pred, mb_ret)

            value_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            value_opt.step()


def train():
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    env = RobotWalkEnv(render=False)

    obs_dim = env.observation_space.shape[0]  # 24
    act_dim = env.action_space.shape[0]       # 8

    policy    = PolicyNetwork(obs_dim, act_dim)
    value_net = ValueNetwork(obs_dim)

    policy_opt = optim.Adam(policy.parameters(), lr=3e-4)
    value_opt  = optim.Adam(value_net.parameters(), lr=1e-3)

    # --- Resume from checkpoint if available ---
    start_iteration = 0
    checkpoint_path = os.path.join(save_dir, "latest_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path)
        policy.load_state_dict(ckpt["policy"])
        value_net.load_state_dict(ckpt["value_net"])
        policy_opt.load_state_dict(ckpt["policy_opt"])
        value_opt.load_state_dict(ckpt["value_opt"])
        start_iteration = ckpt["iteration"]
        env.total_steps = ckpt.get("total_steps", 0)
        print(f"  Resumed at iteration {start_iteration}, "
              f"total_steps={env.total_steps}")

    # --- Training loop ---
    steps_per_rollout = 4096
    max_iterations = 3000
    best_reward = -float("inf")

    for iteration in range(start_iteration, max_iterations):
        # Collect rollout
        all_obs, all_act, all_logp, all_rew, all_val, all_done = (
            [], [], [], [], [], []
        )

        obs, _ = env.reset()
        ep_reward = 0.0
        ep_rewards = []
        ep_lengths = []
        ep_len = 0

        for _ in range(steps_per_rollout):
            action, log_prob, _ = policy.get_action(obs)
            value = value_net(torch.FloatTensor(obs).unsqueeze(0)).item()

            next_obs, reward, done, _, _ = env.step(action)

            all_obs.append(obs)
            all_act.append(action)
            all_logp.append(log_prob)
            all_rew.append(reward)
            all_val.append(value)
            all_done.append(done)

            ep_reward += reward
            ep_len += 1
            obs = next_obs

            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                ep_reward = 0.0
                ep_len = 0
                obs, _ = env.reset()

        # Compute advantages and returns
        advantages, returns = compute_gae(
            all_rew, all_val, all_done, gamma=0.99, lam=0.95
        )

        # PPO update
        ppo_update(
            policy, value_net, policy_opt, value_opt,
            np.array(all_obs), np.array(all_act), np.array(all_logp),
            advantages, returns,
            clip_eps=0.2, epochs=10, batch_size=256,
        )

        # --- Logging ---
        mean_rew = np.mean(ep_rewards) if ep_rewards else 0.0
        mean_len = np.mean(ep_lengths) if ep_lengths else 0
        explore_w = max(0.0, 1.0 - env.total_steps / 200000.0)

        print(
            f"Iter {iteration:4d} | "
            f"episodes {len(ep_rewards):3d} | "
            f"mean_reward {mean_rew:8.2f} | "
            f"mean_len {mean_len:6.0f} | "
            f"total_steps {env.total_steps:,} | "
            f"sine_weight {explore_w:.2f}"
        )

        # --- Save checkpoints ---
        # Save latest (for resuming)
        torch.save({
            "iteration": iteration + 1,
            "policy": policy.state_dict(),
            "value_net": value_net.state_dict(),
            "policy_opt": policy_opt.state_dict(),
            "value_opt": value_opt.state_dict(),
            "total_steps": env.total_steps,
        }, checkpoint_path)

        # Save best
        if mean_rew > best_reward and len(ep_rewards) > 0:
            best_reward = mean_rew
            torch.save(policy.state_dict(),
                       os.path.join(save_dir, "best_policy.pt"))
            print(f"  >>> New best reward: {best_reward:.2f}")

        # Save periodic snapshots
        if (iteration + 1) % 100 == 0:
            torch.save(policy.state_dict(),
                       os.path.join(save_dir, f"policy_iter_{iteration+1}.pt"))

    env.close()
    print("Training complete.")


if __name__ == "__main__":
    train()
