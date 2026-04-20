import torch
import numpy as np
import time
from robot_walk_env import RobotWalkEnv
from train import PolicyNetwork


def watch(checkpoint_path="checkpoints/best_policy.pt", episodes=5):
    env = RobotWalkEnv(render=True)

    obs_dim = env.observation_space.shape[0]  # 24
    act_dim = env.action_space.shape[0]       # 8

    policy = PolicyNetwork(obs_dim, act_dim)
    policy.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    policy.eval()

    # CRITICAL: set total_steps very high so sine weight = 0
    # This ensures you see ONLY what the neural network learned
    env.total_steps = 999999

    print(f"Loaded policy from {checkpoint_path}")
    print("Sine weight = 0 (pure policy output)")
    print("-" * 50)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        step = 0

        while True:
            # Deterministic action (no noise) for clean visualization
            action = policy.get_action(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            step += 1

            if done:
                break

        # Get final x position
        import pybullet as p
        pos, _ = p.getBasePositionAndOrientation(env.robot)

        print(
            f"Episode {ep+1}: "
            f"steps={step}, "
            f"reward={total_reward:.2f}, "
            f"distance={pos[0]:.3f}m"
        )

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_policy.pt"
    watch(path)
