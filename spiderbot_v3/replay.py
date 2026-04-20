from stable_baselines3 import PPO
from robot_env import RobotWalkEnv
import os
import time
import glob

env = RobotWalkEnv(render=True)

# Find all snapshots and sort them from earliest to latest
snapshots = sorted(glob.glob("snapshots/policy_step_*.zip"),
                   key=lambda x: int(x.split("_")[-1].replace(".zip", "")))

if not snapshots:
    print("No snapshots found. Run train.py first.")
    exit()

print(f"Found {len(snapshots)} snapshots. Replaying from worst to best...\n")

for snapshot_path in snapshots:
    step_num = snapshot_path.split("_")[-1].replace(".zip", "")
    print(f"--- Showing policy at step {step_num} ---")

    model = PPO.load(snapshot_path)
    obs, _ = env.reset()

    # Watch this snapshot for 500 steps
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        time.sleep(0.01)  # slow it down so you can see it clearly
        if done:
            obs, _ = env.reset()

    print(f"  ep_rew moving on to next snapshot...\n")
    time.sleep(1)  # pause between snapshots so you can see the difference

print("Replay complete! That was the full learning journey.")
env.close()