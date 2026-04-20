import torch
import numpy as np
import time
from robot_walk_env import RobotWalkEnv
from train import PolicyNetwork

def run_validation_sample(checkpoint_path="checkpoints/best_policy.pt"):
    print("======================================================")
    print(" SpiderBot Control: Single Validation Sample Evaluator  ")
    print("======================================================")
    
    # 1. Initialize environment in validation mode (no rendering for fast evaluation, 
    # but you can change render=True to watch it)
    env = RobotWalkEnv(render=True)
    
    obs_dim = env.observation_space.shape[0]  # 24 features
    act_dim = env.action_space.shape[0]       # 8 joints
    
    # 2. Load the trained policy
    policy = PolicyNetwork(obs_dim, act_dim)
    try:
        policy.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"[INFO] Successfully loaded trained policy from {checkpoint_path}")
    except FileNotFoundError:
        print(f"[WARNING] Model {checkpoint_path} not found.")
        print("[WARNING] Running with an untrained/random policy for demonstration.")
    
    policy.eval()
    
    # CRITICAL: set total_steps very high so the exploration noise (sine weight) = 0
    # We want to evaluate the purely deterministic, learned behavior.
    env.total_steps = 999999
    
    # 3. Reset environment and apply validation-specific perturbations
    obs, _ = env.reset()
    
    # --- VALIDATION SET PERTURBATIONS (As defined in Part 2) ---
    # Apply randomized initial sensor noise
    obs = obs + np.random.normal(0, 0.015, size=obs.shape)
    # -----------------------------------------------------------
    
    total_reward = 0.0
    step = 0
    success_threshold = 500  # Steps required without falling to classify as a "Success"
    
    print("\n[INFO] Starting validation episode with injected sensor noise...")
    
    while True:
        with torch.no_grad():
            # Get deterministic action from the neural network
            action = policy.get_action(obs, deterministic=True)
            
        # Step the physics environment
        obs, reward, done, _, _ = env.step(action)
        
        # Continuously inject mild validation noise to the IMU and joint sensors
        obs = obs + np.random.normal(0, 0.01, size=obs.shape)
        
        total_reward += reward
        step += 1
        
        # Optional: slow down loop if rendering so it's watchable
        time.sleep(1.0 / 240.0)
        
        if done or step >= 1500:
            break

    # 4. Evaluate and "Classify" the trajectory
    print("\n======================================================")
    print("                 VALIDATION RESULTS                   ")
    print("======================================================")
    print(f"Total Steps Survived : {step}")
    print(f"Total Episode Reward : {total_reward:.2f}")
    
    # Transform continuous RL evaluation into a binary classification problem
    if step >= success_threshold and total_reward > 50.0:
        classification = "CORRECT (Successful Locomotion)"
    else:
        classification = "INCORRECT (Failed/Fell/Instability)"
        
    print(f"Trajectory Class     : {classification}")
    print("======================================================\n")

if __name__ == "__main__":
    run_validation_sample()