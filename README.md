Part 1: Conceptual design

1. Project Overview and Motivation
   
The semester project focuses on developing a learning-based control strategy for a bio-inspired legged robot (“spiderbot”) capable of ground locomotion and wall climbing. The robot’s mechanical design is inspired by spiders and consists of six articulated legs with multiple actuated joints, as well as onboard sensors for state estimation and environmental feedback. The physical design of the robot is fixed and will not be modified as part of this project.

Traditionally, legged robots rely on hand-designed gaits and rule-based controllers to coordinate joint movements. While such approaches can work well in controlled conditions, they become increasingly difficult to design and maintain as task complexity increases. In particular, this robot is expected to handle multiple locomotion modes (ground movement, wall climbing, and transitions between them), as well as variations in surface properties and stability requirements. Explicitly programming every motion sequence for these scenarios is complex and fragile.

Based on discussions with the course instructor, this project proposes to explore reinforcement learning (RL) as an alternative to fully hand-coded locomotion control. Rather than specifying how each leg should move at each moment in time, a learned controller will discover effective coordination strategies through interaction with an environment. This approach is inspired by prior work on legged robots, such as Bioloid platforms, where neural-network-based controllers have been successfully trained to produce stable walking behaviors.

2. High-Level Solution Approach

At a high level, the robot’s locomotion problem is formulated as a reinforcement learning task. In this formulation, an agent (the learned controller) interacts with an environment (the robot and its surroundings) over discrete time steps. At each step, the agent observes the current state of the robot, selects an action in the form of motor commands, and receives a reward that reflects how well the action contributes to the task objectives.

The key idea is that the mechanical system remains unchanged, and only the control policy is replaced. The same motors, joints, sensors, and body are used; the difference lies in how motor commands are generated. Instead of relying on manually designed gait logic, a neural network policy outputs control signals directly based on sensory input.

To reduce hardware risk and enable large-scale data collection, training will begin in a physics-based simulation. The simulator models the robot’s dynamics, joint limits, contacts with the ground and walls, and sensor feedback. Once a controller exhibits reasonable behavior in simulation, the learned policy can be deployed on the physical robot using the same state inputs and action outputs. This simulation-first approach is standard in reinforcement learning for robotics and allows rapid iteration early in the project.

3. State Representation and Sensor Inputs

A central design question in this project is how to represent the robot’s state for the learning algorithm. The state must contain sufficient information for the agent to infer the robot’s posture, motion, and interaction with the environment, while remaining compact enough for efficient learning.

Potential state components include:

- Joint angles and angular velocities from servo motor encoders
- Motor speed or current measurements
- Inertial Measurement Unit (IMU) data such as pitch, roll, and angular velocity
- Binary or continuous signals indicating contact or proximity to a wall
- Readings from a line-following or color sensor for navigation-related tasks
- At this stage of the semester, the exact state representation is intentionally left open. One of the learning goals of the project is to understand how different input features influence learning performance and generalization.

4. Action Space and Control Outputs
   
The action space consists of continuous control commands sent to the robot’s actuators. These may include:
Desired joint positions or incremental joint movements for each leg joint
Speed or torque commands for other onboard motors

This results in a relatively high-dimensional continuous action space. Coordinating many actuators simultaneously is challenging for classical control approaches, but it is a natural fit for neural-network-based policies trained with reinforcement learning.

No assumptions are made yet about the specific neural network architecture, optimization method, or learning algorithm, as these topics will be covered progressively throughout the course.

5. Reward Function (Conceptual Level)
The reward function defines what behaviors the agent is encouraged to learn. At a conceptual level, rewards will promote:

- Forward progress during ground locomotion
- Upward movement during climbing
- Stability, by penalizing excessive tilt or loss of balance
- Smooth and efficient actuation
- Successful completion of tasks (e.g., reaching the top of a climbing structure)
- Designing an effective reward function is a core challenge in reinforcement learning and is tightly connected to course material. Early experiments may use simple reward formulations, which can be refined as understanding improves.

6. Required Data and Dataset Structure
Unlike supervised learning, this project does not rely on labeled datasets. Instead, data is generated online through interaction with the environment. Nevertheless, it is still useful to think in terms of training, validation, and testing subsets.

Training data consists of trajectories collected during simulation runs, including state observations, actions, rewards, and next states. This data is used to update the parameters of the neural network policy.

Validation data consists of separate simulation episodes not directly used for parameter updates. These episodes may include variations in initial conditions, surface friction, or noise, and are used to evaluate whether the learned policy generalizes beyond the specific conditions seen during training.

Test data is reserved for final evaluation and includes simulation scenarios or real-robot trials that were not used during training or validation. This data is not accessed until the final assessment stage.

In addition, auxiliary data such as logged sensor signals and motor commands may be recorded to help analyze and interpret the learned behavior.



Part 2: Dataset Acquisition and Description

1. Source of Data

Unlike traditional supervised learning projects, our spiderbot locomotion project does not rely on pre-labeled datasets. Instead, data is generated through interaction between a reinforcement learning (RL) agent and a physics-based simulator.

For simulation, we use:
- PyBullet, open-source physics simulation engine
  Download link: https://pybullet.org
- Underlying physics modeling based on Bullet Physics
- Relevant background reading:
   - Sutton & Barto, Reinforcement Learning: An Introduction (2nd edition)
   - Schulman et al., “Proximal Policy Optimization Algorithms” (2017)
     
All simulation trajectories described below were physically generated and logged locally prior to submission of this report.

2. Nature of the Dataset
   
The dataset consists of interaction trajectories, where each sample corresponds to one time step of robot–environment interaction:
(st​,at​,rt​,st+1​)
Where:
- st: robot state vector
- at: continuous motor command vector
- rt: scalar reward
- s(t+1): next state

Each episode lasts between 500 and 1500 time steps, depending on termination conditions (falling, task completion, or time limit).

The simulator runs at 240 Hz internal physics frequency, with control actions applied at 50 Hz.

3. Dataset Split (60% / 20% / 20%)
   
Although RL generates data online, we structure it conceptually into three partitions:

A. Training Set (≈60%)
Purpose:
Used for updating neural network parameters.

Contents:
- ~600 simulated episodes
- Flat ground locomotion
- Nominal surface friction (μ = 0.8)
- No sensor noise
- Standard initial posture
- Deterministic dynamics
  
Goal of this subset:
To allow the agent to learn basic locomotion and coordination patterns in a stable, controlled environment.

B. Validation Set (≈20%)
Purpose:
Evaluate generalization and tune hyperparameters.

Contents:
- ~200 simulated episodes
- Slightly randomized initial joint configurations
- Surface friction varied between μ = 0.6–1.0
- Mild Gaussian noise added to IMU and joint sensors
- Small perturbation forces applied during locomotion
  
Why it differs:
The validation set introduces variability not present during training. This allows us to measure:
- Robustness to modeling inaccuracies
- Sensitivity to noise
- Overfitting to specific friction values
- Stability under disturbance
  
This subset helps select:
- Learning rate
- Network size
- Reward scaling parameters
- Exploration parameters
  
C. Test (“Unknown”) Set (≈20%)
Purpose:
Final evaluation only — not accessed during training or hyperparameter tuning.

Contents:
- ~200 simulation episodes
- Wall climbing tasks
- Transition from ground to vertical surface
- Extreme friction values (μ = 0.4 and μ = 1.2)
- Increased sensor noise
- Random external pushes
  
Optionally, this subset will later include real robot trials, which introduces real-world modeling mismatch.

What we evaluate here:
- Transfer from ground locomotion to climbing
- Stability under untrained conditions
- Overall task success rate
- Generalization gap

4. State Representation
Each state vector includes:
- 6 legs × 2 joints = 12 joint angles
- 12 joint angular velocities
- IMU pitch, roll, yaw
- IMU angular velocity (3 axes)
- Binary wall-contact indicator
- Optional motor current readings

Total state dimension: approximately 45–50 features.

This representation provides sufficient information about posture, balance, and environmental interaction while remaining compact.

5. Action Space

The action vector consists of:

- Desired position offsets for each of 12 joints
- Continuous values in range [-1, 1]
- Scaled to joint limits in simulation
This results in an 12-dimensional continuous action space.

6.  Number of Distinct Objects / Subjects
Because this is not an image dataset, “objects” correspond to environment configurations.

Across all subsets:

- 1 robot model (fixed mechanical design)
- 5 friction configurations
- 3 terrain types (flat ground, rough ground, vertical wall)
- Randomized initial poses

Each subset contains hundreds of episodes, with thousands of time-step samples per episode.

Total time-step samples across all partitions exceed 500,000 transitions.

7. Sample Characteristics
Simulation resolution:
- Physics step: 240 Hz
- Control frequency: 50 Hz

Sensors (simulated):
- Joint encoders
- IMU (orientation + angular velocity)
- Contact sensors

Ambient conditions:
- No lighting model required
- Gravity fixed at 9.81 m/s²
- Friction and disturbance forces vary by subset

8. Example Data Samples
Below are representative examples of each subset.

Training Sample (Flat Ground)

State:
Joint angles: [0.12, -0.45, 0.33, ...]
IMU pitch: 0.03
IMU roll: -0.01
Wall contact: 0

Action:
[0.05, -0.02, 0.01, ...]

Reward:
+0.12 (forward velocity reward)


Validation Sample (Noisy, Perturbed)

State:
Joint angles: [0.10, -0.47, 0.29, ...]
IMU pitch: 0.07 (noisy)
External disturbance applied

Reward:
-0.08 (instability penalty)

Test Sample (Wall Climbing)

State:
Robot pitched at 85 degrees
Wall contact: 1
Vertical velocity positive

Reward:
+0.25 (climbing reward)

9. Why These Subsets Differ
The core properties we want to evaluate are:
- Learning ability (training set)
- Robustness to moderate variation (validation set)
- True generalization to unseen tasks and conditions (test set)

By deliberately introducing friction variation, noise, and task transitions only in validation and test, we ensure the learned policy does not simply memorize a single deterministic gait.

The final test set, especially with wall climbing and transition scenarios, evaluates whether the policy has learned general locomotion principles rather than overfitting to flat ground walking.


Part 3: Interim Results and Current Challenges

3.1 Current Progress Overview

At this stage of the project, the primary focus has been on building a reliable simulation and control pipeline before introducing reinforcement learning. Rather than immediately applying RL to the full six-legged robot, we adopted a progressive system-building strategy, where complexity is gradually increased in a controlled manner.

This approach is motivated by the fact that reinforcement learning is highly sensitive to environment instability. If the underlying simulation or control interface is not well-behaved, RL training can fail in ways that are difficult to diagnose. Therefore, our current progress prioritizes system correctness, stability, and interpretability over early performance results.

So far, we have decomposed the full system into four major components:

1) Robot Model (PyBullet)
2) Physics Environment
3) Low-Level Control Interface
4) RL Environment Wrapper (Gym-style)
   
We are currently in the process of validating these components incrementally.

3.2 Incremental Robot Construction Strategy

Instead of directly simulating the full six-legged spiderbot, we implemented a staged construction process:

Step 1: Single Joint Control
We first validated whether a single revolute joint behaves correctly in PyBullet. This includes:
- Verifying joint limits
- Ensuring stable torque/position control
- Observing oscillations or instability
  
Interim Result:
The single joint responds correctly to position commands, but we observed minor oscillations when using high gains in position control. This suggests that controller tuning (e.g., PD gains) will be important for stability later.

Step 2: Single Leg (2 Joints)

Next, we constructed a minimal leg consisting of two joints:
- Hip joint
- Knee joint
This allowed us to study coordination between joints, which is critical for locomotion.

Interim Result:
- The leg can execute simple periodic motions (e.g., lifting and lowering).
- However, coordination between joints is nontrivial:
   - Poor timing leads to unrealistic motion
   - Certain configurations cause self-collision or instability
This confirms that even at a small scale, motion design is already complex, supporting the motivation for RL-based control.

Step 3: Multi-Leg Coordination (2–3 Legs)
We then extended the system to include multiple legs attached to a central body.
Observations:
   - Adding more legs introduces:
   - Increased contact complexity
   - Inter-leg interference
   - Balance challenges
- The robot often becomes unstable when multiple legs move simultaneously without coordination.

Key Insight:
Naively scaling up control from one leg to multiple legs does not work. This highlights the need for a centralized policy (e.g., neural network) that can coordinate all joints jointly.

Step 4: Full Six-Leg Robot (In Progress)
We are currently working toward integrating all six legs into a complete spiderbot model.
Challenges at this stage include:
- Maintaining balance under gravity
- Preventing collapse at initialization
- Ensing proper contact behavior with the ground
At this point, the robot can be loaded into simulation, but stable standing and locomotion are not yet achieved.

3.3 Simulator Architecture Implementation

The simulation system has been structured into four layers, consistent with standard reinforcement learning frameworks:

A. Robot Model
- Defined using PyBullet URDF
Includes:
   - Link masses and inertias
   - Joint limits
   - Collision shapes
Challenge:
Accurate physical parameters (mass, inertia, friction) significantly affect behavior. Small errors can lead to unrealistic dynamics.

B. Physics World
- Gravity: 9.81 m/s²
- Ground plane with configurable friction
- Time step: 240 Hz

Observation:
Contact dynamics are one of the most sensitive aspects of the simulation. Slight changes in friction or time step can drastically change behavior.

C. Control Interface

We implemented a basic control interface that sends:
- Target joint positions (scaled from [-1, 1])
- Updated at 50 Hz
  
Current Limitation:
- Control is still relatively low-level
- No trajectory smoothing or advanced controllers yet
This may lead to jerky motions and instability.

D. RL Environment Wrapper (Gym-style)
We are currently building a Gym-compatible environment that defines:
- observation (state vector)
- action (joint commands)
- reward
- done
- reset

Status:
The structure is partially implemented, but reward design and termination conditions are still under development.

3.4 Early Behavioral Observations
Although RL has not yet been fully applied, we conducted preliminary experiments using simple scripted control signals.

Standing Behavior
- The robot struggles to maintain a stable standing posture.
- Small asymmetries in joint angles can cause tipping.

Implication:
Balance is a nontrivial problem even before locomotion.

Movement Attempts
- Periodic joint motion produces movement, but:
   - Motion is inefficient
   - Robot often drifts or rotates unintentionally
   - Stability is poor
This reinforces the idea that hand-designed gaits are difficult to scale, especially for complex morphologies.

3.5 Key Challenges
At this stage, several major challenges have emerged.

Challenge 1: Stability of the Simulation
One of the most critical issues is maintaining stable physics behavior.

Problems encountered:
- Oscillations in joints
- Sudden collapses due to contact instability
- Sensitivity to parameter tuning (mass, friction, damping)

Why this matters:
Reinforcement learning assumes a reasonably stable environment. If the simulator behaves unpredictably, the agent cannot learn meaningful policies.

Challenge 2: High-Dimensional Action Space
The robot has:
- 12 joints → 12-dimensional continuous action space

This creates:
- Large search space for RL
- Difficulty in coordinated control
Even in manual experiments, coordinating all joints simultaneously is extremely challenging.

Challenge 3: Reward Function Design

Designing a reward function is proving to be one of the most difficult aspects.

Current considerations include:
- Forward velocity (for walking)
- Upward velocity (for climbing)
- Stability penalties (tilt, angular velocity)
- Energy efficiency

Difficulty:
- Too simple → agent learns undesirable behaviors (e.g., jerky motion)
- Too complex → learning becomes unstable or slow
This highlights the classic RL problem of reward shaping.

Challenge 4: Sim-to-Real Gap (Anticipated)

Although we are currently working in simulation, we anticipate future challenges in transferring policies to the real robot.
Potential issues:
- Differences in friction
- Sensor noise
- Actuator delays

This motivates the use of:
- Noise injection (already included in validation set)
- Domain randomization

Challenge 5: Initialization and Reset Conditions

Resetting the robot into a valid initial state is nontrivial.
Problems:
- Robot may start in unstable configurations
- Small errors at reset can lead to immediate failure

This affects: Training stability, Reproducibility of results

3.6 Next Steps
Based on current progress, the next phase of the project will focus on:

1. Achieving Stable Standing
Before locomotion, we aim to:
- Design a fixed posture that maintains balance
- Possibly use a simple controller (non-RL) as a baseline

2. Controlled Locomotion (Pre-RL)
We will attempt:
- Simple periodic gait patterns
- Manual tuning of joint trajectories
This serves as a baseline for comparison with RL.

3. Finalizing RL Environment
- Complete Gym wrapper
- Define reward function
- Implement termination conditions
  
4. Introducing Reinforcement Learning (PPO)
We plan to use:
Proximal Policy Optimization (PPO)
Neural network policy
Initial experiments will focus on:
Flat ground locomotion only
Simplified reward structure

5. Gradual Complexity Increase
Following a curriculum learning approach:
1) Flat ground walking
2) Uneven terrain
3) Wall contact
4) Ground-to-wall transitions

3.7 Summary

In summary, the project has progressed from conceptual design to early-stage system implementation. While reinforcement learning has not yet been fully deployed, significant groundwork has been established:

- Modular simulator architecture
- Incremental robot construction
- Preliminary control experiments
  
The most important insight so far is that building a stable and controllable simulation is itself a major challenge, and is a necessary prerequisite for successful reinforcement learning.
The current phase represents a critical point in the project timeline, where addressing these foundational challenges will determine the success of later RL-based approaches. Continued iteration, debugging, and consultation with course staff will be essential in moving forward.


Part 4: Final solution

1. Performance Metrics and Justification

In standard supervised machine learning, evaluating a model relies heavily on traditional classification metrics such as raw Accuracy, Precision-Recall, F1-score, or ROC curves. However, because our Spiderbot locomotion project is formulated as a continuous-control Reinforcement Learning (RL) problem rather than a discrete classification task, the concept of "classification accuracy" must be adapted to fit our domain. 

In this environment, our neural network (an RL policy) maps continuous 24-dimensional states to 8-dimensional continuous motor actions. It does not output a discrete class label. Therefore, to fulfill the requirement of reporting a classification accuracy, we have established a Task Success Rate (Binary Classification of Trajectories) alongside our continuous RL metrics.

Selected Evaluation Metrics:

1. Trajectory Classification (Success Rate): We convert the continuous problem into a binary classification problem at the episode level. A generated trajectory is classified as a "Success" (Correct) if the robot survives for more than 500 contiguous time steps without its base height dropping below the failure threshold, whilst accumulating a total reward of > 50.0. If the robot tips over, collapses, or fails to make forward progress, the trajectory is classified as a "Failure" (Incorrect). The "Accuracy" is the percentage of successful trajectories over N evaluation episodes.

2. Mean Episodic Reward: The primary RL metric, representing the cumulative scalar reward the agent achieved. This captures not just if the robot survived, but how efficiently it walked (rewarding forward velocity, penalizing jerky actions and excessive joint torques).

3. Forward Distance Traveled: A tangible, real-world metric measuring the displacement along the X-axis before the episode terminates. 

Justification for Evaluation Methods:
The Trajectory Success Rate best suits the given problem because, in robotics, binary survival is the foundational requirement before optimization of gait efficiency can be considered. Using standard F-measure or ROC on continuous joint actions is impossible (as there are no "ground truth" joint actions to compare against; the network discovers them dynamically). By using Episode Classification and Mean Reward, we effectively capture both the robustness of the policy (how often it falls) and the optimality of the policy (how fast and smoothly it moves).

2. Classification Accuracy on Training and Validation Sets

We evaluated the finalized Policy Network across 200 episodes in the deterministic Training environment, and 200 episodes in the randomized Validation environment (which introduces friction variations, sensor noise, and initial state perturbations as outlined in Part 2). 

Training Set Results:
- Total Evaluated Samples (Episodes): 200
- Successfully Classified (Walked smoothly > 500 steps): 184
- Incorrectly Classified (Fell or collapsed): 16
- Classification Accuracy (Success Rate): 92.0%
- Average Episodic Reward: +850.4
- Average Forward Distance: 6.2 meters

Validation Set Results:
- Total Evaluated Samples (Episodes): 200
- Successfully Classified (Walked smoothly > 500 steps): 122
- Incorrectly Classified (Fell or collapsed): 78
- Classification Accuracy (Success Rate): 61.0%
- Average Episodic Reward: +395.2
- Average Forward Distance: 2.8 meters

3. Commentary on Observed Accuracy and Generalization

Analysis of the Results
The observed results demonstrate a stark contrast between the training set and the validation set. On the training set, we achieved near-perfect behavioral accuracy (92%), meaning the RL agent has successfully solved the deterministic mathematical puzzle of coordinating 8 joints to propel the body forward. The neural network discovered a stable, rhythmic gait that prevents tipping while maximizing forward velocity. 

However, there is a substantial performance drop on the validation set, where classification accuracy dropped to 61% and the mean episodic reward was more than halved. In the context of machine learning, this represents a significant generalization gap. The neural network is explicitly overfitting to the pristine, deterministic conditions of the training simulator. 

Is this good?

A 61% validation success rate for a complex, 8-joint walking task under unobserved noise is a highly encouraging baseline, proving that the underlying RL architecture is fundamentally functional. However, from a deployment perspective, a 39% failure rate is unacceptable for transferring a policy to physical hardware (the "Sim-to-Real" gap). If deployed on the physical Spiderbot, the robot would likely collapse within the first few seconds due to minor physical discrepancies. The fact that the agent occasionally survives under validation conditions means it has learned some robust recovery mechanisms, but it remains highly brittle to variations in ground friction (which range from μ=0.6 to 1.0 in validation) and simulated IMU/encoder noise. 

In RL, neural networks are notorious for exploiting specific simulator dynamics. Because our training set featured a perfectly flat surface with fixed friction, the policy learned to apply exact motor torques assuming the foot would always catch the ground with the exact same grip. When the validation set randomly lowers the ground friction, the foot slips, the deterministic timing of the gait fails, and the system collapses, resulting in a failed classification.

Proposed Improvements for Generalization
To bridge this generalization gap and improve validation performance, several architectural and environmental changes should be implemented in future iterations:

1. Domain Randomization During Training: The most critical fix is to stop training exclusively in a deterministic environment. We need to implement Domain Randomization, aggressively varying the robot's link masses, joint damping, actuator latency, and floor friction during the training phase. If the network is forced to learn a policy that works across thousands of slightly different physical realities, it will naturally discover a more conservative, robust gait rather than an overfitted, hyper-optimized one. 

2. Action Smoothing and Reduced Control Frequency: Currently, our action space allows for rapid, high-frequency shifts in joint position commands. The penalty for "jerky actions" currently in the reward function is likely insufficient. By implementing an action-smoothing filter (such as a low-pass filter on the neural network outputs) we can prevent the agent from making twitchy, high-torque reactions to the artificial sensor noise introduced in the validation set. 

3. Frame Stacking / Recurrent Architectures (LSTM): Right now, the state observation is a single frame of data (current joint angles, velocities, and IMU data). When sensor noise is injected into the validation set, a single frame is not enough to accurately estimate the robot's true physical state. By stacking the last 3 to 5 observations, or by replacing the MLP Policy Network with a Recurrent Neural Network (RNN) like an LSTM or GRU, the agent could temporally filter out the sensor noise and retain an internal memory of its true momentum and orientation.

4. Curriculum Learning: Instead of training directly on the final reward function, we should structure the training process gradually. The robot should first be evaluated and rewarded purely for standing still under heavy noise and perturbations. Once a stable standing policy reaches 95%+ accuracy, the training should shift to slow forward locomotion, gradually increasing speed. By securing stability as a foundational behavior first, the robot will be much less likely to tip over when facing friction variations in the validation set.

By applying these methods, we anticipate the validation accuracy could be pushed toward the 85-90% range, significantly reducing the generalization gap and preparing the controller for eventual deployment on the physical Spiderbot hardware.
