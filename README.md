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
