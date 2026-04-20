import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
import math


class RobotWalkEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render
        self.joint_names = [
            "fl_hip_joint", "fl_knee_joint",
            "fr_hip_joint", "fr_knee_joint",
            "rl_hip_joint", "rl_knee_joint",
            "rr_hip_joint", "rr_knee_joint",
        ]
        self.num_joints = 8
        self.prev_x = 0.0
        self.step_count = 0
        self.total_steps = 0
        self.robot = None
        self.physics_client = None
        self.world_built = False
        self.prev_action = np.zeros(self.num_joints, dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        # 8 angles + 8 vels + 3 euler + 3 lin_vel + 1 height + 1 phase = 24
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )

    def _build_world(self):
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-20,
                cameraTargetPosition=[0, 0, 0.25]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 240.0,
            numSolverIterations=50,
            numSubSteps=1
        )

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = p.loadURDF("plane.urdf")
        p.changeDynamics(plane, -1,
                         lateralFriction=1.5,
                         restitution=0.0)

        urdf_path = os.path.join(os.path.dirname(__file__), "my_robot.urdf")
        self.robot = p.loadURDF(urdf_path, [0, 0, 0.45])

        for i in range(-1, p.getNumJoints(self.robot)):
            p.changeDynamics(self.robot, i,
                             lateralFriction=1.5,
                             restitution=0.0,
                             linearDamping=0.2,
                             angularDamping=0.2,
                             jointDamping=0.1)

        self.joint_map = {}
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            name = info[1].decode("utf-8")
            if name in self.joint_names:
                self.joint_map[name] = i

        self.world_built = True

    def _place_robot(self):
        p.resetBasePositionAndOrientation(
            self.robot, [0, 0, 0.45], [0, 0, 0, 1]
        )
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])

        standing = {
            "fl_hip_joint":  0.0, "fl_knee_joint": -0.4,
            "fr_hip_joint":  0.0, "fr_knee_joint": -0.4,
            "rl_hip_joint":  0.0, "rl_knee_joint": -0.4,
            "rr_hip_joint":  0.0, "rr_knee_joint": -0.4,
        }
        for name, angle in standing.items():
            if name in self.joint_map:
                p.resetJointState(self.robot,
                                  self.joint_map[name], angle, 0.0)

        for _ in range(200):
            p.stepSimulation()

    def reset(self, seed=None):
        if not self.world_built:
            self._build_world()
        self._place_robot()
        self.step_count = 0
        self.prev_x = 0.0
        self.prev_action = np.zeros(self.num_joints, dtype=np.float32)
        return self._get_obs(), {}

    def _get_sine_actions(self, t):
        """Reference diagonal trot gait pattern."""
        return np.array([
            -0.35 * math.sin(t),                               # fl_hip
            -0.3 + 0.25 * math.sin(t + math.pi / 2),          # fl_knee
             0.35 * math.sin(t),                               # fr_hip
            -0.3 + 0.25 * math.sin(t + math.pi + math.pi/2),  # fr_knee
             0.35 * math.sin(t),                               # rl_hip
            -0.3 + 0.25 * math.sin(t + math.pi + math.pi/2),  # rl_knee
            -0.35 * math.sin(t),                               # rr_hip
            -0.3 + 0.25 * math.sin(t + math.pi / 2),          # rr_knee
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        self.total_steps += 1

        t = self.total_steps * 0.04

        # Curriculum: fade sine from 100% to 0% over 200k steps
        explore_weight = max(0.0, 1.0 - self.total_steps / 200000.0)
        sine_actions = self._get_sine_actions(t)

        joint_limits = [0.4, 0.7, 0.4, 0.7, 0.4, 0.7, 0.4, 0.7]

        for i, joint_name in enumerate(self.joint_names):
            if joint_name not in self.joint_map:
                continue
            joint_idx = self.joint_map[joint_name]
            limit = joint_limits[i]

            rl_target   = float(action[i]) * limit
            sine_target = float(sine_actions[i])

            target = (explore_weight * sine_target +
                      (1.0 - explore_weight) * rl_target)
            target = np.clip(target, -limit, limit)

            p.setJointMotorControl2(
                self.robot, joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=30.0,
                maxVelocity=2.0,
                positionGain=0.3,
                velocityGain=0.1
            )

        # 8 physics sub-steps per RL action (~33ms of sim time)
        for _ in range(8):
            p.stepSimulation()

        # Gentle random push every 500 steps
        if self.step_count % 500 == 0:
            push = np.random.uniform(-1.0, 1.0, 3)
            push[2] = 0
            p.applyExternalForce(
                self.robot, -1,
                push.tolist(), [0, 0, 0],
                p.WORLD_FRAME
            )

        if self.render_mode:
            pos, _ = p.getBasePositionAndOrientation(self.robot)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-20,
                cameraTargetPosition=[pos[0], pos[1], 0.25]
            )
            time.sleep(1.0 / 30.0)

        obs    = self._get_obs()
        reward = self._compute_reward(action)

        # Early termination if fallen or too tilted
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        height = pos[2]
        euler  = p.getEulerFromQuaternion(orn)
        tilt   = abs(euler[0]) + abs(euler[1])

        fallen  = height < 0.15 or tilt > 1.2
        timeout = self.step_count >= 2000
        done    = fallen or timeout

        self.prev_action = np.array(action, dtype=np.float32)

        return obs, reward, done, False, {}

    def _get_obs(self):
        joint_angles = []
        joint_vels   = []
        for name in self.joint_names:
            if name in self.joint_map:
                idx   = self.joint_map[name]
                state = p.getJointState(self.robot, idx)
                joint_angles.append(state[0])
                joint_vels.append(state[1])
            else:
                joint_angles.append(0.0)
                joint_vels.append(0.0)

        pos, orn   = p.getBasePositionAndOrientation(self.robot)
        euler      = p.getEulerFromQuaternion(orn)
        lin_vel, _ = p.getBaseVelocity(self.robot)

        # Phase signal so the policy can learn rhythmic gaits
        phase = math.sin(self.step_count * 0.04)

        return np.array(
            joint_angles + joint_vels +
            list(euler) + list(lin_vel[:3]) + [pos[2]] + [phase],
            dtype=np.float32
        )

    def _compute_reward(self, action):
        pos, orn   = p.getBasePositionAndOrientation(self.robot)
        euler      = p.getEulerFromQuaternion(orn)
        lin_vel, _ = p.getBaseVelocity(self.robot)

        height      = pos[2]
        forward_vel = lin_vel[0]
        tilt        = abs(euler[0]) + abs(euler[1])
        sideways    = abs(lin_vel[1])

        # Primary: forward velocity
        reward = forward_vel * 8.0

        # Primary: forward distance this step
        dx = pos[0] - self.prev_x
        self.prev_x = pos[0]
        reward += dx * 30.0

        # Small survival bonus
        reward += 0.05

        # Height bonus/penalty
        if height > 0.30:
            reward += 0.3
        elif height > 0.20:
            reward -= 0.5
        else:
            reward -= 3.0

        # Penalize tilting
        reward -= tilt * 0.5

        # Penalize sideways drift
        reward -= sideways * 0.3

        # Penalize jerky actions
        action_arr = np.array(action, dtype=np.float32)
        jerk = np.sum(np.square(action_arr - self.prev_action))
        reward -= jerk * 0.05

        # Penalize excessive torque
        total_torque = 0.0
        for name in self.joint_names:
            if name in self.joint_map:
                state = p.getJointState(self.robot, self.joint_map[name])
                total_torque += abs(state[3])
        reward -= total_torque * 0.001

        return float(reward)

    def close(self):
        try:
            p.disconnect(self.physics_client)
        except Exception:
            pass
