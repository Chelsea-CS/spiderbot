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
        self.step_count = 0
        self.total_steps = 0
        self.robot = None
        self.physics_client = None
        self.world_built = False
        self.joint_map = {}
        self.prev_forward_vel = 0.0

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
        )

    def _build_world(self):
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=0,
                cameraPitch=-20,
                cameraTargetPosition=[0, 0, 0.25]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0/240.0,
            numSolverIterations=50,
            numSubSteps=1
        )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = p.loadURDF("plane.urdf")
        p.changeDynamics(plane, -1, lateralFriction=1.5, restitution=0.0)

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

        for name in self.joint_names:
            if name in self.joint_map:
                p.resetJointState(self.robot, self.joint_map[name], 0.0, 0.0)

        for _ in range(200):
            p.stepSimulation()

        self.prev_forward_vel = 0.0

    def reset(self, seed=None):
        if not self.world_built:
            self._build_world()
        self._place_robot()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        self.total_steps += 1

        t = self.total_steps * 0.05

        # Keep sine wave dominant for much longer — 600k steps
        # This forces the trot pattern to be deeply learned before RL takes over
        explore_weight = max(0.0, 1.0 - self.total_steps / 600000.0)

        # Proper trot: FL+RR together, FR+RL together, offset by pi
        # This is the diagonal trot pattern a real dog uses
        sine_actions = [
             0.3 * math.sin(t),                      # fl_hip
            -0.15 + 0.15 * math.sin(t),              # fl_knee
            -0.3 * math.sin(t + math.pi),            # fr_hip  (opposite phase)
            -0.15 + 0.15 * math.sin(t + math.pi),   # fr_knee
            -0.3 * math.sin(t + math.pi),            # rl_hip  (opposite phase)
            -0.15 + 0.15 * math.sin(t + math.pi),   # rl_knee
             0.3 * math.sin(t),                      # rr_hip  (same as fl)
            -0.15 + 0.15 * math.sin(t),              # rr_knee
        ]

        joint_limits = [0.5, 0.7, 0.5, 0.7, 0.5, 0.7, 0.5, 0.7]

        for i, name in enumerate(self.joint_names):
            if name not in self.joint_map:
                continue
            idx = self.joint_map[name]
            rl = float(action[i]) * joint_limits[i]
            blended = explore_weight * sine_actions[i] + (1.0 - explore_weight) * rl
            blended = float(np.clip(blended, -joint_limits[i], joint_limits[i]))
            p.setJointMotorControl2(
                self.robot, idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=blended,
                force=25.0,
                maxVelocity=3.0,
                positionGain=0.3,
                velocityGain=0.1
            )

        p.stepSimulation()

        if self.render_mode:
            try:
                pos, _ = p.getBasePositionAndOrientation(self.robot)
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.5,
                    cameraYaw=0,
                    cameraPitch=-20,
                    cameraTargetPosition=[pos[0], pos[1], 0.25]
                )
            except Exception:
                pass
            time.sleep(1.0 / 120.0)

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self.step_count >= 2000
        return obs, reward, done, False, {}

    def _get_obs(self):
        angles, vels = [], []
        for name in self.joint_names:
            if name in self.joint_map:
                s = p.getJointState(self.robot, self.joint_map[name])
                angles.append(s[0])
                vels.append(s[1])
            else:
                angles.append(0.0)
                vels.append(0.0)

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        euler = p.getEulerFromQuaternion(orn)
        lv, _ = p.getBaseVelocity(self.robot)

        return np.array(
            angles + vels + list(euler) + list(lv[:3]) + [pos[2]],
            dtype=np.float32
        )

    def _compute_reward(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        lv, _ = p.getBaseVelocity(self.robot)

        forward_vel = lv[0]
        height = pos[2]

        # Simple: reward forward movement only
        reward = forward_vel * 4.0

        # Only penalize falling
        if height < 0.20:
            reward -= 1.0

        return float(reward)
    def close(self):
        try:
            p.disconnect(self.physics_client)
        except Exception:
            pass