import pybullet as p
import pybullet_data
import time
import math
import os

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(
    cameraDistance=2.0, cameraYaw=0,
    cameraPitch=-20, cameraTargetPosition=[0, 0, 0.3]
)

robot = p.loadURDF("my_robot.urdf", [0, 0, 0.45])

joint_map = {}
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    name = info[1].decode()
    joint_map[name] = i

joint_names = [
    "fl_hip_joint", "fl_knee_joint",
    "fr_hip_joint", "fr_knee_joint",
    "rl_hip_joint", "rl_knee_joint",
    "rr_hip_joint", "rr_knee_joint",
]

print("Watch the robot — it should move in the +X direction (toward you)")
print("If it moves away, the sine wave is backward")

t = 0
start_x = None
for step in range(1000):
    t += 0.04
    sine = [
         0.35 * math.sin(t),
        -0.3  + 0.25 * math.sin(t + math.pi/2),
        -0.35 * math.sin(t),
        -0.3  + 0.25 * math.sin(t + math.pi/2),
        -0.35 * math.sin(t),
        -0.3  + 0.25 * math.sin(t + math.pi/2),
         0.35 * math.sin(t),
        -0.3  + 0.25 * math.sin(t + math.pi/2),
    ]
    for i, name in enumerate(joint_names):
        if name in joint_map:
            p.setJointMotorControl2(robot, joint_map[name],
                controlMode=p.POSITION_CONTROL,
                targetPosition=sine[i], force=30.0)
    p.stepSimulation()
    time.sleep(1/60)

    pos, _ = p.getBasePositionAndOrientation(robot)
    if start_x is None:
        start_x = pos[0]

pos, _ = p.getBasePositionAndOrientation(robot)
moved = pos[0] - start_x
print(f"\nMoved {moved:.3f}m in X direction")
if moved > 0.05:
    print("FORWARD — sine wave is correct")
elif moved < -0.05:
    print("BACKWARD — need to flip sine signs")
else:
    print("NOT MOVING — sine wave has no effect on this robot geometry")

p.disconnect()