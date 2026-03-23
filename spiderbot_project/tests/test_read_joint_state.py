import os
import time
import math
import pybullet as p
import pybullet_data

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")

urdf_path = os.path.abspath("assets/urdf/simple_spiderbot.urdf")
robot_id = p.loadURDF(urdf_path, [0, 0, 0.3])

joint_index = 0

for step in range(2000):
    t = step / 240.0
    target_angle = 0.7 * math.sin(2 * math.pi * 0.5 * t)

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=joint_index,
        controlMode=p.POSITION_CONTROL,
        targetPosition=target_angle,
        force=5
    )

    p.stepSimulation()

    if step % 120 == 0:
        joint_state = p.getJointState(robot_id, joint_index)
        joint_angle = joint_state[0]
        joint_velocity = joint_state[1]
        print(f"step={step}, angle={joint_angle:.3f}, velocity={joint_velocity:.3f}")

    time.sleep(1 / 240)

p.disconnect()