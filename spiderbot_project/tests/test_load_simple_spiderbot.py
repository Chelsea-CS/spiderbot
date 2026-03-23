import os
import time
import pybullet as p
import pybullet_data

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")

urdf_path = os.path.abspath("assets/urdf/simple_spiderbot.urdf")
robot_id = p.loadURDF(urdf_path, [0, 0, 0.3])

for _ in range(2000):
    p.stepSimulation()
    time.sleep(1 / 240)

p.disconnect()