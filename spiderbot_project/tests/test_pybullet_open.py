import time
import pybullet as p
import pybullet_data

# Open PyBullet window
client = p.connect(p.GUI)

# Tell PyBullet where built-in example files are
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Gravity
p.setGravity(0, 0, -9.81)

# Load a floor
plane_id = p.loadURDF("plane.urdf")

# Load a sample robot that comes with PyBullet
robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5])

# Run simulation for a few seconds
for _ in range(2000):
    p.stepSimulation()
    time.sleep(1 / 240)

p.disconnect()