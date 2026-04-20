import pybullet as p
import pybullet_data
import time

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(
    cameraDistance=0.55, cameraYaw=35,
    cameraPitch=-25, cameraTargetPosition=[0, 0, 0.05]
)

# Body center at 0.18m — upper leg 0.07/2=0.035 below joint,
# knee at -0.035, lower leg 0.09/2=0.045 below knee = total ~0.16m leg reach
robot = p.loadURDF("my_robot.urdf", [0, 0, 0.18])

joint_map = {}
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    joint_map[info[1].decode()] = i
    print(f"Joint {i}: {info[1].decode()}")

# All joints at 0 = legs hang straight down
for name in joint_map:
    p.resetJointState(robot, joint_map[name], 0.0)

for _ in range(300):
    p.stepSimulation()

pos, _ = p.getBasePositionAndOrientation(robot)
print(f"\nBody height after settling: {pos[2]:.4f}m")
print("If ~0.16-0.18 and feet on ground = correct")
print("Watch the window — legs should point straight DOWN from body corners")

while True:
    p.stepSimulation()
    time.sleep(1/60)