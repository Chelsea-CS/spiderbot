import pybullet as p
import pybullet_data
import time
import math
import os

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(
    cameraDistance=0.6, cameraYaw=45,
    cameraPitch=-20, cameraTargetPosition=[0, 0, 0.08]
)

urdf_path = os.path.join(os.path.dirname(__file__), "my_robot.urdf")
robot = p.loadURDF(urdf_path, [0, 0, 0.15])

for i in range(-1, p.getNumJoints(robot)):
    p.changeDynamics(robot, i, lateralFriction=1.5,
                     linearDamping=0.3, angularDamping=0.3)

joint_map = {}
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    joint_map[info[1].decode()] = i

joint_names = list(joint_map.keys())
print("Joints found:", joint_names)
print("Manually driving all joints with sine wave...")
print("You MUST see legs moving. If not, URDF is broken.")

t = 0
while True:
    t += 0.01

    # Drive ALL joints with a slow sine wave — guaranteed movement
    for name, idx in joint_map.items():
        if "hip" in name:
            target = 0.3 * math.sin(t)
        else:
            target = -0.4 + 0.3 * math.sin(t + 1.0)

        p.setJointMotorControl2(
            robot, idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=20.0,
            maxVelocity=3.0
        )

    p.stepSimulation()
    time.sleep(1.0 / 60.0)