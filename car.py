import os
import time
import pybullet as p
import pybullet_data

p.connect(p.GUI)

print(pybullet_data.getDataPath())
p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))


car = p.loadURDF("car.urdf", basePosition=(0,0,0.05))

p.setGravity(0, 0, -10)

p.setJointMotorControl2(
    car,
    0,
    p.VELOCITY_CONTROL,
    force=0
)
p.setJointMotorControl2(
    car,
    1,
    p.VELOCITY_CONTROL,
    force=0
)
p.setJointMotorControl2(
    car,
    2,
    p.VELOCITY_CONTROL,
    force=0
)

while (1):
    force = 0.005
    p.setJointMotorControl2(
        car,
        0,
        p.TORQUE_CONTROL,
        force=force
    )
    p.setJointMotorControl2(
        car,
        1,
        p.TORQUE_CONTROL,
        force=force
    )

    p.stepSimulation()
    time.sleep(1./240.)
