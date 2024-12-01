from dataclasses import dataclass
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import gym
from scipy.spatial.transform import Rotation

def get_homogeneous_transformation_from_pose(position, orientation):
    transform = np.eye(4, dtype=np.float32)
    transform [:3, 3] = position

    rotation_matrix = Rotation.from_quat(orientation).as_matrix()
    transform[:3,:3] = rotation_matrix

    return transform

def create_sphere():
    visual_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius = 0.025,
        rgbaColor=(1,0,0,1),
    )
    multibody_id = p.createMultiBody(baseVisualShapeIndex=visual_id)
    return multibody_id

    
class CarEnv(gym.Env):
    @dataclass
    class Parameters:
        real_time: bool

    def __init__(self, parameters: Parameters):
        self.__parameters = parameters

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        p.connect(p.GUI)

        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
        self.__car = p.loadURDF("car.urdf", basePosition=(0,0,0.05))
        # self.__enable_force_control()

        self.__goal = create_sphere()
        self.__set_new_goal([0,1])

        p.setGravity(0, 0, -10)

    def reset(self, seed=None, options=None):
        new_position_xy = np.random.uniform(0.0, 1.0, (2))
        self.__set_new_goal(new_position_xy)

        observation = self.__get_observation()
        return observation

    def step(self, action):
        target_velocities = action*4*np.pi

        for _ in range(24):
            p.setJointMotorControl2(
                self.__car,
                0,
                p.VELOCITY_CONTROL,
                targetVelocity=target_velocities[0]
            )
            p.setJointMotorControl2(
                self.__car,
                1,
                p.VELOCITY_CONTROL,
                targetVelocity=target_velocities[1]
            )

            p.stepSimulation()
            if self.__parameters.real_time:
                time.sleep(1./240.)

        observation = self.__get_observation()

        delta_position = observation[:3]
        reward = self.__get_reward(delta_position)

        done = reward > -0.05
        truncated = False
        info = {}

        return observation, reward, done, info

    def __enable_force_control(self):
        p.setJointMotorControl2(
            self.__car,
            0,
            p.VELOCITY_CONTROL,
            force=0
        )
        p.setJointMotorControl2(
            self.__car,
            1,
            p.VELOCITY_CONTROL,
            force=0
        )
        p.setJointMotorControl2(
            self.__car,
            2,
            p.VELOCITY_CONTROL,
            force=0
        )
    
    def __set_new_goal(self, position_xy):
        new_position = list(position_xy) + [0.025]
        p.resetBasePositionAndOrientation(
            self.__goal,
            posObj=new_position,
            ornObj=(0,0,0,1)
        )

    def __get_observation(self):
        position, orientation = p.getBasePositionAndOrientation(self.__car)
        world_T_car = get_homogeneous_transformation_from_pose(position, orientation)

        goal_position, _ = p.getBasePositionAndOrientation(self.__goal)
        world_T_goal = get_homogeneous_transformation_from_pose(goal_position, [0,0,0,1])

        wheel_velocities = np.array([
            p.getJointState(self.__car, 0)[1],
            p.getJointState(self.__car, 1)[1],
        ], dtype=np.float32)
        wheel_velocities = wheel_velocities * 0.0

        # car_T_world @ world_T_goal = car_T_goal
        car_T_goal = np.linalg.inv(world_T_car) @ world_T_goal
        delta_position = car_T_goal[:3,3]

        observation =  np.concatenate([
            delta_position,
            wheel_velocities,
        ], dtype=np.float32)
        return observation

    def __get_reward(self, delta_position):
        reward = -np.linalg.norm(delta_position)
        return float(reward)


if __name__ == "__main__":
    parameters = CarEnv.Parameters(real_time=True)
    env = CarEnv(parameters)
    while 1:
        forces = np.array([0.5, 0.5], dtype=np.float32)
        env.step(forces)