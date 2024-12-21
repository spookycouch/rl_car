from dataclasses import dataclass
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from scipy.spatial.transform import Rotation
from urdf.urdf_utils import get_urdf_dir_path

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
    def __init__(
        self,
        real_time: bool,
    ):
        self.__num_steps = 0
        self.__real_time = real_time

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        p.connect(p.GUI)

        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
        self.__car = p.loadURDF(os.path.join(get_urdf_dir_path(), "car.urdf"), basePosition=(0,0,0.05))

        self.__goal = create_sphere()
        self.__set_new_goal([0,1])

        p.setGravity(0, 0, -10)

    def reset(self, seed=None, options=None):
        self.__num_steps = 0
        new_position_xy = np.random.uniform(0.0, 1.0, (2))
        self.__set_new_goal(new_position_xy)

        observation = self.__get_observation()
        return observation, {}

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
            if self.__real_time:
                time.sleep(1./240.)

        observation = self.__get_observation()

        delta_position = observation[:3]
        reward = self.__get_reward(delta_position)

        done = reward > -0.05
        truncated = False
        if self.__num_steps > 600: # 60s
            p.resetBasePositionAndOrientation(self.__car, (0,0,0), (0,0,0,1))
            truncated = True
        info = {}

        self.__num_steps += 1

        return observation, reward, done, truncated, info

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