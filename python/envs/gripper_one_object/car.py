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
    @dataclass
    class Parameters:
        real_time: bool

    def __init__(self, parameters: Parameters):
        self.__num_steps = 0
        self.__parameters = parameters

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        p.connect(p.GUI)

        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
        self.__car = p.loadURDF(os.path.join(get_urdf_dir_path(), "car_gripper.urdf"), basePosition=(0,0,0.05))
        self.__goal = p.loadURDF(os.path.join(get_urdf_dir_path(), "pringles.urdf"), basePosition=(0.5, 0.5, 0.05))
        # self.__enable_force_control()

        self.__set_new_goal([0,1])

        p.setGravity(0, 0, -10)

    def reset(self, seed=None, options=None):
        self.__num_steps = 0

        position, orientation = p.getBasePositionAndOrientation(self.__car)
        car_position_xy = np.array(position[:2])
        new_position_xy = car_position_xy.copy()
        while np.linalg.norm(new_position_xy - car_position_xy) < 0.15:
            new_position_xy = np.random.uniform(0, 1, (2))
        self.__set_new_goal(new_position_xy)

        observation = self.__get_observation()
        return observation, {}

    def step(self, action):
        target_velocities = action*2*np.pi

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

        info = self.__get_info()
        world_T_car = info["world_T_car"]
        world_T_goal = info["world_T_goal"]
        car_T_gripper = info["car_T_gripper"]
        gripper_T_car = np.linalg.inv(car_T_gripper)
        car_T_world = np.linalg.inv(world_T_car)
        # car_T_goal = car_T_world @ world_T_goal
        gripper_T_goal = gripper_T_car @ car_T_world @ world_T_goal

        dist_target = float(np.linalg.norm(gripper_T_goal[:2,3]))
        dist_endgoal = float(np.linalg.norm(world_T_goal[:2,3]))
        reward = -(dist_target * 0.5 + dist_endgoal)

        done = dist_endgoal < 0.05


        truncated = False
        if self.__num_steps > 600: # 60s
            p.resetBasePositionAndOrientation(self.__car, list(np.random.uniform(0, 1, (2))) + [0.05], (0,0,0,1))
            truncated = True
        info = {}

        self.__num_steps += 1

        return observation, reward, done, truncated, info

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
        new_position = list(position_xy) + [0.05]
        p.resetBasePositionAndOrientation(
            self.__goal,
            posObj=new_position,
            ornObj=(0,0,0,1)
        )

    def __get_info(self):
        position, orientation = p.getBasePositionAndOrientation(self.__car)
        world_T_car = get_homogeneous_transformation_from_pose(position, orientation)

        car_T_gripper = np.eye(4)
        car_T_gripper[:3,3] = [0, -0.0825, 0]

        goal_position, _ = p.getBasePositionAndOrientation(self.__goal)
        world_T_goal = get_homogeneous_transformation_from_pose(goal_position, [0,0,0,1])

        info = {
            "world_T_goal": world_T_goal,
            "world_T_car": world_T_car,
            "car_T_gripper": car_T_gripper,
        }
        return info

    def __get_observation(self):
        position, orientation = p.getBasePositionAndOrientation(self.__car)
        world_T_car = get_homogeneous_transformation_from_pose(position, orientation)
        car_rotation_z_rad = Rotation.from_quat(orientation).as_euler("xyz")[2]/(2*np.pi)

        goal_position, _ = p.getBasePositionAndOrientation(self.__goal)
        world_T_goal = get_homogeneous_transformation_from_pose(goal_position, [0,0,0,1])

        wheel_velocities = np.array([
            p.getJointState(self.__car, 0)[1],
            p.getJointState(self.__car, 1)[1],
        ], dtype=np.float32)
        # wheel_velocities = wheel_velocities * 0.0

        # car_T_world @ world_T_goal = car_T_goal
        car_T_world = np.linalg.inv(world_T_car)
        car_T_goal = car_T_world @ world_T_goal
        # angle_car_to_goal_rad = np.arctan2(car_T_goal[1,3], car_T_goal[0,3])
        delta_position = car_T_goal[:3,3]

        observation =  np.concatenate([
            car_T_goal[:3,3],
            car_T_world[:3,3],
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