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

def get_new_distant_point(points, low, high, min_distance=0.15, max_tries=10):
    for _ in range(max_tries):
        proposal = np.random.uniform(low, high, points[0].shape)
        for point in points:
            if np.linalg.norm(proposal - point) < min_distance:
                break
        else:
            return proposal
    
    return None

    
class CarEnv(gym.Env):
    def __init__(
        self,
        real_time: bool
    ):
        self.__num_steps = 0
        self.__real_time = real_time

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        p.connect(p.GUI)

        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
        self.__car = p.loadURDF(os.path.join(get_urdf_dir_path(), "car_gripper.urdf"), basePosition=(0,0,0.05))
        self.__object = p.loadURDF(os.path.join(get_urdf_dir_path(), "kp.urdf"), basePosition=(0.5, 0.5, 0.05))
        self.__goal = create_sphere()

        self.__obs_error = np.zeros(4)
        self.reset()

        p.setGravity(0, 0, -10)

        self.__last_log_time = time.time()
        self.__last_actions = []
        self.__last_rewards = []

    def reset(self, seed=None, options=None):
        self.__num_steps = 0

        position, _orientation = p.getBasePositionAndOrientation(self.__car)
        car_position_xy = np.array(position[:2])
        
        object_position_xy = get_new_distant_point(
            points=[car_position_xy],
            low=-1,
            high=1,
        )
        object_position = list(object_position_xy) + [0.05]
        p.resetBasePositionAndOrientation(
            self.__object,
            posObj=object_position,
            ornObj=(0,0,0,1)
        )

        goal_position_xy = get_new_distant_point(
            points=[car_position_xy, object_position_xy],
            low=-1,
            high=1,
        )
        goal_position = list(goal_position_xy) + [0.025]
        p.resetBasePositionAndOrientation(
            self.__goal,
            posObj=goal_position,
            ornObj=(0,0,0,1)
        )

        self.__obs_error = np.random.uniform(-0.01, 0.01, 4)
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
            if self.__real_time:
                time.sleep(1./240.)

        observation = self.__get_observation()

        info = self.__get_info()
        world_T_car = info["world_T_car"]
        world_T_object = info["world_T_object"]
        world_T_goal = info["world_T_goal"]
        car_T_gripper = info["car_T_gripper"]
        
        gripper_T_car = np.linalg.inv(car_T_gripper)
        car_T_world = np.linalg.inv(world_T_car)
        gripper_T_object = gripper_T_car @ car_T_world @ world_T_object
        dist_object = float(np.linalg.norm(gripper_T_object[:2,3]))

        object_T_world = np.linalg.inv(world_T_object)
        object_T_goal = object_T_world @ world_T_goal
        dist_goal = float(np.linalg.norm(object_T_goal[:2,3]))
        
        reward = -(dist_object * 0.5 + dist_goal)/3.0
        done = dist_goal < 0.05

        truncated = False
        if self.__num_steps > 300: # 30s
            p.resetBasePositionAndOrientation(self.__car, list(np.random.uniform(-1, 1, (2))) + [0.05], (0,0,0,1))
            truncated = True
        info = {}

        self.__num_steps += 1

        self.__last_actions.append(action)
        self.__last_rewards.append(reward)
        if time.time() - self.__last_log_time > 1.0:
            print(
                f"mean action: {np.mean(self.__last_actions, axis=0)} "
                f"mean reward: {np.mean(self.__last_rewards):.3f} "
                f"dists: {dist_goal:.3f} {dist_object:.3f} "
                f"last obs: {observation}"
            )
            self.__last_log_time = time.time()
            self.__last_actions = []
            self.__last_rewards = []

        return observation, reward, done, truncated, info


    def __get_info(self):
        position, orientation = p.getBasePositionAndOrientation(self.__car)
        world_T_car = get_homogeneous_transformation_from_pose(position, orientation)

        car_T_gripper = np.eye(4)
        car_T_gripper[:3,3] = [0, -0.0825, 0]

        object_position, _ = p.getBasePositionAndOrientation(self.__object)
        world_T_object = get_homogeneous_transformation_from_pose(object_position, [0,0,0,1])
        
        goal_position, _ = p.getBasePositionAndOrientation(self.__goal)
        world_T_goal = get_homogeneous_transformation_from_pose(goal_position, [0,0,0,1])

        info = {
            "world_T_object": world_T_object,
            "world_T_car": world_T_car,
            "world_T_goal": world_T_goal,
            "car_T_gripper": car_T_gripper,
        }
        return info

    def __get_observation(self):
        info = self.__get_info()
        world_T_car = info["world_T_car"]
        world_T_object = info["world_T_object"]
        world_T_goal = info["world_T_goal"]

        # wheel_velocities = np.array([
        #     p.getJointState(self.__car, 0)[1],
        #     p.getJointState(self.__car, 1)[1],
        # ], dtype=np.float32)

        car_T_world = np.linalg.inv(world_T_car)
        car_T_goal = car_T_world @ world_T_goal
        car_T_object = car_T_world @ world_T_object

        observation =  np.concatenate([
            car_T_object[:2,3],
            car_T_goal[:2,3],
        ], dtype=np.float32)
        observation += self.__obs_error
        
        return observation

if __name__ == "__main__":
    parameters = CarEnv.Parameters(real_time=True)
    env = CarEnv(parameters)
    while 1:
        forces = np.array([0.5, 0.5], dtype=np.float32)
        env.step(forces)