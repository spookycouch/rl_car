from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Dict, Optional
import cv2
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation
from utils.camera import OpenCVCameraWrapper, CameraFrame
from utils.differential_drive_robot import BluetoothLowEnergyRobot, DifferentialDriveRobot


MARKER_SIDE = 0.09
CAMERA_MATRIX = np.array([
    [616.90312593,   0.        , 342.70431048,],
    [  0.        , 618.48254063, 272.13499089,],
    [  0.        ,   0.        ,   1.        ,],
])
DIST_COEFFS = np.array([[-0.00953016, -0.22793283,  0.00619988, -0.00298346,  0.2422911]])
VELOCITY_SCALE = np.pi
# VELOCITY_SCALE = 2 * np.pi
# VELOCITY_SCALE = 0
GOAL_THRESHOLD = 0.075

class RealEnv(gym.Env):
    @dataclass
    class Transforms:
        camera_T_world: Optional[np.ndarray]
        camera_T_robot: Optional[np.ndarray]
        camera_T_object: Optional[np.ndarray]
        world_T_target: Optional[np.ndarray]

    def __init__(
        self,
        address: str,
    ):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # use hydra for this
        self.__camera = OpenCVCameraWrapper(
            video_source=2,
            intrinsic_matrix=CAMERA_MATRIX,
            distortion_coeffs=DIST_COEFFS,
        )
        # also parameterisable
        self.__detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters(),
        )
        self.__robot = BluetoothLowEnergyRobot(address)

        self.__goal = np.zeros(3)
        self.__camera_T_world = None

        self.__last_log_time = time.time()

    def reset(self, seed=None, options=None):
        self.__goal = (np.random.random() * 0.75, np.random.uniform(0.25,0.75), 0)
        
        info = self.__get_info()
        observation = self.__get_observation(info["transforms"])
        return observation, info

    def step(self, action):
        left_target_velocity_rad, right_taget_velocity_rad = action * VELOCITY_SCALE
        print(action)
        print(left_target_velocity_rad, right_taget_velocity_rad)
        self.__robot.execute_command(DifferentialDriveRobot.Command(
            right_taget_velocity_rad,
            left_target_velocity_rad,
        ))

        info = self.__get_info()
        transforms: RealEnv.Transforms = info["transforms"]
        observation = self.__get_observation(transforms)
        # if time.time() - self.__last_log_time > 1.0:
        #     print(self.__goal, observation, action)
        #     self.__last_log_time = time.time()
        cv2.imshow("vis", info["visualisation"])
        cv2.waitKey(1)

        reward = 0
        done = False
        if transforms.camera_T_object is not None and transforms.world_T_target is not None and transforms.camera_T_world is not None:
            object_T_world = np.linalg.inv(transforms.camera_T_object) @ transforms.camera_T_world
            object_T_target = object_T_world @ transforms.world_T_target

            done = np.linalg.norm(object_T_target[:2,3]) < GOAL_THRESHOLD
        truncated = False
        
        return observation, reward, done, truncated, info


    # not sure I like this as info
    # maybe detect_objects()? run_perception()?
    # it can return the detections + image, that's all.
    # another method for get transforms.
    # we can get rid of this then, and just compute everything in step().
    def __get_info(self):
        frame: CameraFrame = self.__camera.get_frame()
        image = frame.frame.copy()
        
        # this can be refactored into a detector class
        detections: Dict[int, np.ndarray] = {}
        corners, ids, _rejected_img_points = self.__detector.detectMarkers(image)
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        if corners:
            marker_ids = [marker_id[0] for marker_id in ids] 
            rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIDE, CAMERA_MATRIX, DIST_COEFFS)
            for marker_id, rvec, tvec in zip(marker_ids, rvecs, tvecs):
                marker_transform = np.eye(4)
                marker_transform[:3,:3] = Rotation.from_rotvec(rvec).as_matrix()
                marker_transform[:3,3] = tvec
                detections[marker_id] = marker_transform
                cv2.drawFrameAxes(image, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, MARKER_SIDE/2.0)
        
        # use a filter for this, later.
        # likely this could also just be a dataclass.
        # or a tracker on the 6D pose (remove the 4x4 code above).
        camera_T_world = detections.get(0)
        camera_T_robot = detections.get(1)
        camera_T_object = detections.get(2)
        world_T_target = np.eye(4)
        world_T_target[:3,3] = self.__goal

        if self.__camera_T_world is None:
            self.__camera_T_world = camera_T_world
        
        transforms = RealEnv.Transforms(
            self.__camera_T_world,
            camera_T_robot,
            camera_T_object,
            world_T_target,
        )

        if self.__camera_T_world is not None:
            camera_T_target = self.__camera_T_world @ world_T_target
            target_2d, _ = cv2.projectPoints(camera_T_target[:3,3], (0,0,0), (0,0,0), CAMERA_MATRIX, DIST_COEFFS)
            target = int(target_2d[0,0,0]), int(target_2d[0,0,1])
            cv2.circle(image, target, 5, (0,0,255), -1)
        
        # awful
        info = {
            "frame": frame,
            "visualisation": image,
            "transforms": transforms
        }
        return info

    def __run_perception(self):
        pass

    def __update_trackers(self):
        # camera_T_world
        # camera_T_robot
        pass

    def __get_transforms(self):
        # robot_T_goal
        # camera_T_target
        pass

    def __draw_visualisation(
        self,
        frame,
        detections,
        transforms,
    ):
        # drawframeaxes (moved here, using trackers)
        # draw circle
        pass

    def __get_observation(self, transforms: RealEnv.Transforms):
        if transforms.camera_T_world is None or transforms.camera_T_robot is None or transforms.camera_T_object is None:
            return np.zeros(4)

        world_T_robot = np.linalg.inv(transforms.camera_T_world) @ transforms.camera_T_robot
        robot_T_goal = np.linalg.inv(world_T_robot) @ transforms.world_T_target
        robot_T_goal[1,3] += 0.01

        robot_T_camera = np.linalg.inv(transforms.camera_T_robot)
        robot_T_object = robot_T_camera @ transforms.camera_T_object
        robot_T_object[1,3] += 0.055 # 3cm error

        observation =  np.concatenate([
            robot_T_object[:2,3],
            robot_T_goal[:2,3],
        ], dtype=np.float32)
        return observation
