from __future__ import annotations
from dataclasses import dataclass
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
VELOCITY_SCALE = 2 * np.pi
GOAL_THRESHOLD = 0.15

class RealEnv(gym.Env):
    @dataclass
    class Transforms:
        camera_T_world: Optional[np.ndarray]
        camera_T_robot: Optional[np.ndarray]
        world_T_target: Optional[np.ndarray]

    def __init__(
        self,
        address: str,
    ):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
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

    def reset(self, seed=None, options=None):
        self.__goal = (np.random.random() * 0.75, np.random.random() * 0.75, 0)
        
        info = self.__get_info()
        observation = self.__get_observation(info["transforms"])
        return observation, info

    def step(self, action):
        left_target_velocity_rad, right_taget_velocity_rad = action * VELOCITY_SCALE
        self.__robot.execute_command(DifferentialDriveRobot.Command(
            left_target_velocity_rad,
            right_taget_velocity_rad,
        ))

        info = self.__get_info()
        observation = self.__get_observation(info["transforms"])
        print(self.__goal, observation, action)
        cv2.imshow("vis", info["visualisation"])
        cv2.waitKey(1)

        reward = 0
        done = np.linalg.norm(observation[:3]) < GOAL_THRESHOLD
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
        world_T_target = np.eye(4)
        world_T_target[:3,3] = self.__goal
        transforms = RealEnv.Transforms(
            camera_T_world,
            camera_T_robot,
            world_T_target,
        )

        if camera_T_world is not None:
            camera_T_target = camera_T_world @ world_T_target
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
        if transforms.camera_T_world is None or transforms.camera_T_robot is None:
            return np.zeros(5)

        world_T_robot = np.linalg.inv(transforms.camera_T_world) @ transforms.camera_T_robot
        robot_T_goal = np.linalg.inv(world_T_robot) @ transforms.world_T_target
        delta_position = robot_T_goal[:3,3]
        return  np.concatenate([
            delta_position,
            [0,0],
        ], dtype=np.float32)
