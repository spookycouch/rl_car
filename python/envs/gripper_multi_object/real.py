from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Dict, Optional
import cv2
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation
from utils.gui import PointSelector
from utils.oak_d_camera import OakDCamera
from utils.camera import CameraFrame
from utils.sam2.sam2_detector import Sam2Detector
from utils.differential_drive_robot import BluetoothLowEnergyRobot, DifferentialDriveRobot, MockRobot


MARKER_SIDE = 0.09
VELOCITY_SCALE = np.pi
GOAL_THRESHOLD = 0.075

class RealEnv(gym.Env):
    @dataclass
    class Transforms:
        camera_T_robot: Optional[np.ndarray]
        camera_T_object: Optional[np.ndarray]
        camera_T_target: Optional[np.ndarray]

    def __init__(
        self,
        address: str,
    ):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.__camera = OakDCamera(warm_up_frames=30)
        self.__detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters(),
        )

        mock_transform = np.eye(4)
        mock_transform[:3,3] = [0.5, 0.5, 0.0]
        self.__shape_detector = Sam2Detector(
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            "checkpoints/sam2.1_hiera_small.pt",
        )
        self.__robot = BluetoothLowEnergyRobot(address)
        # self.__robot = MockRobot()

        self.__goal = None
        self.__last_log_time = time.time()

    def reset(self, seed=None, options=None):
        self.__goal = None
        info = self.__get_info()
        observation = self.__get_observation(info["transforms"])
        return observation, info

    def step(self, action):
        left_target_velocity_rad, right_taget_velocity_rad = action * VELOCITY_SCALE
        self.__robot.execute_command(DifferentialDriveRobot.Command(
            right_taget_velocity_rad,
            left_target_velocity_rad,
        ))

        info = self.__get_info()
        transforms: RealEnv.Transforms = info["transforms"]
        observation = self.__get_observation(transforms)
        if time.time() - self.__last_log_time > 1.0:
            print(self.__goal, observation, action)
            self.__last_log_time = time.time()
        cv2.imshow("vis", info["visualisation"])
        cv2.waitKey(1)

        reward = 0
        done = False
        if transforms.camera_T_object is not None and transforms.camera_T_target is not None:
            object_T_target = np.linalg.inv(transforms.camera_T_object) @ transforms.camera_T_target

            done = np.linalg.norm(object_T_target[:2,3]) < GOAL_THRESHOLD
        truncated = False
        
        return observation, reward, done, truncated, info


    # see comments in reacher real env on how to make this better.
    def __get_info(self):
        frame: CameraFrame = self.__camera.get_frame()
        image = frame.frame.copy()
        
        # this can be refactored into a detector class
        detections: Dict[int, np.ndarray] = {}
        corners, ids, _rejected_img_points = self.__detector.detectMarkers(image)
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        if corners:
            marker_ids = [marker_id[0] for marker_id in ids] 
            rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                MARKER_SIDE,
                frame.intrinsic_matrix,
                frame.distortion_coeffs,
            )
            for marker_id, rvec, tvec in zip(marker_ids, rvecs, tvecs):
                camera_T_marker = np.eye(4)
                camera_T_marker[:3,:3] = Rotation.from_rotvec(rvec).as_matrix()
                camera_T_marker[:3,3] = tvec
                detections[marker_id] = camera_T_marker
                cv2.drawFrameAxes(image, frame.intrinsic_matrix, frame.distortion_coeffs, rvec, tvec, MARKER_SIDE/2.0)
                if marker_id == 1:
                    for marker_P_gripper in [
                        [0.0, -0.0975, 0, 1], # gripper origin
                        # [-0.065, -0.1875, 0, 1], # gripper left tip
                        # [0.065, -0.1875, 0, 1], # gripper right tip
                    ]:
                        camera_P_gripper = camera_T_marker @ marker_P_gripper
                        cv2.drawFrameAxes(image, frame.intrinsic_matrix, frame.distortion_coeffs, rvec, camera_P_gripper[:3], MARKER_SIDE/4.0)

                
        camera_T_robot = detections.get(1)
        camera_T_target = np.eye(4)

        if self.__goal is None:
            (image_x, image_y) = PointSelector("Click on the target object pose:").detect(image)[0]
            world_z = frame.depth[image_y, image_x]
            image_coords = np.array([image_x, image_y, 1]).T
            camera_P_target = np.linalg.inv(frame.intrinsic_matrix) @ image_coords * world_z
            self.__goal = camera_P_target/1000.0
            
        camera_T_target[:3,3] = self.__goal
        if camera_T_target is not None:
            target_2d, _ = cv2.projectPoints(camera_T_target[:3,3], (0,0,0), (0,0,0), frame.intrinsic_matrix, frame.distortion_coeffs)
            target = int(target_2d[0,0,0]), int(target_2d[0,0,1])
            cv2.circle(image, target, 5, (0,0,255), -1)
        
        camera_T_object = None
        object_centre_point = self.__shape_detector.detect(image)
        if object_centre_point is not None:
            image_y, image_x = [int(index) for index in object_centre_point]
            world_z = frame.depth[image_y, image_x]
            image_coords = np.array([image_x, image_y, 1]).T
            camera_P_object = np.linalg.inv(frame.intrinsic_matrix) @ image_coords * world_z
            camera_T_object = np.eye(4)
            camera_T_object[:3,3] = camera_P_object/1000.0

        transforms = RealEnv.Transforms(
            camera_T_robot,
            camera_T_object,
            camera_T_target,
        )

        # if camera_T_target is not None:
        #     target_2d, _ = cv2.projectPoints(camera_T_target[:3,3], (0,0,0), (0,0,0), frame.intrinsic_matrix, frame.distortion_coeffs)
        #     target = int(target_2d[0,0,0]), int(target_2d[0,0,1])
        #     cv2.circle(image, target, 5, (0,0,255), -1)
        if camera_T_object is not None:
            target_2d, _ = cv2.projectPoints(camera_T_object[:3,3], (0,0,0), (0,0,0), frame.intrinsic_matrix, frame.distortion_coeffs)
            target = int(target_2d[0,0,0]), int(target_2d[0,0,1])
            cv2.circle(image, target, 5, (0,255,0), -1)
        if camera_T_robot is not None:
            target_2d, _ = cv2.projectPoints(camera_T_robot[:3,3], (0,0,0), (0,0,0), frame.intrinsic_matrix, frame.distortion_coeffs)
            target = int(target_2d[0,0,0]), int(target_2d[0,0,1])
            cv2.circle(image, target, 5, (255,255,0), -1)

        # awful
        info = {
            "frame": frame,
            "visualisation": image,
            "transforms": transforms
        }
        return info


    def __get_observation(self, transforms: RealEnv.Transforms):
        if transforms.camera_T_robot is None or transforms.camera_T_object is None:
            return np.zeros(4)

        robot_T_goal = np.linalg.inv(transforms.camera_T_robot) @ transforms.camera_T_target

        robot_T_camera = np.linalg.inv(transforms.camera_T_robot)
        robot_T_object = robot_T_camera @ transforms.camera_T_object
        robot_T_object[1,3] += 0.0675 # the aruco was attached 1.5cm off (should be 0.0825)

        observation =  np.concatenate([
            robot_T_object[:2,3],
            robot_T_goal[:2,3],
        ], dtype=np.float32)
        return observation
