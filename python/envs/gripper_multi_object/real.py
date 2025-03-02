from __future__ import annotations
import array
from dataclasses import dataclass
from enum import unique
import random
import time
from turtle import distance
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
GRIPPER_POSITION = [0.0, -0.085, 0, 1]
ROBOT_P_MARKER = [0.01, 0.0125, 0, 1] # aruco isn't perfectly at 0,0


def draw_robot(
    canvas,
    metres_per_pixel,
    padding_metres = 0,
):
    def array_metres_to_px(array_metres):
        array_px = np.array(array_metres)/metres_per_pixel
        return array_px.astype(int)
    
    offset = (np.array(canvas.shape[:2])/2).astype(int)

    base_radius_px = array_metres_to_px([0.0825])[0]
    
    # approx gripper
    gripper_axes_metres = np.array([0.08, 0.09])
    gripper_centre_metres = np.array(GRIPPER_POSITION[:2]) - np.array([0, gripper_axes_metres[1]])
    gripper_axes_px = array_metres_to_px(gripper_axes_metres)
    gripper_centre_px = array_metres_to_px(gripper_centre_metres) + offset
    gripper_thickness_px = int(np.ceil(0.01/metres_per_pixel))

    mask = np.zeros(canvas.shape[:2])
    cv2.circle(
        mask,
        offset,
        base_radius_px,
        1,
        -1,
    )
    cv2.ellipse(
        mask,
        gripper_centre_px,
        gripper_axes_px,
        0,
        0,
        180,
        1,
        gripper_thickness_px,
    )

    if padding_metres > 0:
        padding_px = int(np.ceil(padding_metres/metres_per_pixel))
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_DILATE, (padding_px,padding_px)),
        )
    return mask


class RealEnv(gym.Env):
    @dataclass
    class Transforms:
        camera_T_robot: Optional[np.ndarray]
        camera_T_object: Optional[np.ndarray]
        camera_T_target: Optional[np.ndarray]
        camera_T_closest_point: Optional[np.ndarray]

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
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "checkpoints/sam2.1_hiera_base_plus.pt",
        )
        self.__robot = BluetoothLowEnergyRobot(address)
        # self.__robot = MockRobot()

        self.__goal = None
        self.__last_log_time = time.time()

    def reset(self, seed=None, options=None):
        # TODO: move robot back a few steps and stop
        # self.__robot.execute_command(DifferentialDriveRobot.Command(
        #     VELOCITY_SCALE,
        #     VELOCITY_SCALE,
        # ))
        # time.sleep(3.0)
        self.__robot.execute_command(DifferentialDriveRobot.Command(
            0.0,
            0.0,
        ))

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


                
        camera_T_robot = None
        camera_T_marker = detections.get(1)
        if camera_T_marker is not None:
            robot_T_marker = np.eye(4)
            robot_T_marker[:,3] = ROBOT_P_MARKER
            camera_T_robot = camera_T_marker @ np.linalg.inv(robot_T_marker)

        camera_T_target = np.eye(4)

        if self.__goal is None:
            (image_x, image_y) = PointSelector("Click on the target object pose:").detect(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )[0]
            world_z = frame.depth[image_y, image_x]
            image_coords = np.array([image_x, image_y, 1]).T
            camera_P_target = np.linalg.inv(frame.intrinsic_matrix) @ image_coords * world_z
            self.__goal = camera_P_target/1000.0
            
        camera_T_target[:3,3] = self.__goal
        camera_T_object = None
        camera_T_closest_point = None
        
        object_centre_point, mask = self.__shape_detector.detect(image)
        # # testing
        # mask = np.zeros_like(mask)
        # mask[100:308, 367:605] = 1.0

        mask_eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)))

        if object_centre_point is not None:
            # image_y, image_x = [int(index) for index in object_centre_point]
            # world_z = frame.depth[image_y, image_x]
            # image_coords = np.array([image_x, image_y, 1]).T
            # camera_P_object = np.linalg.inv(frame.intrinsic_matrix) @ image_coords * world_z
            # camera_T_object = np.eye(4)
            # camera_T_object[:3,3] = camera_P_object/1000.0

            image_points = list(np.argwhere(mask_eroded))
            image_points = random.sample(image_points, k=min(len(image_points), 500))
            image_points = np.array(image_points)[:,:2]
            world_z = frame.depth[image_points[:,0], image_points[:,1]] / 1000.0
            non_zero_indices = world_z != 0
            image_points = image_points[non_zero_indices]
            world_z = world_z[non_zero_indices]
            world_z = np.expand_dims(world_z, axis=-1)

            if camera_T_robot is not None:
                if True:
                    image_points_xyz = np.array(image_points)[:,::-1] # x,y
                    image_points_xyz = np.pad(image_points_xyz, [(0,0),(0,1)], constant_values=1) # add z-dim
                    camera_P_point_cloud_transpose = np.linalg.inv(frame.intrinsic_matrix) @ (image_points_xyz * world_z).T # transform columns of vectors
                    camera_P_point_cloud_transpose = np.pad(camera_P_point_cloud_transpose, [(0,1), (0,0)], constant_values=1)

                    # transform point cloud to gripper frame
                    robot_T_gripper = np.eye(4)
                    robot_T_gripper[:,3] = GRIPPER_POSITION
                    gripper_T_robot = np.linalg.inv(robot_T_gripper)
                    robot_T_camera = np.linalg.inv(camera_T_robot)

                    robot_P_point_cloud_transpose = robot_T_camera @ camera_P_point_cloud_transpose
                    robot_P_point_cloud = robot_P_point_cloud_transpose.T

                    """ortho render and filtering"""
                    if len(robot_P_point_cloud) > 0:
                        metres_per_pixel = 0.01

                        # create canvas
                        canvas_shape_metres = [2.0, 2.0] # ij indexing
                        canvas_shape_pixels = (np.array(canvas_shape_metres)/metres_per_pixel).astype(int)
                        canvas = np.zeros(canvas_shape_pixels)
                        
                        # discretise XY coordinates
                        offset = np.array(canvas_shape_metres + [0,0])/2
                        point_cloud_canvas = robot_P_point_cloud + offset
                        point_cloud_canvas[:,:2] = np.floor(point_cloud_canvas[:,:2]/metres_per_pixel)

                        # remove invalid indices
                        print(
                            point_cloud_canvas[0],
                            point_cloud_canvas[0,0] >= 0,
                            point_cloud_canvas[0,0] < canvas.shape[1],
                            point_cloud_canvas[0,1] >= 0,
                            point_cloud_canvas[0,1] < canvas.shape[0],
                        )
                        valid_indices = (
                            (point_cloud_canvas[:,0] >= 0) &
                            (point_cloud_canvas[:,0] < canvas.shape[1]) &
                            (point_cloud_canvas[:,1] >= 0) &
                            (point_cloud_canvas[:,1] < canvas.shape[0])
                        )
                        point_cloud_canvas = point_cloud_canvas[valid_indices]

                        # sort by Z (works like a Z-buffer)
                        z_indices = np.argsort(point_cloud_canvas[:,2])
                        point_cloud_canvas = point_cloud_canvas[z_indices]

                        x_coords = point_cloud_canvas[:,0].astype(int)
                        y_coords = point_cloud_canvas[:,1].astype(int)
                        canvas[y_coords, x_coords] = point_cloud_canvas[:,2]

                        robot_mask = draw_robot(canvas, metres_per_pixel, 0.07)
                        filtered = canvas * (1-robot_mask)
                        y_coords, x_coords = np.nonzero(filtered)
                        canvas_P_filtered_point_cloud_transposed = np.array([
                            x_coords * metres_per_pixel,
                            y_coords * metres_per_pixel,
                            filtered[filtered != 0],
                            np.ones(len(x_coords))
                        ])
                        canvas_P_point_cloud = canvas_P_filtered_point_cloud_transposed.T
                        robot_P_point_cloud = canvas_P_point_cloud - offset

                        # cv2.imshow("filtered", (filtered != 0).astype(np.float32))
                        # cv2.waitKey(0)

                    if len(robot_P_point_cloud) > 0:
                        robot_P_object = np.mean(robot_P_point_cloud.T, axis=1)
                        camera_T_object = np.eye(4)
                        camera_T_object[:,3] = camera_T_robot @ robot_P_object

                        # cv2.imshow("mask", mask)
                        # cv2.imshow("canvas", (canvas != 0).astype(np.float32))
                        # multiplied = cv2.addWeighted(
                        #     (canvas != 0).astype(np.float32),
                        #     1,
                        #     (mask).astype(np.float32),
                        #     0.5,
                        #     0
                        # )


                    # if len(gripper_P_point_cloud) > 0:
                    #     # voxelise and extract unique XY coordinates
                    #     voxel_scale_m = 0.01
                    #     gripper_P_point_cloud_voxelised = gripper_P_point_cloud.copy()
                    #     gripper_P_point_cloud_voxelised = np.floor(gripper_P_point_cloud_voxelised/voxel_scale_m) * voxel_scale_m
                    #     unique_xy_points = np.unique(gripper_P_point_cloud_voxelised[:,:2], axis=0)
                    #     # print(unique_xy_points)
                    #     print(np.mean(gripper_P_point_cloud.T, axis=1), np.mean(unique_xy_points.T, axis=1))
                    #     # print(gripper_P_point_cloud[0], unique_xy_points[0])
                        

                    #     mean_xy = np.mean(unique_xy_points.T, axis=1)
                    #     p99_z = np.percentile(gripper_P_point_cloud[:,2], 99)
                    #     gripper_P_object = np.concatenate([mean_xy, [p99_z,1]])
                    #     camera_P_object = camera_T_robot @ robot_T_gripper @ gripper_P_object
                    #     camera_T_object = np.eye(4)
                    #     camera_T_object[:3,3] = camera_P_object[:3]
                        
                    #     # get closest point
                    #     distance_array = np.linalg.norm(unique_xy_points, axis=1)
                    #     closest_points = np.argsort(distance_array)
                    #     # print(
                    #     #     f"{distance_array[closest_points[0]]:3f} {distance_array[closest_points[-1]]:.3f}"
                    #     # )
                    #     # camera_P_point_cloud = camera_P_point_cloud_transpose.T # columns of vectors -> array of points
                    #     # camera_P_closest_point = camera_P_point_cloud[closest_points[0]]
                    #     # camera_T_closest_point = np.eye(4)
                    #     # camera_T_closest_point[:3, 3] = camera_P_closest_point[:3]
                    #     gripper_P_closest_point = np.concatenate([unique_xy_points[closest_points[0]], [p99_z,1]])
                    #     camera_P_closest_point = camera_T_robot @ robot_T_gripper @ gripper_P_closest_point
                    #     camera_T_closest_point = np.eye(4)
                    #     camera_T_closest_point[:3, 3] = camera_P_closest_point[:3]


                else:
                    target_2d, _ = cv2.projectPoints(camera_T_robot[:3,3], (0,0,0), (0,0,0), frame.intrinsic_matrix, frame.distortion_coeffs)
                    robot_yx = np.array(target_2d[0,0,1], target_2d[0,0,0])
                    distance_array = np.linalg.norm(image_points - robot_yx, axis=1)
                    closest_points = np.argsort(distance_array)
                    image_y, image_x = image_points[closest_points[0]]
                    world_z = frame.depth[image_y, image_x]/1000.0
                    image_coords = np.array([image_x, image_y, 1]).T
                    camera_P_closest_point = np.linalg.inv(frame.intrinsic_matrix) @ image_coords * world_z
                    camera_T_closest_point = np.eye(4)
                    camera_T_closest_point[:3, 3] = camera_P_closest_point[:3]







        transforms = RealEnv.Transforms(
            camera_T_robot,
            camera_T_object,
            camera_T_target,
            camera_T_closest_point,
        )

        # visualise point cloud in camera image
        if camera_T_object is not None:
            new_mask = np.zeros_like(mask)
            camera_P_point_cloud = (camera_T_robot @ robot_P_point_cloud.T).T
            for point in camera_P_point_cloud:
                target_2d, _ = cv2.projectPoints(point[:3], (0,0,0), (0,0,0), frame.intrinsic_matrix, frame.distortion_coeffs)
                target = int(target_2d[0,0,0]), int(target_2d[0,0,1])
                cv2.circle(new_mask, target, 2, 1, -1)
            image = cv2.addWeighted(
                image,
                1,
                cv2.cvtColor(new_mask * 255, cv2.COLOR_GRAY2BGR),
                0.5,
                0
            )
                

        if camera_T_target is not None:
            target_2d, _ = cv2.projectPoints(camera_T_target[:3,3], (0,0,0), (0,0,0), frame.intrinsic_matrix, frame.distortion_coeffs)
            target = int(target_2d[0,0,0]), int(target_2d[0,0,1])
            cv2.circle(image, target, 5, (0,0,255), -1)
        if camera_T_object is not None:
            target_2d, _ = cv2.projectPoints(camera_T_object[:3,3], (0,0,0), (0,0,0), frame.intrinsic_matrix, frame.distortion_coeffs)
            target = int(target_2d[0,0,0]), int(target_2d[0,0,1])
            cv2.circle(image, target, 5, (0,255,0), -1)
        # if camera_T_closest_point is not None:
        #     target_2d, _ = cv2.projectPoints(camera_T_closest_point[:3,3], (0,0,0), (0,0,0), frame.intrinsic_matrix, frame.distortion_coeffs)
        #     target = int(target_2d[0,0,0]), int(target_2d[0,0,1])
        #     cv2.circle(image, target, 5, (0,255,0), -1)
        if camera_T_robot is not None:
            rvec = Rotation.from_matrix(camera_T_robot[:3,:3]).as_rotvec()
            camera_P_gripper = camera_T_robot @ GRIPPER_POSITION
            cv2.drawFrameAxes(image, frame.intrinsic_matrix, frame.distortion_coeffs, rvec, camera_P_gripper[:3], MARKER_SIDE/4.0)
        # if camera_T_robot is not None:
        #     # visualise the depth error
        #     target_2d, _ = cv2.projectPoints(camera_T_robot[:3,3], (0,0,0), (0,0,0), frame.intrinsic_matrix, frame.distortion_coeffs)
        #     image_x, image_y = int(target_2d[0,0,0]), int(target_2d[0,0,1])
        #     world_z = frame.depth[image_y, image_x]
        #     image_coords = np.array([image_x, image_y, 1]).T
        #     camera_P_robot = np.linalg.inv(frame.intrinsic_matrix) @ image_coords * world_z
        #     cv2.drawFrameAxes(image, frame.intrinsic_matrix, frame.distortion_coeffs, rvec, camera_P_robot, MARKER_SIDE/2.0)


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
        robot_T_object[1,3] += 0.0325 # the aruco was attached ~2cm off (should be 0.0825)

        # robot_T_closest_point = robot_T_camera @ transforms.camera_T_closest_point
        # robot_T_closest_point[1,3] += 0.0325 # the aruco was attached ~2cm off (should be 0.0825)

        observation =  np.concatenate([
            robot_T_object[:2,3],
            robot_T_goal[:2,3],
            # robot_T_closest_point[:2,3],
        ], dtype=np.float32)
        return observation
