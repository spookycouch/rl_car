# import asyncio
# from bleak import BleakScanner

# async def main():
#     devices = await BleakScanner.discover()
#     for d in devices:
#         print(d)

# asyncio.run(main())

import argparse
import asyncio
import math
import time
from typing import Tuple
from bleak import BleakClient
from stable_baselines3 import PPO

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pose_tracker import OpenCVPose, SimplePoseTracker

MARKER_SIDE = 0.09
MARKER_HALF_SIDE = MARKER_SIDE/2.0
CAMERA_MATRIX = np.array([
    [616.90312593,   0.        , 342.70431048,],
    [  0.        , 618.48254063, 272.13499089,],
    [  0.        ,   0.        ,   1.        ,],
])
DIST_COEFFS = np.array([[-0.00953016, -0.22793283,  0.00619988, -0.00298346,  0.2422911]])

COMMAND_CHARACTERISTIC_UUID = "eee4879a-80a8-4d33-af74-b05826ee658f"

VELOCITY_SCALE = 2 * np.pi

WORLD_ID = 0
CAR_ID = 1

markers = {}
markers[WORLD_ID] = SimplePoseTracker(0.9)
markers[CAR_ID] = SimplePoseTracker(0.0)

cam = cv2.VideoCapture(2)



def get_htm_from_opencv_pose(pose):
    htm = np.eye(4)
    htm[:3,:3] = Rotation.from_rotvec(pose.rvec).as_matrix()
    htm[:3,3] = pose.tvec
    return htm

def get_new_goal() -> Tuple[float, float, float]:
    return np.random.random() * 0.75, np.random.random() * 0.75, 0

async def main(address):
    MODEL_NAME = "ppo_car_v2"
    model = PPO.load(MODEL_NAME)

    detector_params = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    for i in range(3):
        image = cv2.aruco.generateImageMarker(dictionary, i, 1000)
        cv2.imwrite(f"marker_{i}.png", image)

    camera_T_world = np.eye(4)

    world_T_target = np.eye(4)
    world_T_target[:3,3] = get_new_goal()

    # time_last = time.time()
    async with BleakClient(address) as client:
        # may need to implement AcquireWrite on the device:
        # https://github.com/hbldh/bleak/blob/e01e2640994b99066552b6161f84799712c396fa/bleak/backends/bluezdbus/client.py#L573
        # print(client.mtu_size)
        # if client._backend.__class__.__name__ == "BleakClientBlueZDBus":
        #     await client._backend._acquire_mtu()
        # print(client.mtu_size)
        
        while 1:
            time_start = time.time()
            _, frame = cam.read()

            corners, ids, rejected_img_points = detector.detectMarkers(frame)
            message_out = "0 0"

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            if corners:
                marker_ids = [marker_id[0] for marker_id in ids] 
                rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIDE, CAMERA_MATRIX, DIST_COEFFS)
                for marker_id, rvec, tvec in zip(marker_ids, rvecs, tvecs):
                    if marker_id not in markers:
                        continue

                    pose_obs = OpenCVPose(rvec, tvec)
                    pose_pred: OpenCVPose = markers[marker_id].update(pose_obs)
                    cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, pose_pred.rvec, pose_pred.tvec, 0.045)

                
                world_pose = markers[WORLD_ID].get_pose()
                car_pose = markers[CAR_ID].get_pose()

                if world_pose is not None:
                    camera_T_world = get_htm_from_opencv_pose(world_pose)
                    camera_T_target = camera_T_world @ world_T_target
                    target_2d, _ = cv2.projectPoints(camera_T_target[:3,3], (0,0,0), (0,0,0), CAMERA_MATRIX, DIST_COEFFS)
                    target = int(target_2d[0,0,0]), int(target_2d[0,0,1])
                    cv2.circle(frame, target, 5, (0,0,255), -1)
                        
                    if car_pose is not None:
                        camera_T_car = get_htm_from_opencv_pose(car_pose)
                        world_T_car = np.linalg.inv(camera_T_world) @ camera_T_car
                        
                        car_T_goal = np.linalg.inv(world_T_car) @ world_T_target
                        delta_position = car_T_goal[:3,3]
                        observation =  np.concatenate([
                            delta_position,
                            [0,0],
                        ], dtype=np.float32)

                        action, _states = model.predict(observation)

                        print(delta_position, action)
                        print(np.linalg.norm(delta_position))
                        if(np.linalg.norm(delta_position)) < 0.15:
                            world_T_target[:3,3] = get_new_goal()
                        action = np.clip(action, -1.0, 1.0)
                        left_target_velocity_rad, right_taget_velocity_rad = action * VELOCITY_SCALE

                        message_out = f"{left_target_velocity_rad:.3f} {right_taget_velocity_rad:.3f}"

            time_model = time.time()

            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            await client.write_gatt_char(COMMAND_CHARACTERISTIC_UUID, bytes(message_out, encoding="utf8"))
            time_end = time.time()
            print(
                f"model_freq: {1/(time_model - time_start)} "
                f"send_freq: {1/(time_end - time_model)} "
                f"total_freq: {1/(time_end - time_start)} "
            )
            # print(1/(time.time() - time_last))
            # time_last = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("address", help="MAC address of the BLE server.")
    args = parser.parse_args()
    asyncio.run(main(args.address))
