from enum import Enum, auto
import os
import time
from typing import Dict, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from rl_car.utils.differential_drive_robot import BluetoothLowEnergyRobot, DifferentialDriveRobot, MockRobot
from rl_car.utils.oak_d_camera import OakDRGB, CameraFrame

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

from gamepad import Gamepad, GamepadConfig, GamepadInput

ROBOT_MAC_ADDRESS_ENV = "ROBOT_MAC_ADDRESS"
VELOCITY_SCALE = -3 * 3.142

DATASET_TASK = "Push the object into the green target area."
DATASET_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (3,),
        "names": {
            "axes": ["x", "y", "theta"],
        },
    },
    "observation.image": {
        "dtype": "image",
        "shape": (3, 360, 640),
        "names": [
            "channels",
            "height",
            "width",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": {
            "axes": ["left", "right"],
        },
    },
    "extras.teleop_input": {
        "dtype": "float32",
        "shape": (3,),
        "names": {
            "axes": ["steer", "accelerate", "reverse"],
        },
    },
    "extras.raw_image": {
        "dtype": "image",
        "shape": (3, 360, 640),
        "names": [
            "channels",
            "height",
            "width",
        ],
    },
}
DATASET_REPO_ID = "differential_drive_push"
DATASET_ROBOT_TYPE = "differential drive"

class RecordingStatus(Enum):
    NOT_STARTED = "Recording not started"
    RECORDING = "REC"
    EPISODE_SAVED = "Last episode: SAVED"
    EPISODE_DISCARDED = "Last episode: DISCARDED"



def detect_aruco_pose(
    frame: CameraFrame,
    debug=False,
) -> Tuple[float, float, float]:
    MARKER_SIDE = 0.09
    ROBOT_ID = 1
    
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters(),
    )

    corners, ids, _rejected_img_points = detector.detectMarkers(frame.frame)
    image_aruco = frame.frame.copy()

    camera_R_world = Rotation.from_euler("x", 180, degrees=True)

    x, y, theta = 0, 0, 0

    if corners:
        marker_ids = [marker_id[0] for marker_id in ids] 
        rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIDE, frame.intrinsic_matrix, frame.distortion_coeffs)
        for marker_id, rvec, tvec in zip(marker_ids, rvecs, tvecs):
            if marker_id == ROBOT_ID:
                x, y, _ = tvec[0]
                camera_R_robot = Rotation.from_rotvec(rvec[0])
                world_R_robot = camera_R_world.inv() * camera_R_robot

                quat_theta = world_R_robot.as_quat()
                quat_theta[0] = 0
                quat_theta[1] = 0
                quat_theta = quat_theta/np.linalg.norm(quat_theta)
                theta = Rotation.from_quat(quat_theta).magnitude()

                cv2.drawFrameAxes(image_aruco, frame.intrinsic_matrix, frame.distortion_coeffs, rvec, tvec, MARKER_SIDE/2.0, thickness=2)
    
    if debug:
        cv2.imshow("aruco", image_aruco)
        cv2.waitKey(1)

    return x, y, theta


def draw_goal_position(
    frame,
    bbox,
):
    """
    Draw goal position onto image.
    
    param:  bbox    x,y,w,h
    """
    mask = np.zeros_like(frame)
    cv2.rectangle(
        mask,
        (bbox[0], bbox[1]),
        (bbox[0] + bbox[2], bbox[1] + bbox[3]),
        (0,255,0),
        -1,
    )
    return cv2.addWeighted(
        frame,
        1,
        mask,
        0.25,
        0,
    )    


def draw_pilot_view(
    frame,
    robot_pose: Tuple[float, float, float],
    recording_status: RecordingStatus,
    total_episodes: int,
    reset_points: Dict[str, Tuple[int, int]],
):
    pilot_image = frame.copy()

    for name, (x, y) in reset_points.items():
        cv2.putText(
            pilot_image,
            name,
            (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,255),
            2,
        )

    colour_recording_status = (255,255,255)
    if recording_status == RecordingStatus.RECORDING:
        colour_recording_status = (0,0,255)
    elif recording_status == RecordingStatus.EPISODE_SAVED:
        colour_recording_status = (0,255,0)
    elif recording_status == RecordingStatus.EPISODE_DISCARDED:
        colour_recording_status = (255,0,0)
        
    cv2.putText(
        pilot_image,
        recording_status.value,
        (10,20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        colour_recording_status,
        1,
    )

    cv2.putText(
        pilot_image,
        f"episodes: {total_episodes}",
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        1,
    )

    x, y, theta = robot_pose
    cv2.putText(
        pilot_image,
        f"x: {x:.2f} y: {y:.2f} theta: {theta:.2f}",
        (10,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        1,
    )

    return pilot_image

def get_lerobot_frame(
    robot_pose,
    image,
    image_raw,
    action,
    teleop_input: GamepadInput,
    
):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_raw_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    teleop_input_array = np.array(
        [
            teleop_input.left_stick_x_axis,
            teleop_input.right_trigger,
            teleop_input.left_trigger,
        ],
        dtype=np.float32
    )

    return {
        "observation.state": robot_pose,
        "observation.image": image_rgb,
        "action": action,
        "task": DATASET_TASK,
        "extras.teleop_input": teleop_input_array, 
        "extras.raw_image": image_raw_rgb, 
    }

def main():
    if ROBOT_MAC_ADDRESS_ENV not in os.environ:
        raise RuntimeError(f"{ROBOT_MAC_ADDRESS_ENV} not found in env.")
    address = os.environ[ROBOT_MAC_ADDRESS_ENV]

    fps = 10 # adjust based on file write speed
    delta_time = 1.0/fps

    robot = MockRobot()
    robot = BluetoothLowEnergyRobot(address)
    gamepad = Gamepad(GamepadConfig())
    camera = OakDRGB(fps=fps)


    bbox = (425,120,60,60) # goal pose
    reset_points = {
        "R": (64, 168),
        "1": (208, 132),
        "2": (204, 231),
        "3": (305, 74),
        "4": (310, 150),
        "5": (308, 232),
    }
    recording_status: RecordingStatus = RecordingStatus.NOT_STARTED

    total_episodes = 0
    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=fps,
        robot_type=DATASET_ROBOT_TYPE,
        features=DATASET_FEATURES,
        image_writer_threads=4,
    )

    while 1:
        step_start_time = time.perf_counter()

        frame = camera.get_frame()
        image = frame.frame
        x, y, theta = robot_pose = detect_aruco_pose(frame, debug=False)

        image_with_goal = draw_goal_position(
            image,
            bbox,
        )
        pilot_image = draw_pilot_view(
            image_with_goal,
            robot_pose,
            recording_status,
            total_episodes,
            reset_points,
        )
        cv2.imshow("raw", image)
        cv2.imshow("goal", image_with_goal)
        cv2.imshow("pilot", pilot_image)
        cv2.waitKey(1)

        teleop_input = gamepad.get_input()

        # convert user input to motor velocities.
        # note left and right are flipped since the gripper is on the back of the robot.
        (
            right_target_velocity_rad,
            left_target_velocity_rad,
        ) = gamepad.convert_user_input_to_velocities(teleop_input)

        left_target_velocity_rad *= VELOCITY_SCALE
        right_target_velocity_rad *= VELOCITY_SCALE

        robot.execute_command(DifferentialDriveRobot.Command(
            left_target_velocity_rad,
            right_target_velocity_rad,
        ))

        # start recording
        if recording_status != RecordingStatus.RECORDING and teleop_input.button_a:
            recording_status = RecordingStatus.RECORDING
        # end recording and save
        elif recording_status == RecordingStatus.RECORDING and teleop_input.button_b:
            recording_status = RecordingStatus.EPISODE_SAVED
            total_episodes += 1
            if dataset is not None:
                dataset.save_episode()
        # end recording and discard
        elif recording_status == RecordingStatus.RECORDING and teleop_input.button_y:
            recording_status = RecordingStatus.EPISODE_DISCARDED
            if dataset is not None:
                dataset.clear_episode_buffer()
        
        if recording_status == RecordingStatus.RECORDING:
            dataset_frame = get_lerobot_frame(
                robot_pose=np.array(robot_pose, dtype=np.float32),
                image=image_with_goal,
                image_raw=image,
                action=np.array([left_target_velocity_rad, right_target_velocity_rad], dtype=np.float32),
                teleop_input = teleop_input
            )
            if dataset is not None:
                dataset.add_frame(dataset_frame)

        # hit FPS
        time_elapsed = time.perf_counter() - step_start_time
        time_to_sleep = max(0, delta_time - time_elapsed)
        time.sleep(time_to_sleep)

if __name__ == "__main__":
    main()