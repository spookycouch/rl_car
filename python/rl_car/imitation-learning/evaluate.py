from enum import Enum, auto
import os
from termios import VEOL
import time
from typing import Dict, Tuple
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from rl_car.utils.differential_drive_robot import BluetoothLowEnergyRobot, DifferentialDriveRobot, MockRobot
from rl_car.utils.oak_d_camera import OakDRGB, CameraFrame

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

from teleop import detect_aruco_pose, draw_goal_position, draw_pilot_view, GOAL_AREA, get_random_pose
from lerobot.common.policies.act.modeling_act import ACTPolicy

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



def create_obs(
    robot_pose,
    image,
    image_raw,
    
):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_raw_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

    state = torch.from_numpy(np.array(robot_pose))
    image = torch.from_numpy(image_rgb)
    raw_image = torch.from_numpy(image_raw_rgb)

    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    state = state.to(torch.float32)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)
    raw_image = raw_image.to(torch.float32) / 255
    raw_image = raw_image.permute(2, 0, 1)


    # Send data tensors from CPU to GPU
    state = state.to("cuda", non_blocking=True)
    image = image.to("cuda", non_blocking=True)
    raw_image = raw_image.to("cuda", non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)
    raw_image = raw_image.unsqueeze(0)


    return {
        "observation.state": state,
        "observation.image": image,
        "extras.raw_img": raw_image, 
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

    # pretrained_policy_path = Path("checkpoints/2025-03-09/22-36-18_act/checkpoints/200000/pretrained_model")
    # pretrained_policy_path = Path("checkpoints/2025-05-18/20250518_differential_drive_push_soy/checkpoints/400000/pretrained_model")
    pretrained_policy_path = Path("checkpoints/2025-05-18/20250518_differential_drive_push_soy/checkpoints/0950000/pretrained_model")
    # pretrained_policy_path = Path("checkpoints/2025-05-12/20250512_differential_drive_push_soy/checkpoints/010000/pretrained_model")
    policy = ACTPolicy.from_pretrained(pretrained_policy_path, map_location="cuda")
    policy.config.n_action_steps = 5
    print(policy.config.input_features)
    print(policy.config.output_features)


    recording_status: RecordingStatus = RecordingStatus.NOT_STARTED

    total_episodes = 0

    robot_start_pose = get_random_pose()
    object_start_pose = get_random_pose()

    execute = False
    while 1:
        step_start_time = time.perf_counter()

        frame = camera.get_frame()
        image = frame.frame
        x, y, theta = robot_pose = detect_aruco_pose(frame, debug=False)

        image_with_goal = draw_goal_position(
            image,
            GOAL_AREA,
        )
        pilot_image = draw_pilot_view(
            image_with_goal,
            robot_pose,
            recording_status,
            total_episodes,
            robot_start_pose,
            object_start_pose,
        )
        cv2.imshow("raw", image)
        cv2.imshow("goal", image_with_goal)
        cv2.imshow("pilot", pilot_image)
        cv2.waitKey(1)

        observation = create_obs(
            robot_pose,
            image_with_goal,
            image,
        )

        teleop_input = gamepad.get_input()
        if teleop_input.button_a:
            execute = True
            policy.reset()
        elif teleop_input.button_b:
            execute = False
            robot_start_pose = get_random_pose()
            object_start_pose = get_random_pose()

        if execute:
            with torch.inference_mode():
                action = policy.select_action(observation)
                numpy_action = action.squeeze(0).to("cpu").numpy()
                (
                    left_target_velocity_rad,
                    right_target_velocity_rad,
                ) = numpy_action

            robot.execute_command(DifferentialDriveRobot.Command(
                left_target_velocity_rad,
                right_target_velocity_rad,
            ))
        else:
            # convert user input to motor velocities.
            # note left and right are flipped since the gripper is on the back of the robot.
            (
                right_target_velocity_rad,
                left_target_velocity_rad,
            ) = gamepad.convert_user_input_to_velocities(teleop_input)

            robot.execute_command(DifferentialDriveRobot.Command(
                left_target_velocity_rad * VELOCITY_SCALE,
                right_target_velocity_rad * VELOCITY_SCALE,
            ))

        # hit FPS
        time_elapsed = time.perf_counter() - step_start_time
        time_to_sleep = max(0, delta_time - time_elapsed)
        time.sleep(time_to_sleep)

if __name__ == "__main__":
    main()