import argparse
from threading import Thread
import time
import cv2
import torch
import numpy as np

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.cnn.modeling_cnn import CNNPolicy

from rl_car.utils.oak_d_camera import OakDRGB
from rl_car.utils.differential_drive_robot import BluetoothLowEnergyRobot, DifferentialDriveRobot, MockRobot

def create_obs(
    image,
    
):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    state = torch.from_numpy(np.zeros(2))
    image = torch.from_numpy(image_rgb)

    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    state = state.to(torch.float32)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)

    # Send data tensors from CPU to GPU
    state = state.to("cuda", non_blocking=True)
    image = image.to("cuda", non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    return {
        "observation.state": state,
        "observation.image": image,
    }


class Camera:
    def __init__(self):
        self.latest_image = None
        image_thread = Thread(target=self.__image_loop, daemon=True)
        image_thread.start()

    def __image_loop(self):
        camera = OakDRGB(fps=30)
        while 1:
            self.latest_image = camera.get_frame().frame
    
    def get_image(self):
        if self.latest_image is None:
            return None
        return self.latest_image.copy()


def main(
    args: argparse.Namespace,
):
    camera = Camera()
    if args.mock:
        robot = MockRobot()
    else:
        robot = BluetoothLowEnergyRobot(args.address)

    policy = CNNPolicy.from_pretrained(args.model_path)
    goal = np.random.randint(24,200,(2))


    time_last_msg = time.perf_counter()
    time_last_goal = time.perf_counter()
    while True:
        image = camera.get_image()
        if image is None:
            continue
        
        image = cv2.circle(image, list(goal), 6, (0,175,0), -1)
        image = cv2.circle(image, list(goal), 4, (0,200,0), -1)
        cv2.imshow("img", image)

        image = cv2.resize(image, (64,64), interpolation=cv2.INTER_AREA)
        cv2.imshow("model", image)

        obs_model = create_obs(image)
        action = policy.select_action(obs_model)
        numpy_action = action.squeeze(0).to("cpu").numpy()

        print(numpy_action)

        # execute robot rate limited
        if time.perf_counter() - time_last_msg > 0.03:
            robot.execute_command(DifferentialDriveRobot.Command(
                numpy_action[1] * np.pi * 2,
                numpy_action[0] * np.pi * 2,
            ))
            time_last_msg = time.perf_counter()
        
        # new goal
        if (
            cv2.waitKey(1) & 0xFF == 83 or # right arrow
            time.perf_counter() - time_last_goal > 120
        ):
            goal = np.random.randint(24,200,(2))
            time_last_goal = time.perf_counter()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL car.")
    parser.add_argument("model_path", type=str, help="Path to a trained model checkpoint for the RL task.")
    parser.add_argument("address", type=str, help="Robot mac address.")
    parser.add_argument('--mock', action='store_true', help="Run with mock robot.")
    args = parser.parse_args()

    main(args)
