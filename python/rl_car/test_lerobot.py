import argparse
import os
import cv2
from lerobot.policies.cnn.modeling_cnn import CNNPolicy
import torch
import numpy as np

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig
from lerobot.policies.act.modeling_act import ACTPolicy

def create_obs(
    image,
    
):
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image

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

def main(
    cfg: DictConfig,
    args: argparse.Namespace,
):
    if args.use_real:
        env = instantiate(cfg.env_real)
    else:
        env = instantiate(cfg.env)
    
    policy = CNNPolicy.from_pretrained(args.model_path)
    policy.config.n_action_steps = 5
    obs, info = env.reset()
    while True:
        obs_model = create_obs(info["image"])
        action = policy.select_action(obs_model)
        numpy_action = action.squeeze(0).to("cpu").numpy()

        obs, reward, done, truncated, info = env.step(numpy_action)
        if done or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL car.")
    parser.add_argument("config_path", type=str, help="Path to a hydra config for the RL task.")
    parser.add_argument("model_path", type=str, help="Path to a trained model checkpoint for the RL task.")
    parser.add_argument('--use_real', action='store_true', help="Run a demo using the real env part of the config.")
    args = parser.parse_args()

    config_dir, config_name = os.path.split(args.config_path)

    with initialize(config_path=config_dir):
        cfg = compose(config_name=config_name)
    main(cfg, args)
