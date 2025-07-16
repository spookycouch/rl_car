import argparse
import os

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

DATASET_TASK = "Push red cube to green circle."
DATASET_FEATURES = {
    "observation.image": {
        "dtype": "image",
        "shape": (3, 64, 64),
        "names": [
            "channels",
            "height",
            "width",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (2,),
        "names": {
            "axes": ["pad0", "pad1"],
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": {
            "axes": ["left", "right"],
        },
    },
}
DATASET_REPO_ID = "ppo_car_gripper_sim"
DATASET_ROBOT_TYPE = "differential drive"

def main(
    cfg: DictConfig,
    args: argparse.Namespace,
):
    if args.use_real:
        env = instantiate(cfg.env_real)
    else:
        env = instantiate(cfg.env)

    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=30,
        robot_type=DATASET_ROBOT_TYPE,
        features=DATASET_FEATURES,
        image_writer_threads=4,
    )

    model: BaseAlgorithm = instantiate(cfg.model, env=env)
    model = model.load(args.model_path)

    obs, info = env.reset()
    episodes = 0
    while episodes < 10_000:
        action, _states = model.predict(obs)

        dataset_frame = {
            "observation.image": info["image"],
            "observation.state": np.zeros(2, dtype=np.float32),
            "action": action,
        }
        dataset.add_frame(dataset_frame, task=DATASET_TASK)

        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, info = env.reset()
            dataset.save_episode()
            episodes += 1

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
