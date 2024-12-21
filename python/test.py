import argparse
from datetime import datetime

from dataclasses import dataclass
import os
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

MODEL_NAME = "ppo_car_gripper_v1"

def main(
    cfg: DictConfig,
    model_path: str,
):
    # env = instantiate(cfg.model.env)
    # model = PPO.load(MODEL_NAME)

    env = instantiate(cfg.env)
    model: BaseAlgorithm = instantiate(cfg.model, env=env)
    model = model.load(MODEL_NAME)

    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            obs, _ = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL car.")
    parser.add_argument("config_path", type=str, help="Path to a hydra config for the RL task.")
    parser.add_argument("model_path", type=str, help="Path to a trained model checkpoint for the RL task.")
    args = parser.parse_args()

    config_dir, config_name = os.path.split(args.config_path)

    with initialize(config_path=config_dir):
        cfg = compose(config_name=config_name)
        main(cfg, args.model_path)
