import argparse
import os

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm

def main(
    cfg: DictConfig,
    args: argparse.Namespace,
):
    if args.use_real:
        env = instantiate(cfg.env_real)
    else:
        env = instantiate(cfg.env)
    
    model: BaseAlgorithm = instantiate(cfg.model, env=env)
    model = model.load(args.model_path)

    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

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
