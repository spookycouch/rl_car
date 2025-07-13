import argparse
from datetime import datetime

from dataclasses import dataclass
import os
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv

MODELS_SAVE_DIR = "./models"

@dataclass
class TrainingParameters:
    save_prefix: str
    total_timesteps: int

def get_model_save_path(
    save_prefix: str,
    training_start_time: datetime,
):
    training_start_time_str = training_start_time.strftime('%Y-%m-%dT%H:%M:%S')
    model_save_name = f"{save_prefix}_{training_start_time_str}"
    model_save_path = os.path.join(MODELS_SAVE_DIR, model_save_name)
    return model_save_path

def main(cfg: DictConfig):
    training_start_time = datetime.now()

    def make_env():
        return instantiate(cfg.env)
    
    num_envs = 1
    if num_envs > 1:
        env = SubprocVecEnv([make_env for _ in range(num_envs)])
    else:
        env = make_env()
    
    model: BaseAlgorithm = instantiate(cfg.model, env=env)
    training_params: TrainingParameters = instantiate(cfg.training_parameters)
    

    model.learn(total_timesteps=training_params.total_timesteps)

    model_save_path = get_model_save_path(training_params.save_prefix, training_start_time)
    model.save(model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL car.")
    parser.add_argument("config_path", type=str, help="Path to a hydra config for the RL task.")
    args = parser.parse_args()

    config_dir, config_name = os.path.split(args.config_path)

    with initialize(config_path=config_dir):
        cfg = compose(config_name=config_name)
        main(cfg)
