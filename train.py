import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from car import CarEnv

MODEL_NAME = "ppo_car"

parameters = CarEnv.Parameters(real_time=False)
env = CarEnv(parameters)
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1e5)
model.save(MODEL_NAME)
