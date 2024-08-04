import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from car import CarEnv

MODEL_NAME = "ppo_car"

parameters = CarEnv.Parameters(real_time=True)
env = CarEnv(parameters)

model = PPO.load(MODEL_NAME)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()