from stable_baselines3 import PPO
from robotic_env_real import KukaArmEnv
import os
import numpy as np

env = KukaArmEnv(render=False)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=30000)

model.save("ppo_kuka_arm")

os.makedirs("logs", exist_ok=True)
np.save("logs/rewards_kuka.npy", [])  # puedes guardar reales si deseas luego
