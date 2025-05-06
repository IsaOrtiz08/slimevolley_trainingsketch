from stable_baselines3 import PPO
import gym
import slimevolleygym
import numpy as np

env = gym.make("SlimeVolley-v0")
model = PPO.load("ppo_slimevolley")  # adjust path if needed

obs = env.reset()
done = False
total_reward = 0

while not done:
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, done, _ = env.step(action)
    env.render()
    total_reward += reward

print("Cumulative reward:", total_reward)
env.close()
