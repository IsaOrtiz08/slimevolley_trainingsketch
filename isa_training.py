from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
import slimevolleygym  

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
    # Get the current observation, reward, done, and info from the environment
        obs, reward, done, info = self.env.step(action)
        
        x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy = obs

        if (ball_x - x) * vx > 0:
            reward += 1.0  # Custom reward based on the ball position and agent's velocity

        return obs, reward, done, info


env = DummyVecEnv([lambda: CustomRewardWrapper(gym.make("SlimeVolley-v0"))])

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")

# Train model
model.learn(total_timesteps=50000)

# Save model
model.save("ppo_slimevolley")

# Clean up
env.close()
