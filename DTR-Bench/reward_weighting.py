import gymnasium as gym
from gymnasium.core import RewardWrapper
import DTRGym  
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import random
import torch
from stable_baselines3 import PPO 
from stable_baselines3.common.env_util import make_vec_env 

# Trains PPO agent in the GhaffariCancerEnv environment with a custom reward weighting

# --- Parse Arguments ---
parser = argparse.ArgumentParser(description='Train RL agent with custom reward weighting')
parser.add_argument('--lambda_nk', type=float, default=1.0, help='Lambda value for reward weighting (default: 1.0)')
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility.')
args = parser.parse_args()

# Use the provided lambda value
lambda_nk = args.lambda_nk
seed = args.seed

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# --- Parameters ---
ENV_NAME = 'GhaffariCancerEnv-continuous'
TOTAL_TRAINING_TIMESTEPS = 500000  
NUM_EVAL_EPISODES = 20         
MODEL_SAVE_PATH = f"policies/ppo_ghaffari_cancer_model__{int(lambda_nk * 100)}"
PLOT_DIR = "rl_agent_plots"

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env, lambda_nk: float = 0.5):
        super().__init__(env)
        self.lambda_nk = lambda_nk
        self.init_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.init_obs = np.array(obs, copy=True)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.init_obs is not None:
            N_p  = obs[1]
            N_s  = obs[5]
            N_p0 = self.init_obs[1]
            N_s0 = self.init_obs[5]
            N  = max(np.e,  N_p  + N_s)
            N0 = max(np.e,  N_p0 + N_s0)
            logN, logN0 = np.log(N), np.log(N0)
            nk_penalty = -np.abs(logN / logN0 - 1)
            reward = reward + self.lambda_nk * nk_penalty

        return obs, reward, terminated, truncated, info

# --- Create Environment ---
def make_env():
    env = gym.make(ENV_NAME)
    env = CustomRewardWrapper(env, lambda_nk=lambda_nk)
    return env

env = make_vec_env(make_env, n_envs=1)
env.seed(seed)

# --- Define and Train the Agent ---
print(f"--- Training PPO Agent on {ENV_NAME} ---")
gamma = 0.99
model = PPO("MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=os.path.join(PLOT_DIR, "tensorboard_logs"), 
            gamma=gamma,
            seed=seed,
)

# Train the agent
model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS, progress_bar=True)
model.save(MODEL_SAVE_PATH)
print(f"--- Training Complete. Model saved to {MODEL_SAVE_PATH}.zip ---")

# --- Evaluate the Trained Agent ---
print(f"\n--- Evaluating Trained Agent for {NUM_EVAL_EPISODES} episodes ---")

obs = env.reset()
eval_actions = []
eval_episode_rewards = []        
eval_per_timestep_rewards = [] 


for episode in range(NUM_EVAL_EPISODES):
    done = False
    episode_reward = 0
    rewards = []

    if episode > 0:
         obs = env.reset() 

    print(f"Starting Evaluation Episode {episode + 1}")
    while not done:
        action, _states = model.predict(obs, deterministic=False)
        eval_actions.append(action[0])
        obs, reward, done, info = env.step(action)
        scalar_reward = reward[0]
        scalar_done = done[0]
        rewards.append(scalar_reward)
        episode_reward += scalar_reward
        done = scalar_done

    eval_episode_rewards.append(episode_reward)
    eval_per_timestep_rewards.append(rewards)
    print(f"Finished Evaluation Episode {episode + 1}, Total Reward: {episode_reward}")

env.close()
print("--- Evaluation Complete ---")
