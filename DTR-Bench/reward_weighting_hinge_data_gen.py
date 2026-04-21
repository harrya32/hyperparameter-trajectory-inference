import DTRGym  
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from stable_baselines3 import PPO 
from stable_baselines3.common.env_util import make_vec_env 
import torch

SEED = 1
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

#load in agents
agent_0 = PPO.load("policies_hinge/ppo_ghaffari_cancer_model__0.zip")

agent_5 = PPO.load("policies_hinge/ppo_ghaffari_cancer_model__500.zip")

agent_10 = PPO.load("policies_hinge/ppo_ghaffari_cancer_model__1000.zip")
models = [agent_0, agent_5, agent_10]
# --- Parameters ---
ENV_NAME = 'GhaffariCancerEnv-continuous'
NUM_EVAL_EPISODES = 20         
DATASET_DIR = "reward_weighting_hinge_data"
os.makedirs(DATASET_DIR, exist_ok=True)

env = make_vec_env(ENV_NAME, n_envs=1)
env.seed(SEED)

print('Evaluating agents, using states from agent_10')

state_action_data = [[] for _ in range(len(models))] 
total_steps = 0
for episode in range(NUM_EVAL_EPISODES):
    obs = env.reset()
    done = False
    episode_steps = 0
    while not done:
        actions_by_model = []
        for model in models:
            model_actions = []
            for _ in range(10):
                action, _ = model.predict(obs, deterministic=False)
                model_actions.append(action)
            actions_by_model.append(model_actions)

        next_obs, reward, done_array, info = env.step(actions_by_model[-1][0])

        state = obs[0]
        for i, model_actions in enumerate(actions_by_model):
            for action in model_actions:
                action_flat = action.flatten() 
                state_action = np.concatenate([action_flat, state])
                state_action_data[i].append(state_action)
        
        obs = next_obs
        done = done_array[0]
        episode_steps += 1
        total_steps += 1
        
        if done:
            print(f"Episode {episode + 1} completed in {episode_steps} steps")

print(f"\nTotal steps across all episodes: {total_steps}")

# Convert data to tensor and reshape
tensor_data = []
for i in range(len(models)):
    agent_data = np.array(state_action_data[i])
    tensor_data.append(torch.tensor(agent_data, dtype=torch.float32))

dataset = torch.stack(tensor_data)
dataset_path = os.path.join(DATASET_DIR, "reward_weighting_hinge_data.pt")
print("Shape of dataset:", dataset.shape)
torch.save(dataset, dataset_path)
print(f"Saved local dataset to {dataset_path}")

nlot_data_dir = os.path.join("..", "NLOT", "data")
os.makedirs(nlot_data_dir, exist_ok=True)
nlot_dataset_path = os.path.join(nlot_data_dir, "reward_weighting_hinge_data.pt")
torch.save(dataset, nlot_dataset_path)
print(f"Copied dataset to {nlot_dataset_path}")
