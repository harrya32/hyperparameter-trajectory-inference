import DTRGym
import gymnasium as gym
import numpy as np
import os
import sys
import pickle as pkl
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import argparse
import csv
from gymnasium.core import RewardWrapper
from scipy import stats

sys.path.append('../NLOT')

AGENT_PATH = "policies_hinge/ppo_ghaffari_cancer_model__0.zip" 
AGENT_NAME = "single_agent_eval"
ENV_NAME = 'GhaffariCancerEnv-continuous'
NUM_EVAL_EPISODES = 1
LAMBDA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

parser = argparse.ArgumentParser(description='Evaluate a single agent with a pushforward function.')
parser.add_argument('--lambda_pushforward', type=float, default=0, help='Lambda value for the pushforward function.')
parser.add_argument('--workspace_path', type=str, default="../NLOT/surrogate_models/ours_new.pkl", help='Path to the trained OT workspace pickle file.')
parser.add_argument('--all_lambdas', action='store_true', help='Evaluate all lambda values and generate plot')
parser.add_argument('--name', type=str, default="learned_w_potential", help='method to evaluate')
parser.add_argument('--iter', type=int, default=0, help='number to put on results')
args = parser.parse_args()
LAMBDA_PUSHFORWARD = args.lambda_pushforward
RUN_NAME = args.name
ITER = args.iter

PLOT_DIR = f"surrogate_plots_hinge/{RUN_NAME}"
os.makedirs(PLOT_DIR, exist_ok=True)
WORKSPACES_DIR = f"../NLOT/surrogate_models/reward_weighting_hinge/{RUN_NAME}/"
if not os.path.isdir(WORKSPACES_DIR):
    raise FileNotFoundError(
        f"Workspace directory not found: {WORKSPACES_DIR}. "
        "Train surrogates first via ./hti_scripts/reward_weighting_hinge.sh."
    )

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env, lambda_nk: float = 0.5, tau: float = 0.1):
        super().__init__(env)
        self.lambda_nk = lambda_nk
        self.tau = tau    # threshold on NK deviation
        self.init_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.init_obs = np.array(obs, copy=True)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.init_obs is not None:
            # NK variables
            N_p  = obs[1]
            N_s  = obs[5]
            N_p0 = self.init_obs[1]
            N_s0 = self.init_obs[5]

            N  = max(np.e, N_p  + N_s)
            N0 = max(np.e, N_p0 + N_s0)

            ratio = np.log(N) / np.log(N0)
            deviation = abs(ratio - 1.0)

            # HINGE penalty: zero if deviation <= tau. introduces non-linearity into reward shaping, for more challening HTI.
            hinge_penalty = - max(0.0, deviation - self.tau)

            reward = reward + self.lambda_nk * hinge_penalty

        return obs, reward, terminated, truncated, info

def load_workspace(workspace_path):
    if os.path.exists(workspace_path):
        print(f"Loading OT workspace from {workspace_path}")
        try:
            with open(workspace_path, "rb") as f:
                ws = pkl.load(f)
            print("Workspace loaded successfully")
            return ws
        except Exception as e:
            print(f"Error loading workspace: {e}")
    return None

def pushforward(action, obs, lambda_val, workspace):
    """
    Uses trained OT model to push actions forward to target lambda.
    """
    if workspace is None:
        print("Workspace not available, using original action")
        return action
        
    time_points = workspace.time_points
    
    if lambda_val in time_points:
        time_idx = list(time_points).index(lambda_val)
        if time_idx == 0:
            return action
    
    current_sample = np.concatenate([action.flatten(), obs.flatten()])
    
    for k in range(len(time_points) - 1):
        T_k = time_points[k]
        T_k_plus_1 = time_points[k+1]
        
        if T_k <= lambda_val <= T_k_plus_1:
            params_source_map_k = workspace.state_source_maps[k].params
            end_sample = workspace.neural_dual_solver.source_map_apply_jit(
                {'params': params_source_map_k},
                current_sample
            )
            
            if lambda_val < T_k_plus_1:
                s_fraction = (lambda_val - T_k) / (T_k_plus_1 - T_k)
                current_sample = workspace.geometry.apply(
                    {'params': workspace.params_geometry},
                    current_sample,
                    end_sample,
                    s_fraction,
                    method=workspace.geometry.point_on_path
                )
            else:
                current_sample = end_sample
            break
        
        params_source_map_k = workspace.state_source_maps[k].params
        current_sample = workspace.neural_dual_solver.source_map_apply_jit(
            {'params': params_source_map_k},
            current_sample
        )
    
    action_dim = action.shape[1] if len(action.shape) > 1 else action.shape[0]
    pushforward_action = current_sample[:action_dim].reshape(action.shape)
    
    pushforward_action = np.array(pushforward_action, dtype=np.float32)

    return pushforward_action

def calculate_nk_penalty(current_obs_vec, initial_obs_vec):
    current_obs = current_obs_vec[0]
    initial_obs = initial_obs_vec[0]

    N_p  = current_obs[1]
    N_s  = current_obs[5]
    N_p0 = initial_obs[1]
    N_s0 = initial_obs[5]

    N  = max(np.e,  N_p  + N_s)
    N0 = max(np.e,  N_p0 + N_s0)

    logN = np.log(N)
    logN0 = np.log(N0)

    penalty = -np.abs(logN / logN0 - 1)
    return penalty

def evaluate_lambda(model, lambda_val, workspace):
    """Evaluate the agent with a specific lambda pushforward value."""
    def make_env():
        env = gym.make(ENV_NAME)
        env = CustomRewardWrapper(env, lambda_nk=lambda_val * 10)
        return env
    
    env = make_vec_env(make_env, n_envs=1)
    
    overall_average_penalties = []
    episode_rewards = []
    
    print(f'\n--- Evaluating agent {AGENT_NAME} for NK penalty with lambda = {lambda_val} ---')

    for episode_num in range(NUM_EVAL_EPISODES):
        env.seed(episode_num)
        initial_obs_for_episode = env.reset()
        obs = initial_obs_for_episode
        done = False
        penalties_this_episode = []
        episode_reward = 0
        episode_steps = 0
        
        print(f"Starting Episode {episode_num + 1}/{NUM_EVAL_EPISODES}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            modified_action = pushforward(action, obs, lambda_val, workspace)
            next_obs, reward, done_array, info = env.step(modified_action)
            penalty = calculate_nk_penalty(next_obs, initial_obs_for_episode)
            penalties_this_episode.append(penalty)
            episode_reward += reward[0] 
            obs = next_obs
            done = done_array[0]
            episode_steps += 1
        
        avg_penalty_for_this_episode = np.mean(penalties_this_episode)
        overall_average_penalties.append(avg_penalty_for_this_episode)
        episode_rewards.append(episode_reward)
        print(f"Episode {episode_num + 1}: Average NK Penalty = {avg_penalty_for_this_episode:.4f}, Steps = {episode_steps}")

    env.close()
    final_avg_nk_penalty = np.mean(overall_average_penalties)
    std_dev_nk_penalty = np.std(overall_average_penalties)
    final_avg_reward = np.mean(episode_rewards) 
    print(f"\n--- Summary for Lambda = {lambda_val} ---")
    print(f"Overall Average NK Penalty across {NUM_EVAL_EPISODES} episodes = {final_avg_nk_penalty:.4f}")
    print(f"Standard Deviation of Per-Episode Average NK Penalties = {std_dev_nk_penalty:.4f}")
    print(f"Overall Average Reward across {NUM_EVAL_EPISODES} episodes = {final_avg_reward:.4f}")

    return final_avg_nk_penalty, std_dev_nk_penalty, final_avg_reward

def evaluate_single_lambda(model, lambda_val, workspace):
    env = make_vec_env(ENV_NAME, n_envs=1)

    print(f'\n--- Evaluating agent {AGENT_NAME} for NK penalty with pushforward function ---')

    overall_average_penalties = [] 
    episode_rewards = []

    for episode_num in range(NUM_EVAL_EPISODES):
        env.seed(episode_num)
        initial_obs_for_episode = env.reset()
        obs = initial_obs_for_episode
        done = False
        penalties_this_episode = []
        episode_reward = 0
        episode_steps = 0
        
        print(f"Starting Episode {episode_num + 1}/{NUM_EVAL_EPISODES}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            modified_action = pushforward(action, obs, lambda_val, workspace)
            
            next_obs, reward, done_array, info = env.step(modified_action)
            
            penalty = calculate_nk_penalty(next_obs, initial_obs_for_episode)
            penalties_this_episode.append(penalty)
            episode_reward += reward[0] 
            
            obs = next_obs
            done = done_array[0]
            episode_steps += 1
        
        if penalties_this_episode:
            avg_penalty_for_this_episode = np.mean(penalties_this_episode)
            overall_average_penalties.append(avg_penalty_for_this_episode)
            episode_rewards.append(episode_reward) 
            print(f"Episode {episode_num + 1}: Average NK Penalty = {avg_penalty_for_this_episode:.4f}, Steps = {episode_steps}")
        else:
            print(f"Warning: Agent {AGENT_NAME}, Episode {episode_num + 1} had no penalties recorded.")

    env.close()

    if overall_average_penalties:
        final_avg_nk_penalty = np.mean(overall_average_penalties)
        std_dev_nk_penalty = np.std(overall_average_penalties)
        final_avg_reward = np.mean(episode_rewards) 
        print(f"\n--- Summary for Agent: {AGENT_NAME} ---")
        print(f"Overall Average NK Penalty across {NUM_EVAL_EPISODES} episodes = {final_avg_nk_penalty:.4f}")
        print(f"Standard Deviation of Per-Episode Average NK Penalties = {std_dev_nk_penalty:.4f}")
        print(f"Overall Average Reward across {NUM_EVAL_EPISODES} episodes = {final_avg_reward:.4f}")
        return final_avg_nk_penalty, std_dev_nk_penalty, final_avg_reward
    else:
        print(f"\nNo penalties recorded for agent {AGENT_NAME} across {NUM_EVAL_EPISODES} episodes.")
        return None, None, None

print(f"Current working directory: {os.getcwd()}")

print(f"--- Loading Agent: {AGENT_NAME} ---")
model = PPO.load(AGENT_PATH)
print(f"Successfully loaded model '{AGENT_NAME}' from '{AGENT_PATH}'")
workspace_files = sorted([f for f in os.listdir(WORKSPACES_DIR) if f.endswith('.pkl')])
print(f"Found {len(workspace_files)} workspace files in {WORKSPACES_DIR}")
all_workspace_results = {}

if args.all_lambdas:
    combined_results = {}
    for lambda_val in LAMBDA_VALUES:
        combined_results[lambda_val] = {
            'penalties': [],
            'rewards': []
        }

    for workspace_file in workspace_files:
        workspace_path = os.path.join(WORKSPACES_DIR, workspace_file)
        workspace_name = os.path.splitext(os.path.basename(workspace_path))[0]
        print(f"\n===== Processing Workspace: {workspace_name} =====")
        
        workspace = load_workspace(workspace_path)
        if workspace is None:
            print(f"Could not load workspace from {workspace_path}, skipping...")
            continue
            
        lambda_results = []
        all_lambda_rewards = [] 
        
        for lambda_val in LAMBDA_VALUES:
            avg_penalty, std_dev, avg_reward = evaluate_lambda(model, lambda_val, workspace)
            lambda_results.append((lambda_val, avg_penalty, std_dev))
            all_lambda_rewards.append(avg_reward)
            
            combined_results[lambda_val]['penalties'].append(avg_penalty)
            combined_results[lambda_val]['rewards'].append(avg_reward)
        
        overall_avg_reward = np.mean(all_lambda_rewards)
        print(f"Average Reward across all lambda values for {workspace_name} = {overall_avg_reward:.4f}")
        all_workspace_results[workspace_name] = overall_avg_reward
        
        plot_lambdas = [item[0] for item in lambda_results]
        plot_avg_penalties = [item[1] for item in lambda_results]
        plot_std_devs = [item[2] for item in lambda_results]

        plt.figure(figsize=(10, 6))
        plt.errorbar(plot_lambdas, plot_avg_penalties, yerr=plot_std_devs, marker='o', linestyle='-', capsize=5)
        plt.title(f"{workspace_name}: Overall Average NK Penalty vs. Lambda Value ({NUM_EVAL_EPISODES} Episodes)")
        plt.xlabel("Lambda Value")
        plt.ylabel("Overall Average NK Penalty")
        plt.grid(True)
        plt.xticks(plot_lambdas)
        plt.tight_layout()
        plot_filename = os.path.join(PLOT_DIR, f"surrogate_avg_nk_penalty_vs_lambda_{RUN_NAME}_{workspace_name}.png")
        plt.savefig(plot_filename)
        plt.close()
        
        csv_filename = os.path.join(PLOT_DIR, f"surrogate_nk_penalty_data_{RUN_NAME}_{workspace_name}.csv")
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Lambda', 'Average Penalty', 'Standard Deviation', 'Average Reward'])
            for i, row in enumerate(lambda_results):
                writer.writerow(list(row) + [all_lambda_rewards[i]])
        print(f"Saved data for {workspace_name} to {csv_filename}")
    
    avg_penalties = []
    std_penalties = []
    avg_rewards = []
    
    for lambda_val in LAMBDA_VALUES:
        penalties = combined_results[lambda_val]['penalties']
        rewards = combined_results[lambda_val]['rewards']
        if penalties:
            avg_penalties.append(np.mean(penalties))
            std_penalties.append(np.std(penalties))
            avg_rewards.append(np.mean(rewards))
        else:
            avg_penalties.append(0)
            std_penalties.append(0)
            avg_rewards.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(LAMBDA_VALUES, avg_penalties, yerr=std_penalties, marker='o', linestyle='-', capsize=5)
    plt.title(f"Average NK Penalty vs. Lambda Value (Across All Workspaces)")
    plt.xlabel("Lambda Value")
    plt.ylabel("Average NK Penalty")
    plt.grid(True)
    plt.xticks(LAMBDA_VALUES)
    plt.tight_layout()
    avg_plot_filename = os.path.join(PLOT_DIR, f"surrogate_avg_nk_penalty_vs_lambda_all_{RUN_NAME}_{ITER}.png")
    plt.savefig(avg_plot_filename)
    print(f"\nSaved average plot across all workspaces to {avg_plot_filename}")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(LAMBDA_VALUES, avg_rewards, marker='o', linestyle='-')
    plt.title(f"Average Reward vs. Lambda Value (Across All Workspaces)")
    plt.xlabel("Lambda Value")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.xticks(LAMBDA_VALUES)
    plt.tight_layout()
    reward_plot_filename = os.path.join(PLOT_DIR, f"surrogate_avg_reward_vs_lambda_all_{RUN_NAME}_{ITER}.png")
    plt.savefig(reward_plot_filename)
    print(f"Saved average reward plot across all workspaces to {reward_plot_filename}")
    plt.close()
    
    combined_csv = os.path.join(PLOT_DIR, f"surrogate_results_{RUN_NAME}_{ITER}.csv")
    with open(combined_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Lambda', 'Average Penalty', 'Penalty StdDev', 'Average Reward'])
        for i, lambda_val in enumerate(LAMBDA_VALUES):
            writer.writerow([lambda_val, avg_penalties[i], std_penalties[i], avg_rewards[i]])
    print(f"Saved combined data to {combined_csv}")

    valid_rewards = [r for r in all_workspace_results.values() if r is not None]
    overall_reward = np.mean(valid_rewards)
    n = len(valid_rewards)
    if n > 1:
        reward_std = np.std(valid_rewards)
        reward_se = reward_std / np.sqrt(n)
        t_value = stats.t.ppf(0.975, n-1)
        reward_ci = t_value * reward_se
        
        print(f"\n===== Overall Results =====")
        print(f"Average Reward across all lambda values and all workspaces = {overall_reward:.4f}")
        print(f"95% CI for Final Average: [{overall_reward - reward_ci:.4f}, {overall_reward + reward_ci:.4f}]")
    else:
        print(f"\n===== Overall Results =====")
        print(f"Average Reward across all lambda values and all workspaces = {overall_reward:.4f}")
        print("(Not enough samples to calculate confidence interval)")
    
    final_results_file = os.path.join(PLOT_DIR, f"final_avg_reward_{RUN_NAME}_{ITER}.txt")
    with open(final_results_file, 'w') as f:
        f.write(f"===== Overall Results =====\n")
        f.write(f"Average Reward across all lambda values and all workspaces = {overall_reward:.4f}\n")
        if n > 1:
            f.write(f"95% CI for Final Average: [{overall_reward - reward_ci:.4f}, {overall_reward + reward_ci:.4f}]\n")
        else:
            f.write("(Not enough samples to calculate confidence interval)\n")
    print(f"Saved final average reward to {final_results_file}")
    print("\n===== Workspace Comparison =====")
    for workspace_name, avg_reward in all_workspace_results.items():
        print(f"{workspace_name}: Average Reward = {avg_reward:.4f}")
    
else:
    for workspace_file in workspace_files:
        workspace_path = os.path.join(WORKSPACES_DIR, workspace_file)
        workspace_name = os.path.splitext(os.path.basename(workspace_path))[0]
        print(f"\n===== Processing Workspace: {workspace_name} with Lambda = {LAMBDA_PUSHFORWARD} =====")
        
        workspace = load_workspace(workspace_path)
        if workspace is None:
            print(f"Could not load workspace from {workspace_path}, skipping...")
            continue
            
        avg_penalty, std_dev, avg_reward = evaluate_single_lambda(model, LAMBDA_PUSHFORWARD, workspace)
        all_workspace_results[workspace_name] = avg_reward
    
    print("\n===== Workspace Comparison =====")
    valid_rewards = []
    for workspace_name, avg_reward in all_workspace_results.items():
        if avg_reward is not None:
            print(f"{workspace_name}: Average Reward = {avg_reward:.4f}")
            valid_rewards.append(avg_reward)
        else:
            print(f"{workspace_name}: Evaluation failed")
    
    if valid_rewards:
        overall_avg = np.mean(valid_rewards)
        print(f"\nAverage reward across all workspaces = {overall_avg:.4f}")

print("\nEvaluation complete.")
