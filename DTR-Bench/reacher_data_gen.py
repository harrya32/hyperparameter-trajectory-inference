import argparse
import os
import random

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


POLICY_CONFIG = {
    1: "policies_reacher/ppo_reacher_weight_1.zip",
    5: "policies_reacher/ppo_reacher_weight_5.zip",
}
DRIVER_WEIGHT = 1
DRIVER_ENV_FILE = "policies_reacher/vec_normalize_reacher_weight_1.pkl"


class ReacherRewardWrapper(gym.Wrapper):
    def __init__(self, env, control_cost_weight=1.0):
        super().__init__(env)
        self.control_cost_weight = control_cost_weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_dist = info["reward_dist"]
        reward_ctrl = info["reward_ctrl"]
        new_reward = reward_dist + self.control_cost_weight * reward_ctrl
        return obs, new_reward, terminated, truncated, info


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def make_reacher_env():
    env = gym.make("Reacher-v4")
    env = ReacherRewardWrapper(env, control_cost_weight=DRIVER_WEIGHT)
    return env


def main():
    parser = argparse.ArgumentParser(description="Generate Reacher HTI training data from saved PPO policies.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of environment steps to record.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="reacher_data/reacher_data.pt",
        help="Local output path inside DTR-Bench.",
    )
    parser.add_argument(
        "--nlot_output_path",
        type=str,
        default="../NLOT/data/reacher_data.pt",
        help="Path for the NLOT-compatible copy.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    print("--- Loading Reacher policies ---")
    loaded_policies = {}
    for weight, filename in POLICY_CONFIG.items():
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Missing policy checkpoint: {filename}")
        loaded_policies[weight] = PPO.load(filename, device="cpu")
        print(f"Loaded policy for weight {weight} from {filename}")

    if not os.path.exists(DRIVER_ENV_FILE):
        raise FileNotFoundError(f"Missing VecNormalize stats file: {DRIVER_ENV_FILE}")

    eval_env = VecNormalize.load(DRIVER_ENV_FILE, DummyVecEnv([make_reacher_env]))
    eval_env.training = False
    eval_env.seed(args.seed)

    actions_log = {weight: [] for weight in POLICY_CONFIG.keys()}
    obs_log = []

    print(f"--- Collecting {args.num_steps} steps with driver weight {DRIVER_WEIGHT} ---")
    obs = eval_env.reset()
    for _ in range(args.num_steps):
        obs_log.append(obs.flatten())

        for weight, policy in loaded_policies.items():
            action, _ = policy.predict(obs, deterministic=True)
            actions_log[weight].append(action.flatten())

        driver_action, _ = loaded_policies[DRIVER_WEIGHT].predict(obs, deterministic=True)
        obs, _, _, _ = eval_env.step(driver_action)

    eval_env.close()

    obs_log = np.array(obs_log, dtype=np.float32)
    tensors = []
    for weight in sorted(POLICY_CONFIG.keys()):
        actions = np.array(actions_log[weight], dtype=np.float32)
        state_action = np.concatenate((actions, obs_log), axis=1)
        tensors.append(torch.tensor(state_action, dtype=torch.float32))

    # Shape: [2, num_steps, 13] matching expected reacher HTI format.
    dataset = torch.stack(tensors, dim=0)
    print(f"Final dataset shape: {tuple(dataset.shape)}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(dataset, args.output_path)
    print(f"Saved local dataset to {args.output_path}")

    nlot_dir = os.path.dirname(args.nlot_output_path)
    if nlot_dir:
        os.makedirs(nlot_dir, exist_ok=True)
    torch.save(dataset, args.nlot_output_path)
    print(f"Copied dataset to {args.nlot_output_path}")


if __name__ == "__main__":
    main()
