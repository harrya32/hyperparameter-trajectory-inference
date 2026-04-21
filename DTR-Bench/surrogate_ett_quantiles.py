import argparse
import sys
import os
import pickle as pkl
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Evaluate a single ett surrogate.')
parser.add_argument('--lambda_pushforward', type=float, default=0, help='Lambda value for the surrogate function.')
parser.add_argument('--workspace_path', type=str, default="../NLOT/surrogate_models/ett_quantile/learned_w_potential/latest_0.pkl", help='Path to the trained OT workspace pickle file.')
parser.add_argument('--all_lambdas', action='store_true', help='Evaluate all lambda values and generate plot')
parser.add_argument('--name', type=str, default="learned_w_potential", help='method to evaluate')
parser.add_argument('--iter', type=int, default=0, help='number to put on results')
sys.path.append('../NLOT')
args = parser.parse_args()
LAMBDA_PUSHFORWARD = args.lambda_pushforward
RUN_NAME = args.name
ITER = args.iter
device = "cuda:0"
PLOT_DIR = f"surrogate_plots_ett_quantile/{RUN_NAME}"
os.makedirs(PLOT_DIR, exist_ok=True)
WORKSPACES_DIR = f"../NLOT/surrogate_models/ett_quantile/{RUN_NAME}/"
if not os.path.isdir(WORKSPACES_DIR):
    raise FileNotFoundError(
        f"Workspace directory not found: {WORKSPACES_DIR}. "
        "Train surrogates first via ./hti_scripts/ett_quantiles.sh."
    )
LAMBDA_VALUES = [0.1, 0.25, 0.5, 0.75, 0.9]

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

def pushforward(forecast, input, lambda_val, workspace):
    """
    Uses trained OT model to push low (0.01) quantile forecast forward to target quantile.
    """
    if workspace is None:
        print("Workspace not available, using original forecast")
        return forecast
        
    time_points = [0.01,0.99]
    
    if lambda_val in time_points:
        time_idx = list(time_points).index(lambda_val)
        if time_idx == 0:
            return forecast

    current_sample = np.concatenate([forecast.flatten(), input.flatten()])
    
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

    forecast_dim = forecast.shape[1] if len(forecast.shape) > 1 else forecast.shape[0]
    pushforward_forecast = current_sample[:forecast_dim].reshape(forecast.shape)

    pushforward_forecast = np.array(pushforward_forecast, dtype=np.float32)

    return pushforward_forecast

def evaluate_lambda(data, lambda_val, workspace):
    """
    Evaluate the surrogate model by comparing estimates quantiles with true quantile forecasts.
    """

    base_data = data[0]
    lambda_to_index = {0.01:0, 0.1:1, 0.25:2, 0.5:3, 0.75:4, 0.9:5, 0.99:6}
    true_lambda_data = data[lambda_to_index[lambda_val]]

    mses = []
    surrogate_forecasts = []
    for i in range(len(base_data)):
        base_history = base_data[i, :12]
        base_forecast = base_data[i, 12:]
        surrogate_forecast = pushforward(base_forecast, base_history, lambda_val, workspace)
        true_forecast = true_lambda_data[i, 12:]

        """plot_path = os.path.join(PLOT_DIR, f"sample_plots/pushforward_lambda{lambda_val}_sample{i}.png")
        plt.figure(figsize=(10, 5))
        x_hist = np.arange(12)
        x_forecast = np.arange(12, 15)

        plt.plot(x_hist, base_history.flatten(), label="Base History", marker='o', color='black')
        plt.plot(x_forecast, base_forecast.flatten(), label="Base Forecast", marker='o', color='blue')
        plt.plot(x_forecast, surrogate_forecast.flatten(), label="Surrogate Forecast", marker='x', color='orange')
        plt.plot(x_forecast, true_forecast.flatten(), label="True Forecast", marker='^', color='green')

        plt.title(f"Forecasts (lambda={lambda_val}, sample={i})")
        plt.xlabel("Time Step")
        plt.ylabel("Forecast Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()"""
        mse = np.mean((surrogate_forecast - true_forecast.numpy()) ** 2)
        mses.append(mse)
        surrogate_forecasts.append(surrogate_forecast)

    return np.mean(mses), surrogate_forecasts

def main():
    print(f"Loading ett data")
    ett_data = torch.load("../quantile_regression/hti_data/hti_data_combined.pt")
    ett_testing_data = ett_data[:,1200:,:]
    print(f"Current working directory: {os.getcwd()}")
    workspace_files = sorted([f for f in os.listdir(WORKSPACES_DIR) if f.endswith('.pkl')])
    print(workspace_files)
    print(f"Found {len(workspace_files)} workspace files in {WORKSPACES_DIR}")

    combined_results = {}
    for lambda_val in LAMBDA_VALUES:
        combined_results[lambda_val] = {
            'mse': []
        }

    mses = []

    surrogate_forecasts_01 = []
    surrogate_forecasts_09 = []
    for workspace_file in workspace_files:
        workspace_mses = []
        workspace_path = os.path.join(WORKSPACES_DIR, workspace_file)
        workspace_name = os.path.splitext(os.path.basename(workspace_path))[0]
        print(f"\n===== Processing Workspace: {workspace_name} =====")
        
        workspace = load_workspace(workspace_path)
        if workspace is None:
            print(f"Could not load workspace from {workspace_path}, skipping...")
            continue
            
        lambda_results = []
        for lambda_val in LAMBDA_VALUES:
            lambda_mse, surrogate_forecasts = evaluate_lambda(ett_testing_data, lambda_val, workspace)
            workspace_mses.append(lambda_mse)
            lambda_results.append((lambda_val, lambda_mse))
            print(f"Lambda {lambda_val}: MSE = {lambda_mse}")
            combined_results[lambda_val]['mse'].append(lambda_mse)

            if lambda_val == 0.1:
                surrogate_forecasts_01.append(surrogate_forecasts)
            elif lambda_val == 0.9:
                surrogate_forecasts_09.append(surrogate_forecasts)
        
        mses.append(np.mean(workspace_mses))
        print(f"Average MSE for workspace {workspace_name}: {np.mean(workspace_mses)}")

    return combined_results, mses, surrogate_forecasts_01, surrogate_forecasts_09

def print_results(results, name):
    mean = np.mean(results)
    std = np.std(results)
    ci = 1.96 * std / np.sqrt(len(results))
    print(f"{name}: {mean:.3f} ± {ci:.3f}")

if __name__ == "__main__":
    combined_results, mses, surrogate_forecasts_01, surrogate_forecasts_09 = main()
    print("Evaluation complete. Results:")
    for lambda_val, results in combined_results.items():
        print(f"Lambda {lambda_val}: MSE = {np.mean(results['mse'])}")
    
    print_results(mses, f"Overall MSE for {RUN_NAME} (iter {ITER})")
    
    # Save MSEs to a text file
    mse_file = os.path.join(PLOT_DIR, f"mses.txt")
    with open(mse_file, "w") as f:
        f.write(f"Overall MSEs for {RUN_NAME}\n")
        for i, mse in enumerate(mses):
            f.write(f"Workspace {i}: {mse:.6f}\n")
        f.write(f"\nMean: {np.mean(mses):.6f}, Std: {np.std(mses):.6f}\n")
    print(f"Saved MSEs to {mse_file}")

    #combine surrogate forecasts into a single tensor and save
    surrogate_forecasts_01 = np.array(surrogate_forecasts_01)
    print(surrogate_forecasts_01.shape)
    surrogate_forecasts_09 = np.array(surrogate_forecasts_09)
    print(surrogate_forecasts_09.shape)
    if surrogate_forecasts_01.size > 0 and surrogate_forecasts_09.size > 0:
        combined_forecasts = np.concatenate(
            [surrogate_forecasts_01, surrogate_forecasts_09],
            axis=0
        )
        fp = os.path.join(PLOT_DIR, "surrogate_forecasts_q01_q09.npy")
        np.save(fp, combined_forecasts)
        print(f"Saved combined forecasts as NumPy array to {fp}, shape {combined_forecasts.shape}")
    else:
        print("No surrogate forecasts available to combine into a NumPy artifact.")
