import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
import os
import random

# --- 1. Time Series Dataset (Copied from training script) ---
class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time-series forecasting.
    """
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# --- 2. Model Architecture (Copied from training script) ---
class MLPForecaster(nn.Module):
    """
    A simple MLP for multi-step time-series forecasting.
    """
    def __init__(self, seq_len, pred_len):
        super(MLPForecaster, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.model = nn.Sequential(
            nn.Linear(self.seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.pred_len)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 3. Main Generation Function ---
def generate_data(args):
    """
    Loads a trained model and generates an HTI training dataset from the test data.
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Construct Model Path ---
    model_filename = f"expert_q{str(args.quantile).replace('.', '')}.pth"
    model_path = os.path.join(args.model_dir, model_filename)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # --- Load and Process Data ---
    print(f"Loading and processing data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    data = df['OT'].values.astype(float)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_split_idx = int(len(data) * 0.7)
    scaler.fit(data[:train_split_idx].reshape(-1, 1))
    data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()

    val_split_idx = int(len(data) * 0.98)
    test_data = data_scaled[val_split_idx:]
    
    test_dataset = TimeSeriesDataset(test_data, args.seq_len, args.pred_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Generating data from {len(test_dataset)} test samples.")

    # --- Load Model and Generate Forecasts ---
    print(f"Loading expert model from {model_path}...")
    model = MLPForecaster(args.seq_len, args.pred_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_inputs = []
    all_forecasts = []

    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            forecasts_batch = model(x_batch)
            all_inputs.append(x_batch.cpu())
            all_forecasts.append(forecasts_batch.cpu())

    # Concatenate all batches into single Tensors
    inputs_tensor = torch.cat(all_inputs, dim=0)
    forecasts_tensor = torch.cat(all_forecasts, dim=0)

    print(f"Shape of input histories tensor: {inputs_tensor.shape}")
    print(f"Shape of model forecasts tensor: {forecasts_tensor.shape}")

    # Horizontally concatenate the tensors to get the desired [input_history, forecast] format
    hti_dataset_tensor = torch.cat([inputs_tensor, forecasts_tensor], dim=1)
    
    print(f"Shape of final HTI dataset tensor: {hti_dataset_tensor.shape}")

    # Create output directory and save the file
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"hti_data_q{str(args.quantile).replace('.', '')}.pt"
    output_path = os.path.join(args.output_dir, output_filename)
    
    torch.save(hti_dataset_tensor, output_path)
    print(f"\nHTI training dataset successfully saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate HTI training data from a trained quantile forecaster.')
    
    parser.add_argument('--quantile', type=float, required=True, help='The quantile of the expert model to load (e.g., 0.1, 0.5, 0.9).')
    
    parser.add_argument('--data_path', type=str, default='./ETTm2.csv', help='Path to the dataset file.')
    parser.add_argument('--seq_len', type=int, default=12, help='Length of the input sequence.')
    parser.add_argument('--pred_len', type=int, default=3, help='Length of the prediction horizon.')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory where trained models are stored.')
    parser.add_argument('--output_dir', type=str, default='hti_data', help='Directory to save the generated HTI training datasets.')
    
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility.')

    args = parser.parse_args()
    
    generate_data(args)
