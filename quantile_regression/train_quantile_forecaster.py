import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
import os
import time
import random
import matplotlib.pyplot as plt # <-- NEW IMPORT

# --- 1. Quantile Loss Function (Pinball Loss) ---
def quantile_loss(preds, target, quantile):
    """
    Calculates the pinball loss between predictions and target.
    """
    assert 0 < quantile < 1
    errors = target - preds
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    return loss.mean()

# --- 2. Time Series Dataset ---
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

# --- 3. Model Architecture (Simple MLP) ---
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
        # Flatten the input sequence
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

# --- 4. Main Training Function ---
def train(args):
    """
    Main function to handle data loading, training, and saving the model.
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preprocessing ---
    print("Loading and preprocessing data...")
    df = pd.read_csv(args.data_path)
    data = df['OT'].values.astype(float)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_split_idx = int(len(data) * 0.7)
    scaler.fit(data[:train_split_idx].reshape(-1, 1))
    data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()

    val_split_idx = int(len(data) * 0.85)
    train_data = data_scaled[:train_split_idx]
    val_data = data_scaled[train_split_idx:val_split_idx]
    
    train_dataset = TimeSeriesDataset(train_data, args.seq_len, args.pred_len)
    val_dataset = TimeSeriesDataset(val_data, args.seq_len, args.pred_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")

    # --- Model, Optimizer, and Training ---
    model = MLPForecaster(args.seq_len, args.pred_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n--- Training Expert Model for Quantile τ = {args.quantile} ---")
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = quantile_loss(preds, y_batch, args.quantile)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                loss = quantile_loss(preds, y_batch, args.quantile)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_time:.2f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0 
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"expert_q{str(args.quantile).replace('.', '')}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  -> Validation loss improved. Model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement in validation loss for {epochs_no_improve} epoch(s). Patience: {args.patience}")

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    print("\n--- Training complete! ---")
    print(f"Best model for quantile {args.quantile} saved with validation loss: {best_val_loss:.6f}")

    print("\nGenerating and saving loss plot...")
    epochs_ran = len(train_losses)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs_ran + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs_ran + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Quantile Loss')
    plt.title(f'Training & Validation Loss for Quantile τ = {args.quantile}')
    plt.legend()
    plt.grid(True)
    
    # Construct filename and save
    plot_filename = f"loss_plot_q{str(args.quantile).replace('.', '')}.png"
    plot_save_path = os.path.join(args.save_dir, plot_filename)
    plt.savefig(plot_save_path)
    print(f"Loss plot saved to {plot_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a quantile regressor for time-series forecasting.')
    
    parser.add_argument('--quantile', type=float, required=True, help='The target quantile to train for (e.g., 0.1, 0.5, 0.9).')
    
    parser.add_argument('--data_path', type=str, default='./ETTm2.csv', help='Path to the dataset file.')
    parser.add_argument('--seq_len', type=int, default=12, help='Length of the input sequence.')
    parser.add_argument('--pred_len', type=int, default=3, help='Length of the prediction horizon.')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models and plots.')
    
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait for improvement before stopping.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility.')

    args = parser.parse_args()
    
    train(args)
