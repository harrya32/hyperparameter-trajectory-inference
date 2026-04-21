import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SEED = 1


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 1. DATASET SETUP
# ==========================================
def get_data(n_samples=2000):
    y, x = make_moons(n_samples=n_samples, noise=0.05)
    y = (y - y.mean(axis=0)) / y.std(axis=0) 
    return torch.FloatTensor(y), torch.LongTensor(x)

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, dropout_p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class Diffusion(nn.Module):
    def __init__(self, dropout_p=0.0, steps=100):
        super().__init__()
        self.steps = steps
        self.net = MLP(2 + 1 + 1, 2, dropout_p=dropout_p)
        self.beta = torch.linspace(1e-4, 0.02, steps)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def forward(self, y, x, t):
        t_norm = t.view(-1, 1).float() / self.steps
        x_cat = x.view(-1, 1).float()
        return self.net(torch.cat([y, t_norm, x_cat], dim=1))

# ==========================================
# 3. TRAINING & SAMPLING UTILS
# ==========================================
def train_model(dropout_rate, epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true, x_true = get_data()
    dataset = TensorDataset(y_true, x_true)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = Diffusion(dropout_p=dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    pbar = tqdm(range(epochs), desc=f"Training λ={dropout_rate}", leave=False)
    for epoch in pbar:
        for batch_y, batch_x in loader:
            batch_y, batch_x = batch_y.to(device), batch_x.to(device)
            t = torch.randint(0, model.steps, (batch_y.shape[0],)).to(device)
            eps = torch.randn_like(batch_y)
            a_hat = model.alpha_hat[t.cpu()].view(-1, 1).to(device)
            y_t = torch.sqrt(a_hat) * batch_y + torch.sqrt(1 - a_hat) * eps
            eps_pred = model(y_t, batch_x, t)
            loss = criterion(eps_pred, eps)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model

def generate_samples(model, num_samples=1000):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        # Half class 0, half class 1
        test_x = torch.cat([torch.zeros(num_samples//2), torch.ones(num_samples//2)]).long().to(device)
        y_gen = torch.randn(num_samples, 2).to(device)
        
        for i in reversed(range(model.steps)):
            t_batch = torch.full((num_samples,), i).to(device)
            eps_pred = model(y_gen, test_x, t_batch)
            alpha, alpha_hat, beta = model.alpha[i].to(device), model.alpha_hat[i].to(device), model.beta[i].to(device)
            z = torch.randn_like(y_gen) if i > 0 else 0
            y_gen = (1/torch.sqrt(alpha)) * (y_gen - (beta/torch.sqrt(1-alpha_hat))*eps_pred) + torch.sqrt(beta)*z
            
        # Concat coordinates with condition: (N, 2) + (N, 1) -> (N, 3)
        samples_with_cond = torch.cat([y_gen.cpu(), test_x.cpu().view(-1, 1).float()], dim=1)
        return samples_with_cond

# ==========================================
# 4. MAIN EXPERIMENT LOOP
# ==========================================
if __name__ == "__main__":
    set_seed(SEED)
    dropout_settings = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    num_samples_per_setting = 1000
    all_results = []

    print(f"Generating data for {len(dropout_settings)} settings...")
    
    for dr in dropout_settings:
        model = train_model(dr, epochs=250)
        final_samples = generate_samples(model, num_samples=num_samples_per_setting)
        all_results.append(final_samples)
        
        # Save a quick diagnostic plot for this specific setting
        plt.figure(figsize=(5, 5))
        plt.scatter(final_samples[:, 0], final_samples[:, 1], c=final_samples[:, 2], cmap='viridis', s=5, alpha=0.5)
        plt.title(f"Dropout λ={dr}")
        plt.xlim(-3, 3); plt.ylim(-3, 3)
        plt.savefig(f"plots/diffusion_2moons_dropout_{dr}.png")
        plt.close()

    # Stack results: (num_settings, num_samples, 3)
    hti_dataset = torch.stack(all_results)
    
    save_path = "diffusion_2moons_dropout.pt"
    torch.save(hti_dataset, save_path)
    nlot_save_path = os.path.join("..", "NLOT", "data", "diffusion_2moons_dropout.pt")
    os.makedirs(os.path.dirname(nlot_save_path), exist_ok=True)
    torch.save(hti_dataset, nlot_save_path)
    
    print(f"\nSuccessfully saved HTI dataset to {save_path}")
    print(f"Copied HTI dataset to {nlot_save_path}")
    print(f"Final Tensor Shape: {hti_dataset.shape}")
