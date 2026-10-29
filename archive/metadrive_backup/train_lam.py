"""
LAM Behavior Cloning / Imitation Learning Trainer (Phase 2)
===========================================================
Trains a Large Action Model (LAM) policy using Supervised Learning (MSE Loss)
on human expert demonstration data collected via `collect_human_data.py`.

Usage:
    python train_lam.py --dataset data/human_driving_data.pkl --epochs 50
"""

import os
import sys
import pickle
import argparse
import numpy as np

# Ensure dependencies are available
user_site = os.path.expanduser("~/Library/Python/3.9/lib/python/site-packages")
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class HumanDrivingDataset(Dataset):
    """PyTorch Dataset wrapper for human teleoperation trajectories."""
    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset '{dataset_path}' not found! Run 'python collect_human_data.py' first.")
        
        with open(dataset_path, "rb") as f:
            raw_data = pickle.load(f)

        all_obs = []
        all_actions = []

        for episode in raw_data:
            obs_list = episode["observations"]
            act_list = episode["actions"]
            infos_list = episode.get("infos", [])
            
            # Check if episode ended in a crash
            has_crashed = any(info.get("crash", False) for info in infos_list)
            
            # If episode crashed, drop the last 20 frames (the crash impact / bad steering steps)
            cutoff = -20 if (has_crashed and len(obs_list) > 30) else len(obs_list)
            
            for o, a in zip(obs_list[:cutoff], act_list[:cutoff]):
                if o is not None and a is not None:
                    all_obs.append(o)
                    all_actions.append(a)

        self.observations = torch.tensor(np.array(all_obs), dtype=torch.float32)
        self.actions = torch.tensor(np.array(all_actions), dtype=torch.float32)

        print(f"[Dataset Loaded] Total Samples: {len(self.observations)} | Observation Shape: {self.observations.shape[1]} | Action Dim: {self.actions.shape[1]}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class LAMPolicyNetwork(nn.Module):
    """
    Large Action Model Policy Network for Continuous Vehicle Control.
    Maps high-dimensional sensor/state observations -> [Steering, Throttle/Brake].
    """
    def __init__(self, input_dim, action_dim=2, hidden_dim=256):
        super(LAMPolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Output range strictly constrained to [-1.0, +1.0]
        )

    def forward(self, x):
        return self.net(x)


def train_lam(dataset_path="data/human_driving_data.pkl", output_model_dir="models", epochs=50, batch_size=64, lr=1e-3):
    os.makedirs(output_model_dir, exist_ok=True)
    model_save_path = os.path.join(output_model_dir, "lam_policy.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\n[Training Device]: {device}")

    # Load Dataset
    dataset = HumanDrivingDataset(dataset_path)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize Model & Optimizer
    input_dim = dataset.observations.shape[1]
    model = LAMPolicyNetwork(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 70)
    print("                STARTING LAM BEHAVIOR CLONING TRAINING               ")
    print("=" * 70)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_obs, batch_act in train_loader:
            batch_obs, batch_act = batch_obs.to(device), batch_act.to(device)

            optimizer.zero_grad()
            pred_act = model(batch_obs)
            loss = criterion(pred_act, batch_act)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_obs)

        train_loss /= train_size
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_obs, batch_act in val_loader:
                batch_obs, batch_act = batch_obs.to(device), batch_act.to(device)
                pred_act = model(batch_obs)
                loss = criterion(pred_act, batch_act)
                val_loss += loss.item() * len(batch_obs)

        val_loss /= val_size

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            saved_str = " [Checkpoint Saved!]"
        else:
            saved_str = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss (MSE): {train_loss:.6f} | Val Loss: {val_loss:.6f}{saved_str}")

    print("=" * 70)
    print(f" SUCCESS: Trained LAM Policy model saved to '{model_save_path}'. Best Val Loss: {best_val_loss:.6f}")
    print(" Next step: Run 'python eval_lam.py' to evaluate your trained LAM policy!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LAM Policy via Imitation Learning")
    parser.add_argument("--dataset", type=str, default="data/human_driving_data.pkl", help="Path to recorded human driving dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_lam(dataset_path=args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
