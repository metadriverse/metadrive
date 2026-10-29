"""
Training Pipeline for CARLA Large Action Model (LAM) Variable Corrector.
Trains the network on realistically deviated city-block route data (straights, curves,
turns, intersections, traffic, lead vehicles) to minimize deviance from golden Autopilot reference.
"""

import os
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from carla_schema import CarlaFrame
from carla_autopilot_generator import CarlaAutopilotStreamGenerator
from carla_random_stream_generator import CarlaDevianceInductionGenerator
from carla_lam_corrector import CarlaLAMCorrectorNet


class CarlaStreamDataset(Dataset):
    def __init__(self, pairs: List[Tuple[CarlaFrame, CarlaFrame]]):
        self.features = []
        self.target_actions = []
        self.target_states = []

        for noisy_frame, ref_frame in pairs:
            feat = noisy_frame.to_feature_vector()
            target_act = ref_frame.extract_action_vector()
            target_st = np.array([
                ref_frame.environment_context.lane_offset_meters,
                ref_frame.ego_state.velocity.speed_mps
            ], dtype=np.float32)

            self.features.append(feat)
            self.target_actions.append(target_act)
            self.target_states.append(target_st)

        self.features = torch.tensor(np.array(self.features), dtype=torch.float32)
        self.target_actions = torch.tensor(np.array(self.target_actions), dtype=torch.float32)
        self.target_states = torch.tensor(np.array(self.target_states), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.target_actions[idx], self.target_states[idx]


def generate_training_data() -> Tuple[List[Tuple[CarlaFrame, CarlaFrame]], List[Tuple[CarlaFrame, CarlaFrame]]]:
    auto_gen = CarlaAutopilotStreamGenerator()
    dev_gen = CarlaDevianceInductionGenerator(seed=42)

    all_pairs: List[Tuple[CarlaFrame, CarlaFrame]] = []

    # Primary dataset: full city-block route (covers all scenario types)
    print("Generating city-block route reference stream...")
    city_frames = auto_gen.generate_city_block_route()
    city_pairs = dev_gen.generate_paired_dataset(city_frames, variations=8)
    all_pairs.extend(city_pairs)

    # Supplementary: individual scenarios for extra coverage
    print("Generating supplementary single-scenario streams...")
    for sc in ["straight_cruising", "gentle_curve", "lead_vehicle_acc", "traffic_light_stop"]:
        ref_frames = auto_gen.generate_reference_sequence(
            num_frames=150, scenario_type=sc
        )
        pairs = dev_gen.generate_paired_dataset(ref_frames, variations=4)
        all_pairs.extend(pairs)

    print(f"Total training pairs generated: {len(all_pairs)}")

    np.random.seed(42)
    indices = np.random.permutation(len(all_pairs))
    split_idx = int(len(all_pairs) * 0.8)

    train_pairs = [all_pairs[i] for i in indices[:split_idx]]
    val_pairs = [all_pairs[i] for i in indices[split_idx:]]

    return train_pairs, val_pairs


def train_lam(
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    save_path: str = "models/carla_lam_corrector.pt"
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CARLA LAM Corrector on device: {device}")

    train_pairs, val_pairs = generate_training_data()
    train_dataset = CarlaStreamDataset(train_pairs)
    val_dataset = CarlaStreamDataset(val_pairs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CarlaLAMCorrectorNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_action = nn.MSELoss()
    criterion_state = nn.SmoothL1Loss()

    best_val_loss = float("inf")

    print(f"\nStarting Training ({len(train_dataset)} train / {len(val_dataset)} val samples)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for feats, target_acts, target_sts in train_loader:
            feats = feats.to(device)
            target_acts = target_acts.to(device)
            target_sts = target_sts.to(device)

            optimizer.zero_grad()
            pred_acts, pred_sts = model(feats)

            loss_act = criterion_action(pred_acts, target_acts)
            loss_st = criterion_state(pred_sts, target_sts)

            loss = loss_act + 0.5 * loss_st
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(feats)

        scheduler.step()
        train_loss = total_train_loss / len(train_dataset)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for feats, target_acts, target_sts in val_loader:
                feats = feats.to(device)
                target_acts = target_acts.to(device)
                target_sts = target_sts.to(device)

                pred_acts, pred_sts = model(feats)
                loss_act = criterion_action(pred_acts, target_acts)
                loss_st = criterion_state(pred_sts, target_sts)
                loss = loss_act + 0.5 * loss_st
                total_val_loss += loss.item() * len(feats)

        val_loss = total_val_loss / len(val_dataset)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print(f"\nTraining Complete! Best model saved to: {save_path} (Val Loss: {best_val_loss:.5f})")


if __name__ == "__main__":
    train_lam(epochs=12, batch_size=64)
