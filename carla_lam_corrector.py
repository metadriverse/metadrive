"""
CARLA Large Action Model (LAM) Variable Corrector.
Neural Action-Correction network that consumes deviated CARLA simulation streams
and outputs altered, crashless variables minimizing deviance from perfect Autopilot reference models.
"""

import math
import copy
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from carla_schema import CarlaFrame


class CarlaResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.05):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.fc1(x)))
        x = self.drop(x)
        x = self.norm2(self.fc2(x))
        x = self.act(x + residual)
        return x


class CarlaLAMCorrectorNet(nn.Module):
    """
    Deep Residual Large Action Model for CARLA Variable Stream Alteration.
    Maps realistic simulation state-action deviances to optimal reference actions and state corrections.
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        action_dim: int = 3   # (steer, throttle, brake)
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.blocks = nn.ModuleList([
            CarlaResidualBlock(hidden_dim=hidden_dim) for _ in range(num_blocks)
        ])

        # Action Correction Head: [steer in [-1, 1], throttle in [0, 1], brake in [0, 1]]
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, action_dim)
        )

        # Dynamic State Recovery Head: predicts ideal lane_offset (target 0) and speed
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)  # [predicted_lane_offset, predicted_speed_mps]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)

        raw_actions = self.action_head(h)
        steer = torch.tanh(raw_actions[..., 0:1])
        throttle = torch.sigmoid(raw_actions[..., 1:2])
        brake = torch.sigmoid(raw_actions[..., 2:3])
        actions = torch.cat([steer, throttle, brake], dim=-1)

        states = self.state_head(h)
        return actions, states


class CarlaLAMCorrector:
    """
    High-level engine that wraps the neural network for training, real-time stream alteration,
    and deviance minimization.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = CarlaLAMCorrectorNet().to(self.device)

        if model_path:
            self.load(model_path)

    def alter_frame(
        self,
        noisy_frame: CarlaFrame,
        blend_factor: float = 1.0
    ) -> Tuple[CarlaFrame, Dict[str, float]]:
        self.model.eval()
        feat = noisy_frame.to_feature_vector()
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            pred_actions, pred_states = self.model(feat_t)

        pred_act = pred_actions.squeeze(0).cpu().numpy()
        pred_st = pred_states.squeeze(0).cpu().numpy()

        initial_steer = noisy_frame.control.steer
        initial_throttle = noisy_frame.control.throttle
        initial_brake = noisy_frame.control.brake
        initial_offset = noisy_frame.environment_context.lane_offset_meters

        corrected_dict = copy.deepcopy(noisy_frame.to_dict())
        corrected_frame = CarlaFrame.from_dict(corrected_dict)

        target_steer = float(pred_act[0])
        target_throttle = float(pred_act[1])
        target_brake = float(pred_act[2])

        if target_brake > 0.15:
            target_throttle = 0.0

        corrected_frame.control.steer = round(
            float((1.0 - blend_factor) * initial_steer + blend_factor * target_steer), 3
        )
        corrected_frame.control.throttle = round(
            float((1.0 - blend_factor) * initial_throttle + blend_factor * target_throttle), 3
        )
        corrected_frame.control.brake = round(
            float((1.0 - blend_factor) * initial_brake + blend_factor * target_brake), 3
        )

        recovered_offset = float(pred_st[0])
        corrected_frame.environment_context.lane_offset_meters = round(
            float((1.0 - blend_factor) * initial_offset + blend_factor * recovered_offset), 3
        )

        if abs(corrected_frame.environment_context.lane_offset_meters) < 0.35:
            corrected_frame.environment_context.collision_warning = False

        deviance_metrics = {
            "initial_steer": round(initial_steer, 3),
            "corrected_steer": corrected_frame.control.steer,
            "steer_correction_delta": round(corrected_frame.control.steer - initial_steer, 3),
            "initial_throttle": round(initial_throttle, 3),
            "corrected_throttle": corrected_frame.control.throttle,
            "initial_brake": round(initial_brake, 3),
            "corrected_brake": corrected_frame.control.brake,
            "initial_lane_offset": round(initial_offset, 3),
            "corrected_lane_offset": corrected_frame.environment_context.lane_offset_meters,
            "offset_reduction_percent": round(
                max(0.0, (abs(initial_offset) - abs(corrected_frame.environment_context.lane_offset_meters)) / (abs(initial_offset) + 1e-5)) * 100.0, 1
            ),
            "collision_warning_resolved": (
                noisy_frame.environment_context.collision_warning and not corrected_frame.environment_context.collision_warning
            )
        }

        return corrected_frame, deviance_metrics

    def alter_stream(
        self,
        noisy_stream: List[CarlaFrame]
    ) -> Tuple[List[CarlaFrame], List[Dict[str, float]]]:
        corrected_stream = []
        metrics_stream = []
        for frame in noisy_stream:
            corr_frame, metrics = self.alter_frame(frame)
            corrected_stream.append(corr_frame)
            metrics_stream.append(metrics)
        return corrected_stream, metrics_stream

    def save(self, filepath: str) -> None:
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        state_dict = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Loaded LAM model weights from {filepath}")
