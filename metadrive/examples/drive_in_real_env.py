#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
import random

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv

RENDER_MESSAGE = {
    "Quit": "ESC",
    "Switch perspective": "Q or B",
    "Reset Episode": "R",
    "Keyboard Control": "W,A,S,D",
}

import numpy as np


def wrap_to_pi(radians_array):
    """
    Wrap all input radians to range [-pi, pi]
    """
    if isinstance(radians_array, np.ndarray):
        wrapped_radians_array = np.mod(radians_array, 2 * np.pi)
        wrapped_radians_array[wrapped_radians_array > np.pi] -= 2 * np.pi
    # elif isinstance(radians_array, torch.Tensor):
    #     wrapped_radians_array = radians_array % (2 * torch.tensor(np.pi))
    #     wrapped_radians_array[wrapped_radians_array > torch.tensor(np.pi)] -= 2 * np.pi
    elif isinstance(radians_array, (float, np.float32)):
        wrapped_radians_array = radians_array % (2 * np.pi)
        if wrapped_radians_array > np.pi:
            wrapped_radians_array -= 2 * np.pi
    else:
        raise ValueError("Input must be a NumPy array or PyTorch tensor")

    return wrapped_radians_array


def masked_average_numpy(tensor, mask, dim):
    """
    Compute the average of tensor along the specified dimension, ignoring masked elements.
    """
    assert tensor.shape == mask.shape
    count = mask.sum(axis=dim)
    count = np.maximum(count, np.ones_like(count))
    return (tensor * mask).sum(axis=dim) / count

class TurnAction:
    STOP = 0
    KEEP_STRAIGHT = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    U_TURN = 4

    num_actions = 5

    @classmethod
    def get_str(cls, action):
        if action == TurnAction.STOP:
            return "STOP"
        elif action == TurnAction.KEEP_STRAIGHT:
            return "KEEP_STRAIGHT"
        elif action == TurnAction.TURN_LEFT:
            return "TURN_LEFT"
        elif action == TurnAction.TURN_RIGHT:
            return "TURN_RIGHT"
        elif action == TurnAction.U_TURN:
            return "U_TURN"
        else:
            raise ValueError("Unknown action: {}".format(action))


def get_direction_action_from_trajectory_batch(
    traj,
    mask,
    dt,
    U_TURN_DEG=115,
    LEFT_TURN_DEG=25,
    RIGHT_TURN_DEG=-25,
    STOP_SPEED=0.06,
):
    assert traj.ndim == 3
    traj_diff = traj[1:] - traj[:-1]
    mask_diff = mask[1:] & mask[:-1]

    displacement = np.linalg.norm(traj_diff, axis=-1)

    mask_diff_stop = mask_diff & (displacement > 0.1)

    pred_angles = np.arctan2(traj_diff[..., 1], traj_diff[..., 0])
    pred_angles_diff = wrap_to_pi(pred_angles[1:] - pred_angles[:-1])

    # It's meaning less to compute heading for a stopped vehicle. So mask them out!
    mask_diff_diff = mask_diff_stop[1:] & mask_diff_stop[:-1]
    # Note that we should not wrap to pi here because the sign is important.
    accumulated_heading_change_rad = (pred_angles_diff * mask_diff_diff).sum(axis=0)
    accumulated_heading_change_deg = np.degrees(accumulated_heading_change_rad)

    # print("accumulated_heading_change_deg: ", list(zip(ooi, accumulated_heading_change_deg)))

    speed = displacement / dt
    avg_speed = masked_average_numpy(speed, mask_diff, dim=0)

    actions = np.zeros(accumulated_heading_change_deg.shape, dtype=int)
    actions.fill(TurnAction.KEEP_STRAIGHT)
    actions[accumulated_heading_change_deg > LEFT_TURN_DEG] = TurnAction.TURN_LEFT
    actions[accumulated_heading_change_deg < RIGHT_TURN_DEG] = TurnAction.TURN_RIGHT
    actions[accumulated_heading_change_deg > U_TURN_DEG] = TurnAction.U_TURN
    actions[accumulated_heading_change_deg < -U_TURN_DEG] = TurnAction.U_TURN
    actions[avg_speed < STOP_SPEED] = TurnAction.STOP
    return actions


def get_navigation_signal(scenario, t):

    U_TURN_DEG = 115
    TURN_DEG = 25
    STOP_SPEED = 0.06
    chunk_size = 10  # 10 frames = 1s

    ego_id = scenario["metadata"]["sdc_id"]
    ego_track = scenario["tracks"][ego_id]
    ego_traj = ego_track["state"]["position"][..., :2]
    ego_valid_mask = ego_track["state"]["valid"].astype(bool)
    T = ego_traj.shape[0]
    for i in range(0, T, chunk_size):
        traj = ego_traj[i:i + chunk_size]
        mask = ego_valid_mask[i:i + chunk_size]
        if traj.shape[0] < chunk_size:
            break
        actions = get_direction_action_from_trajectory_batch(
            traj.reshape(traj, 1, -1),
            mask.reshape(chunk_size, 1),
            dt=0.1,
            U_TURN_DEG=U_TURN_DEG,
            LEFT_TURN_DEG=TURN_DEG,
            RIGHT_TURN_DEG=-TURN_DEG,
            STOP_SPEED=STOP_SPEED,
        )
        # actions.shape = (1,)
        actions = actions[0]
        print("Action at step {}, t={}s: {}".format(i, i / 10, TurnAction.get_str(actions)))

    print(111)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactive_traffic", action="store_true")
    parser.add_argument("--top_down", "--topdown", action="store_true")
    parser.add_argument("--waymo", action="store_true")
    args = parser.parse_args()
    extra_args = dict(film_size=(2000, 2000)) if args.top_down else {}
    asset_path = AssetLoader.asset_path
    use_waymo = args.waymo
    print(HELP_MESSAGE)
    try:
        env = ScenarioEnv(
            {
                "manual_control": True,
                "sequential_seed": True,
                "reactive_traffic": True if args.reactive_traffic else False,
                "use_render": True if not args.top_down else False,
                "data_directory": AssetLoader.file_path(
                    asset_path, "waymo" if use_waymo else "nuscenes", unix_style=False
                ),
                "num_scenarios": 3 if use_waymo else 10
            }
        )
        o, _ = env.reset()

        for i in range(1, 100000):
            o, r, tm, tc, info = env.step([1.0, 0.])

            get_navigation_signal(env.engine.data_manager.current_scenario, t=10)

            env.render(
                mode="top_down" if args.top_down else None,
                text=None if args.top_down else RENDER_MESSAGE,
                **extra_args
            )
            if tm or tc:
                env.reset()
    finally:
        env.close()
