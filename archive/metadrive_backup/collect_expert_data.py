"""
Expert Data Collector for MetaDrive (LAM Training Pipeline)
===========================================================
Collects high-precision, continuous expert driving trajectories using MetaDrive's
built-in expert policy. This provides a crash-free, smooth, production-grade
dataset for Large Action Model (LAM) Imitation Learning.
"""

import os
import sys
import pickle
import numpy as np

user_site = os.path.expanduser("~/Library/Python/3.9/lib/python/site-packages")
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples import expert


def collect_expert_dataset(output_dir="data", num_episodes=30):
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "expert_driving_data.pkl")

    config = {
        "use_render": False,
        "manual_control": False,
        "traffic_density": 0.10,
        "num_scenarios": num_episodes,
        "start_seed": 1000,
        "map": "SSSS",
    }

    env = MetaDriveEnv(config)
    collected_dataset = []
    total_samples = 0

    print("=" * 70)
    print("      COLLECTING HIGH-PRECISION EXPERT DEMONSTRATIONS FOR LAM      ")
    print("=" * 70)

    try:
        for ep in range(num_episodes):
            obs, info = env.reset()
            episode_data = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "infos": []
            }

            for step in range(1000):
                # Generate smooth expert continuous action [steering, acceleration]
                action = expert(env.agent)
                action = [float(action[0]), float(action[1])]

                next_obs, reward, terminated, truncated, info = env.step(action)

                episode_data["observations"].append(obs)
                episode_data["actions"].append(action)
                episode_data["rewards"].append(reward)
                episode_data["infos"].append({
                    "speed": info.get("speed", 0.0),
                    "crash": info.get("crash", False),
                    "arrive_dest": info.get("arrive_dest", False)
                })

                obs = next_obs

                if terminated or truncated:
                    status = "ARRIVED DESTINATION!" if info.get("arrive_dest") else ("Crashed" if info.get("crash") else "Ended")
                    print(f" -> Episode {ep + 1}/{num_episodes} (Seed {env.current_seed}): {status} in {step + 1} steps.")
                    break

            if info.get("arrive_dest", False) or len(episode_data["actions"]) > 50:
                collected_dataset.append(episode_data)
                total_samples += len(episode_data["actions"])

    finally:
        env.close()
        if total_samples > 0:
            with open(output_filepath, "wb") as f:
                pickle.dump(collected_dataset, f)
            print("=" * 70)
            print(f" SUCCESS: Collected {total_samples} expert samples across {len(collected_dataset)} episodes.")
            print(f" Saved dataset to '{output_filepath}'.")
            print("=" * 70)

if __name__ == "__main__":
    collect_expert_dataset()
