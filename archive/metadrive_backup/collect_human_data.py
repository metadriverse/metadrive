"""
Human Data Collector for MetaDrive (LAM Training Pipeline - Phase 1)
====================================================================
This script launches an interactive MetaDrive environment where a human drives
the vehicle using keyboard controls. Continuous actions [steering, throttle/brake]
and sensor observations are saved into a dataset for Large Action Model (LAM)
Imitation Learning / Behavior Cloning.

Controls:
  - W / UP ARROW    : Accelerate / Throttle
  - S / DOWN ARROW  : Brake / Reverse
  - A / LEFT ARROW  : Steer Left
  - D / RIGHT ARROW : Steer Right
  - R               : Reset Episode
  - ESC / Ctrl+C    : Save Dataset & Exit
"""

import os
import sys
import pickle
import numpy as np

# Ensure site-packages or local metadrive is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
user_site = os.path.expanduser("~/Library/Python/3.9/lib/python/site-packages")
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.manual_control_policy import ManualControlPolicy


def collect_data(output_dir="data", max_episodes=10, max_steps_per_episode=1000):
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "human_driving_data.pkl")

    config = {
        "use_render": True,
        "manual_control": True,
        "agent_policy": ManualControlPolicy,
        "traffic_density": 0.10,
        "num_scenarios": 20,
        "start_seed": 1000,
        "map": "SSSS",  # Surface streets with curves and traffic
        "window_size": (1200, 800),
    }

    print("=" * 70)
    print("      METADRIVE HUMAN TELEOPERATION & DATA COLLECTOR FOR LAM      ")
    print("=" * 70)
    print("Instructions:")
    print("  • Drive smoothly using W/A/S/D or Arrow keys.")
    print("  • Maintain lane position, avoid collisions, and yield to traffic.")
    print("  • Dataset will record (Observation, Action) pairs at 20 Hz.")
    print("  • Press 'R' to reset episode, or close window / Ctrl+C to save.")
    print("=" * 70)

    env = MetaDriveEnv(config)
    collected_dataset = []
    total_samples = 0

    try:
        for ep in range(max_episodes):
            obs, info = env.reset()
            episode_data = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "infos": []
            }
            
            print(f"\n---> Starting Episode {ep + 1}/{max_episodes} (Seed: {env.current_seed})")
            
            for step in range(max_steps_per_episode):
                # Step environment with dummy action [0,0]; ManualControlPolicy overrides action via keyboard
                next_obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
                
                # Retrieve actual action taken by the human policy
                policy = env.engine.get_policy(env.agent.id)
                action = policy.action_info.get("action", [0.0, 0.0])
                
                env.render(
                    text={
                        "Episode": f"{ep + 1}/{max_episodes}",
                        "Step": step,
                        "Collected Samples": total_samples + len(episode_data["actions"]),
                        "Speed (km/h)": f"{info.get('speed', 0.0):.1f}",
                        "Steering": f"{action[0]:+.2f}",
                        "Throttle/Brake": f"{action[1]:+.2f}"
                    }
                )

                # Record frame sample
                episode_data["observations"].append(obs)
                episode_data["actions"].append(action)
                episode_data["rewards"].append(reward)
                episode_data["infos"].append({
                    "speed": info.get("speed", 0.0),
                    "crash": info.get("crash", False),
                    "out_of_road": info.get("out_of_road", False),
                    "arrive_dest": info.get("arrive_dest", False)
                })

                obs = next_obs

                if terminated or truncated:
                    status = "Arrived" if info.get("arrive_dest") else ("Crashed" if info.get("crash") else "Ended")
                    print(f"     Episode {ep + 1} finished [{status}] in {step + 1} steps.")
                    break

            if len(episode_data["actions"]) > 10:
                collected_dataset.append(episode_data)
                total_samples += len(episode_data["actions"])
                print(f"     Recorded {len(episode_data['actions'])} frames. Total dataset size: {total_samples} samples.")

    except KeyboardInterrupt:
        print("\n[!] Data collection interrupted by user.")
    except Exception as e:
        print(f"\n[!] Environment window closed or exception: {e}")
    finally:
        env.close()
        if total_samples > 0:
            with open(output_filepath, "wb") as f:
                pickle.dump(collected_dataset, f)
            print("=" * 70)
            print(f" SUCCESS: Saved {total_samples} human driving samples to '{output_filepath}'.")
            print(" Next step: Run 'python train_lam.py' to train the LAM policy!")
            print("=" * 70)
        else:
            print("[!] No samples recorded.")

if __name__ == "__main__":
    collect_data()
