"""
Test Native Navigation Lane Switching with Expert Policy
"""

import os
import sys
import numpy as np

user_site = os.path.expanduser("~/Library/Python/3.9/lib/python/site-packages")
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples import expert


def test_native_overtake():
    env = MetaDriveEnv({"use_render": False, "traffic_density": 0.12, "map": "SSSS", "num_scenarios": 5, "start_seed": 2000})
    success_count = 0

    try:
        for ep in range(5):
            obs, info = env.reset()
            v = env.agent
            print(f"\n---> Starting Episode {ep + 1}/5 (Seed: {env.current_seed})")

            target_lane_idx = 0

            for step in range(1000):
                # 1. Lead Vehicle & Multi-Lane Scan
                lead_dist = 999.0
                surrounding_objects = v.lidar.get_surrounding_objects(v)
                heading = getattr(v, "heading_theta", 0.0)
                cos_h, sin_h = np.cos(-heading), np.sin(-heading)

                for obj in surrounding_objects:
                    dx_w = obj.position[0] - v.position[0]
                    dy_w = obj.position[1] - v.position[1]
                    dx = dx_w * cos_h - dy_w * sin_h
                    dy = dx_w * sin_h + dy_w * cos_h

                    if dy > 0 and abs(dx) < 2.2:
                        if dy < lead_dist:
                            lead_dist = dy

                # 2. Hard Braking & Overtake Trigger
                current_speed_kmh = info.get("speed", 0.0)
                
                # Get base expert action
                action = expert(v)
                steering = float(action[0])
                throttle = float(action[1])

                # SAFETY GOVERNOR: If obstacle < 6.0m, APPLY HARD BRAKE (Throttle = -1.0)
                if lead_dist < 6.0:
                    throttle = -1.0  # HARD BRAKE! ZERO COLLISIONS ALLOWED!
                elif lead_dist < 18.0:
                    # Slow down to match lead vehicle speed
                    desired_speed = max(0.0, (lead_dist - 4.0) * 2.0)
                    speed_diff = desired_speed - current_speed_kmh
                    throttle = float(np.clip(speed_diff * 0.15, -1.0, 1.0))

                final_action = [steering, throttle]
                obs, reward, terminated, truncated, info = env.step(final_action)

                if terminated or truncated:
                    status = "ARRIVED DESTINATION!" if info.get("arrive_dest") else ("Crashed" if info.get("crash") else "Ended")
                    if info.get("arrive_dest"):
                        success_count += 1
                    print(f"     Episode {ep + 1} finished [{status}] in {step + 1} steps. Total Reward: {reward:.2f}")
                    break

        print("=" * 70)
        print(f" SUMMARY: Arrival Rate: {success_count}/5 ({(success_count/5)*100:.1f}%)")
        print("=" * 70)

    finally:
        env.close()

if __name__ == "__main__":
    test_native_overtake()
