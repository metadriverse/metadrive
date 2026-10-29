"""
Dedicated All-in-One Autonomous Overtaking Scenario
===================================================
Demonstrates smooth, high-speed overtaking in MetaDrive:
  1. Detects slow/stopped vehicle ahead in the current lane.
  2. Executes a smooth, native lane change to the open adjacent lane.
  3. Accelerates past the obstacle and cruises continuously to Destination B!

Usage:
    python demo_overtake_scenario.py
"""

import os
import sys
import numpy as np

user_site = os.path.expanduser("~/Library/Python/3.9/lib/python/site-packages")
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples import expert


class SmoothOvertakePlanner:
    def __init__(self, cruise_speed_kmh=60.0):
        self.cruise_speed = cruise_speed_kmh
        self.target_lane_idx = None
        self.cooldown = 0

    def get_action(self, env):
        vehicle = env.agent
        current_speed = getattr(vehicle, "speed", 0.0) * 3.6  # km/h
        
        ref_lanes = vehicle.navigation.current_ref_lanes
        num_lanes = len(ref_lanes) if ref_lanes else 3
        
        # Current lane index (0 = Right, 1 = Middle, 2 = Left)
        current_lane_idx = 0
        if hasattr(vehicle, "lane_index") and len(vehicle.lane_index) == 3:
            current_lane_idx = vehicle.lane_index[2]

        if self.target_lane_idx is None:
            self.target_lane_idx = current_lane_idx

        if self.cooldown > 0:
            self.cooldown -= 1

        # ── 1. Obstacle Detection ─────────────────────────────────────────────
        lead_dist = 999.0
        surrounding_objects = vehicle.lidar.get_surrounding_objects(vehicle)
        heading = getattr(vehicle, "heading_theta", 0.0)
        cos_h, sin_h = np.cos(-heading), np.sin(-heading)

        for obj in surrounding_objects:
            dx_w = obj.position[0] - vehicle.position[0]
            dy_w = obj.position[1] - vehicle.position[1]
            dx = dx_w * cos_h - dy_w * sin_h  # lateral (+ is left)
            dy = dx_w * sin_h + dy_w * cos_h  # longitudinal (+ is forward)

            # Detect obstacles in the current forward path
            if dy > 0 and abs(dx) < 2.2:
                if dy < lead_dist:
                    lead_dist = dy

        # ── 2. Native Overtaking Decision ──────────────────────────────────────
        if lead_dist < 22.0 and self.cooldown == 0:
            if current_lane_idx < num_lanes - 1:
                self.target_lane_idx = current_lane_idx + 1
                self.cooldown = 40
                # Update navigation reference lane natively
                if ref_lanes and self.target_lane_idx < len(ref_lanes):
                    vehicle.navigation.current_ref_lanes = [ref_lanes[self.target_lane_idx]] + [l for i, l in enumerate(ref_lanes) if i != self.target_lane_idx]
                print(f"\n [OVERTAKE TRIGGERED] Slow vehicle at {lead_dist:.1f}m -> Changing LEFT to lane {self.target_lane_idx}!")

        # ── 3. Native Expert Control ──────────────────────────────────────────
        base_act = expert(vehicle)
        steering = float(base_act[0])
        throttle = float(base_act[1])

        # Emergency Distance Keeping (Brake if vehicle cuts in < 5.0m)
        if lead_dist < 5.0:
            throttle = -1.0  # Apply hard brake

        return [steering, throttle], lead_dist, (self.cooldown > 0)


def run_demo(num_episodes=5):
    config = {
        "use_render": True,
        "manual_control": False,
        "traffic_density": 0.05,  # Low density ensures clear adjacent lanes for high-speed pass
        "num_scenarios": 10,
        "start_seed": 2000,
        "map": "SSSS",
        "window_size": (1200, 800),
    }

    env = MetaDriveEnv(config)
    success_count = 0

    print("=" * 70)
    print("      ALL-IN-ONE AUTONOMOUS OVERTAKING & LANE CHANGE DEMO      ")
    print("=" * 70)

    try:
        for ep in range(num_episodes):
            obs, info = env.reset()
            planner = SmoothOvertakePlanner(cruise_speed_kmh=60.0)
            print(f"\n---> Starting Overtake Demo Episode {ep + 1}/{num_episodes} (Seed: {env.current_seed})")

            for step in range(1000):
                action, lead_dist, is_passing = planner.get_action(env)
                obs, reward, terminated, truncated, info = env.step(action)
                current_speed = info.get("speed", 0.0)

                env.render(
                    text={
                        "Episode": f"{ep + 1}/{num_episodes}",
                        "Step": step,
                        "Speed (km/h)": f"{current_speed:.1f}",
                        "Tactical State": "PASSING / OVERTAKING!" if is_passing else "CRUISING LANE",
                        "Lead Vehicle": f"{lead_dist:.1f}m" if lead_dist < 50 else "Clear (>50m)",
                        "Status": "ARRIVED!" if info.get("arrive_dest") else ("Crashed!" if info.get("crash") else "Driving")
                    }
                )

                if terminated or truncated:
                    status = "ARRIVED DESTINATION!" if info.get("arrive_dest") else ("Crashed" if info.get("crash") else "Ended")
                    if info.get("arrive_dest"):
                        success_count += 1
                    print(f"     Episode {ep + 1} finished [{status}] in {step + 1} steps.")
                    break

        print("=" * 70)
        print(f" SUMMARY: Overtake Demo completed {num_episodes} episodes. Arrival Rate: {success_count}/{num_episodes} ({(success_count/num_episodes)*100:.1f}%)")
        print("=" * 70)

    finally:
        env.close()

if __name__ == "__main__":
    run_demo()
