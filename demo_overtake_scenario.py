"""
Dedicated All-in-One Autonomous Overtaking Scenario
===================================================
Demonstrates smooth, high-speed overtaking in MetaDrive:
  1. Detects slow/stopped vehicle ahead in the current lane.
  2. Executes an active, smooth lane change to the open adjacent lane.
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
    def __init__(self, cruise_speed_kmh=55.0):
        self.cruise_speed = cruise_speed_kmh
        self.target_lane_idx = None
        self.overtaking = False

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
            if dy > 0 and abs(dx) < 2.0:
                if dy < lead_dist:
                    lead_dist = dy

        # ── 2. Overtaking Tactical Decision ───────────────────────────────────
        if lead_dist < 22.0 and not self.overtaking:
            # Shift target lane to left adjacent lane if available, else right
            if self.target_lane_idx < num_lanes - 1:
                self.target_lane_idx += 1
                self.overtaking = True
                print(f"\n [OVERTAKE TRIGGERED] Slow vehicle at {lead_dist:.1f}m -> Changing LEFT to lane {self.target_lane_idx}!")
            elif self.target_lane_idx > 0:
                self.target_lane_idx -= 1
                self.overtaking = True
                print(f"\n [OVERTAKE TRIGGERED] Slow vehicle at {lead_dist:.1f}m -> Changing RIGHT to lane {self.target_lane_idx}!")

        # ── 3. Steering & Lane Centering ──────────────────────────────────────
        target_lane = ref_lanes[min(self.target_lane_idx, num_lanes - 1)]
        long_dist, lat_error = target_lane.local_coordinates(vehicle.position)
        heading_error = vehicle.heading_diff(target_lane)

        if abs(lat_error) < 0.35:
            # Aligned in target lane: use MetaDrive expert steering for smooth tracking
            base_act = expert(vehicle)
            steering = float(base_act[0])
            if lead_dist > 30.0:
                self.overtaking = False  # Overtake completed!
        else:
            # Smooth steering to transition into open target lane center
            steering = float(np.clip(-0.25 * lat_error - 0.65 * heading_error, -1.0, 1.0))

        # ── 4. High-Speed Cruise & Speed Control ──────────────────────────────
        target_sp = self.cruise_speed
        if lead_dist < 6.0:
            throttle = -1.0  # Emergency brake if vehicle cuts in
        else:
            speed_error = target_sp - current_speed
            throttle = float(np.clip(speed_error * 0.18, -1.0, 1.0))
            if self.overtaking:
                throttle = max(throttle, 0.70)  # Maintain strong speed during pass

        return [steering, throttle], lead_dist, self.overtaking


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

    except KeyboardInterrupt:
        print("\n[!] Demo stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    run_demo()
