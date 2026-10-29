"""
Test Overtaking & Distance Keeping ADAS Controller
===================================================
1. Detects slow lead vehicle ahead.
2. If adjacent lane is available and clear -> smooth lane change & overtake.
3. If no adjacent lane is available -> brakes and maintains safe follow distance behind lead vehicle.
"""

import os
import sys
import numpy as np

user_site = os.path.expanduser("~/Library/Python/3.9/lib/python/site-packages")
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.examples import expert


class OvertakeADASController:
    def __init__(self, cruise_speed_kmh=60.0):
        self.cruise_speed_kmh = cruise_speed_kmh
        self.target_lane_idx = None
        self.cooldown = 0

    def get_action(self, env):
        vehicle = env.agent
        current_speed_kmh = info.get("speed", 0.0) if 'info' in locals() else getattr(vehicle, "speed", 0.0) * 3.6
        
        ref_lanes = vehicle.navigation.current_ref_lanes
        current_lane_idx = vehicle.lane_index[2] if hasattr(vehicle, "lane_index") and len(vehicle.lane_index) == 3 else 0
        
        if self.target_lane_idx is None:
            self.target_lane_idx = current_lane_idx

        if self.cooldown > 0:
            self.cooldown -= 1

        # 1. Detect lead vehicle in current target lane & check adjacent lane clearance
        lead_dist = 999.0
        left_lane_clear = True
        right_lane_clear = True

        surrounding_objects = vehicle.lidar.get_surrounding_objects(vehicle)
        heading = getattr(vehicle, "heading_theta", 0.0)
        cos_h, sin_h = np.cos(-heading), np.sin(-heading)

        target_lane = ref_lanes[self.target_lane_idx] if (ref_lanes and self.target_lane_idx < len(ref_lanes)) else vehicle.lane

        for obj in surrounding_objects:
            dx_w = obj.position[0] - vehicle.position[0]
            dy_w = obj.position[1] - vehicle.position[1]
            dx = dx_w * cos_h - dy_w * sin_h  # lateral (+ is left)
            dy = dx_w * sin_h + dy_w * cos_h  # longitudinal (+ is forward)

            # Check lead vehicle in current target corridor
            if dy > 0 and abs(dx) < 2.2:
                if dy < lead_dist:
                    lead_dist = dy

            # Check adjacent lane clearance (-15m to +25m zone)
            if -15.0 < dy < 25.0:
                if 1.8 < dx < 5.5:
                    left_lane_clear = False
                elif -5.5 < dx < -1.8:
                    right_lane_clear = False

        # 2. Tactical Overtaking / Lane Change Decision
        can_change_left = (self.target_lane_idx < len(ref_lanes) - 1) and left_lane_clear
        can_change_right = (self.target_lane_idx > 0) and right_lane_clear

        if lead_dist < 20.0 and self.cooldown == 0:
            if can_change_left:
                self.target_lane_idx += 1
                self.cooldown = 40
                print(f" [ADAS Overtake] Slow vehicle at {lead_dist:.1f}m -> Changing LEFT to lane {self.target_lane_idx}")
            elif can_change_right:
                self.target_lane_idx -= 1
                self.cooldown = 40
                print(f" [ADAS Overtake] Slow vehicle at {lead_dist:.1f}m -> Changing RIGHT to lane {self.target_lane_idx}")
            else:
                print(f" [ADAS Following] Slow vehicle at {lead_dist:.1f}m -> No lane available! Braking and maintaining follow distance.")

        # 3. Steering Control (Lane Centering on Target Lane)
        if self.target_lane_idx == current_lane_idx:
            # When aligned in lane, use expert steering for smooth tracking
            base_act = expert(vehicle)
            steering_action = float(base_act[0])
        else:
            # Smooth lane-change steering to target lane center line
            target_lane = ref_lanes[min(self.target_lane_idx, len(ref_lanes) - 1)]
            _, lat_error = target_lane.local_coordinates(vehicle.position)
            heading_error = vehicle.heading_diff(target_lane)
            steering_action = float(np.clip(-0.35 * lat_error - 0.8 * heading_error, -1.0, 1.0))
            if abs(lat_error) < 0.3:  # Completed lane change
                self.target_lane_idx = current_lane_idx

        # 4. Speed & Throttle Control (ACC & AEB)
        target_speed = self.cruise_speed_kmh
        if lead_dist < 22.0:
            # Slow down to match lead vehicle or brake
            target_speed = max(0.0, (lead_dist - 4.0) * 2.5)

        speed_error = target_speed - current_speed_kmh
        throttle_action = float(np.clip(speed_error * 0.15, -1.0, 1.0))

        if lead_dist < 4.5:
            throttle_action = -1.0  # Emergency brake

        return [steering_action, throttle_action], lead_dist


if __name__ == "__main__":
    env = MetaDriveEnv({"use_render": False, "traffic_density": 0.12, "map": "SSSS", "num_scenarios": 5, "start_seed": 2000})
    for ep in range(5):
        obs, info = env.reset()
        ctrl = OvertakeADASController(cruise_speed_kmh=60.0)
        for step in range(1000):
            action, dist = ctrl.get_action(env)
            obs, r, term, trunc, info = env.step(action)
            if term or trunc:
                status = "ARRIVED DESTINATION!" if info.get("arrive_dest") else ("Crashed" if info.get("crash") else "Ended")
                print(f"Episode {ep + 1} (Seed {env.current_seed}): {status} at step {step + 1}")
                break
    env.close()
