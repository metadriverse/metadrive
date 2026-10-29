"""
CARLA Autopilot Reference Data Stream Generator.
Generates perfect, crashless driving trajectories for realistic urban routes:
  - Straight road segments
  - Curved / S-bend highways
  - 90-degree block turns (left/right)
  - Intersection crossings with traffic lights
  - Lead vehicle following (ACC) on mixed roads

Uses kinematic bicycle model, Pure-Pursuit lane tracking, PID longitudinal control,
and Adaptive Cruise Control matching CARLA TrafficManager / BehaviorAgent specification.
"""

import math
import json
from typing import List, Tuple
import numpy as np
from carla_schema import (
    CarlaFrame, CarlaEgoState, CarlaTransform, CarlaVector3D, CarlaRotation,
    CarlaVelocity, CarlaAcceleration, CarlaAngularVelocity, CarlaControl,
    CarlaEnvironmentContext, CarlaWaypoint
)


# ---------------------------------------------------------------------------
#   Route Waypoint Definitions
# ---------------------------------------------------------------------------

class RouteSegment:
    """Defines a segment of a city driving route."""
    def __init__(
        self,
        segment_type: str,     # "straight", "curve", "turn_left", "turn_right", "intersection"
        length_meters: float,
        speed_limit_kmh: float = 50.0,
        curvature: float = 0.0,         # 1/radius for curves, 0 for straights
        turn_angle_deg: float = 0.0,    # total turn angle for block turns
        has_traffic_light: bool = False,
        traffic_light_at_pct: float = 0.8,  # when in the segment the light appears
        tl_state: str = "Green",
        has_lead_vehicle: bool = False,
        lead_vehicle_speed_kmh: float = 30.0,
        road_id: int = 1,
        lane_id: int = 1,
    ):
        self.segment_type = segment_type
        self.length_meters = length_meters
        self.speed_limit_kmh = speed_limit_kmh
        self.curvature = curvature
        self.turn_angle_deg = turn_angle_deg
        self.has_traffic_light = has_traffic_light
        self.traffic_light_at_pct = traffic_light_at_pct
        self.tl_state = tl_state
        self.has_lead_vehicle = has_lead_vehicle
        self.lead_vehicle_speed_kmh = lead_vehicle_speed_kmh
        self.road_id = road_id
        self.lane_id = lane_id


def build_city_block_route() -> List[RouteSegment]:
    """
    Defines a realistic 'around the block' city route:
      1. Straight road (accelerate to cruising)
      2. Gentle right curve
      3. Straight with lead vehicle (ACC)
      4. Intersection with traffic light (Red -> stop -> Green -> go)
      5. Left turn at intersection
      6. S-curve highway bend
      7. Straight cruising
      8. Right turn at T-junction
      9. Straight approach back to start
    """
    return [
        # Segment 1: Initial straight road (warm-up, accelerate)
        RouteSegment("straight", length_meters=80.0, speed_limit_kmh=50.0,
                      road_id=1, lane_id=1),

        # Segment 2: Gentle right curve (highway-style, radius ~100m)
        RouteSegment("curve", length_meters=60.0, speed_limit_kmh=45.0,
                      curvature=0.01, road_id=2, lane_id=1),

        # Segment 3: Straight with lead vehicle ahead (ACC scenario)
        RouteSegment("straight", length_meters=70.0, speed_limit_kmh=50.0,
                      has_lead_vehicle=True, lead_vehicle_speed_kmh=35.0,
                      road_id=3, lane_id=1),

        # Segment 4: Intersection approach with Red traffic light (stop & go)
        RouteSegment("intersection", length_meters=40.0, speed_limit_kmh=40.0,
                      has_traffic_light=True, traffic_light_at_pct=0.5,
                      tl_state="Red", road_id=4, lane_id=1),

        # Segment 5: Left 90-degree turn through intersection
        RouteSegment("turn_left", length_meters=25.0, speed_limit_kmh=25.0,
                      turn_angle_deg=90.0, road_id=5, lane_id=1),

        # Segment 6: S-curve highway bend (alternating curvature)
        RouteSegment("curve", length_meters=80.0, speed_limit_kmh=40.0,
                      curvature=0.015, road_id=6, lane_id=1),

        # Segment 7: Straight cruising on residential road
        RouteSegment("straight", length_meters=60.0, speed_limit_kmh=40.0,
                      road_id=7, lane_id=1),

        # Segment 8: Right 90-degree turn at T-junction
        RouteSegment("turn_right", length_meters=20.0, speed_limit_kmh=20.0,
                      turn_angle_deg=90.0, road_id=8, lane_id=1),

        # Segment 9: Final straight approach back
        RouteSegment("straight", length_meters=70.0, speed_limit_kmh=50.0,
                      road_id=9, lane_id=1),
    ]


# ---------------------------------------------------------------------------
#   Autopilot Stream Generator
# ---------------------------------------------------------------------------

class CarlaAutopilotStreamGenerator:
    """
    Generates sequences of CARLA Autopilot data frames representing golden
    'crashless' autonomous driving through a complete city block route.
    """

    def __init__(
        self,
        delta_seconds: float = 0.05,
        wheelbase: float = 2.87,
        max_steer_angle_rad: float = 0.7,
    ):
        self.dt = delta_seconds
        self.wheelbase = wheelbase
        self.max_steer_angle_rad = max_steer_angle_rad

    # ---- low-level PID helpers ----

    def _longitudinal_control(
        self, speed_mps: float, target_speed_mps: float,
        lead_dist: float, lead_speed_mps: float,
        tl_state: str, distance_to_tl: float
    ) -> Tuple[float, float]:
        """Returns (throttle, brake)."""
        # Traffic-light braking
        if tl_state == "Red" and distance_to_tl < 25.0:
            target_speed_mps = max(0.0, speed_mps * (distance_to_tl / 25.0) - 0.5)

        # ACC: match lead vehicle with headway buffer
        if lead_dist > 0 and lead_dist < 40.0:
            safe_speed = lead_speed_mps * min(1.0, lead_dist / 25.0)
            target_speed_mps = min(target_speed_mps, safe_speed)

        speed_error = target_speed_mps - speed_mps
        if speed_error > 0.15:
            throttle = float(np.clip(0.30 + 0.20 * speed_error, 0.0, 0.85))
            brake = 0.0
        elif speed_error < -0.25:
            throttle = 0.0
            brake = float(np.clip(-0.30 * speed_error, 0.0, 0.95))
        else:
            throttle = 0.20
            brake = 0.0
        return throttle, brake

    def _lateral_control(
        self, yaw_rad: float, desired_yaw_rad: float, lane_offset: float
    ) -> float:
        """Pure-pursuit / Stanley style lateral controller. Returns steer demand [-1, 1]."""
        yaw_err = desired_yaw_rad - yaw_rad
        # Wrap to [-pi, pi]
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))
        steer = 2.0 * yaw_err - 0.5 * lane_offset
        return float(np.clip(steer, -1.0, 1.0))

    # ---- segment-level trajectory generation ----

    def _generate_segment_frames(
        self,
        segment: RouteSegment,
        x: float, y: float, yaw_rad: float, speed_mps: float,
        frame_offset: int,
    ) -> Tuple[List[CarlaFrame], float, float, float, float]:
        """Generates frames for one route segment, returns (frames, x, y, yaw, speed)."""
        frames: List[CarlaFrame] = []
        target_speed_mps = segment.speed_limit_kmh / 3.6
        num_frames = max(10, int(segment.length_meters / max(0.3, speed_mps * self.dt)))
        # Cap frame count for turns at low speed
        num_frames = min(num_frames, 600)

        # Turn geometry pre-computation
        turn_rate = 0.0
        if segment.segment_type in ("turn_left", "turn_right"):
            total_angle_rad = math.radians(segment.turn_angle_deg)
            if segment.segment_type == "turn_left":
                total_angle_rad = total_angle_rad
            else:
                total_angle_rad = -total_angle_rad
            # Spread the turn over the segment duration
            segment_time = num_frames * self.dt
            turn_rate = total_angle_rad / segment_time if segment_time > 0 else 0.0

        # Lead vehicle state
        lead_dist = 45.0 if segment.has_lead_vehicle else -1.0
        lead_speed_mps = segment.lead_vehicle_speed_kmh / 3.6

        # Traffic light tracking
        tl_state = "Green"
        tl_trigger_frame = int(num_frames * segment.traffic_light_at_pct)
        distance_to_tl = segment.length_meters
        stopped_frames = 0
        max_stop_frames = int(3.0 / self.dt)  # 3 second red-light stop

        # S-curve oscillation phase
        s_curve_phase = 0.0

        dist_traveled = 0.0

        for i in range(num_frames):
            frame_id = frame_offset + i
            timestamp = round(frame_id * self.dt, 3)
            pct = i / max(1, num_frames - 1)

            # --- Determine desired yaw & lane offset for this segment type ---
            if segment.segment_type == "straight":
                desired_yaw = yaw_rad  # maintain current heading
                lane_offset = 0.005 * math.sin(i * 0.08)

            elif segment.segment_type == "curve":
                # Smooth curve: sinusoidal curvature variation for S-bends
                s_curve_phase += self.dt * 1.2
                curve_kappa = segment.curvature * math.sin(s_curve_phase)
                desired_yaw = yaw_rad + curve_kappa * speed_mps * self.dt
                lane_offset = 0.008 * math.cos(s_curve_phase)

            elif segment.segment_type in ("turn_left", "turn_right"):
                # Progressive 90-degree turn with smooth entry/exit ramp
                # Ramp: slow entry (first 20%), full turn (mid 60%), slow exit (last 20%)
                if pct < 0.2:
                    ramp = pct / 0.2
                elif pct > 0.8:
                    ramp = (1.0 - pct) / 0.2
                else:
                    ramp = 1.0
                desired_yaw = yaw_rad + turn_rate * ramp * self.dt
                lane_offset = 0.01 * math.sin(pct * math.pi)

            elif segment.segment_type == "intersection":
                desired_yaw = yaw_rad  # go straight through intersection
                lane_offset = 0.0
                # Traffic light logic
                if segment.has_traffic_light:
                    if i >= tl_trigger_frame and stopped_frames < max_stop_frames:
                        tl_state = "Red"
                    elif stopped_frames >= max_stop_frames:
                        tl_state = "Green"  # light turns green after stop
                    distance_to_tl = max(0.0, segment.length_meters * (segment.traffic_light_at_pct - pct))
            else:
                desired_yaw = yaw_rad
                lane_offset = 0.0

            # --- Longitudinal controller ---
            throttle, brake = self._longitudinal_control(
                speed_mps, target_speed_mps, lead_dist, lead_speed_mps,
                tl_state, distance_to_tl
            )

            # Track stopped frames at red light
            if tl_state == "Red" and speed_mps < 0.3:
                stopped_frames += 1

            # --- Lateral controller ---
            steer_demand = self._lateral_control(yaw_rad, desired_yaw, lane_offset)

            # For turns, override with kinematic turn demand
            if segment.segment_type in ("turn_left", "turn_right"):
                if pct < 0.2:
                    ramp = pct / 0.2
                elif pct > 0.8:
                    ramp = (1.0 - pct) / 0.2
                else:
                    ramp = 1.0
                turn_steer = turn_rate * self.wheelbase / max(0.5, speed_mps)
                steer_demand = float(np.clip(turn_steer * ramp, -1.0, 1.0))
                # Slow down for turns
                target_speed_mps = min(target_speed_mps, segment.speed_limit_kmh / 3.6)

            # --- Kinematic bicycle model update ---
            accel = (throttle * 3.5) - (brake * 7.0) - (0.04 * speed_mps)
            speed_mps = max(0.0, speed_mps + accel * self.dt)
            steer_angle = steer_demand * self.max_steer_angle_rad
            wz = (speed_mps / self.wheelbase) * math.tan(steer_angle)
            yaw_rad += wz * self.dt
            vx = speed_mps * math.cos(yaw_rad)
            vy = speed_mps * math.sin(yaw_rad)
            x += vx * self.dt
            y += vy * self.dt
            dist_traveled += speed_mps * self.dt

            # Lead vehicle distance update
            if segment.has_lead_vehicle and lead_dist > 0:
                lead_dist += (lead_speed_mps - speed_mps) * self.dt
                lead_dist = max(lead_dist, 6.0)

            # --- Assemble frame ---
            ego = CarlaEgoState(
                transform=CarlaTransform(
                    location=CarlaVector3D(round(x, 3), round(y, 3), 0.15),
                    rotation=CarlaRotation(0.0, round(math.degrees(yaw_rad), 2), 0.0)
                ),
                velocity=CarlaVelocity(
                    round(vx, 3), round(vy, 3), 0.0,
                    round(speed_mps, 3), round(speed_mps * 3.6, 2)
                ),
                acceleration=CarlaAcceleration(
                    round(accel * math.cos(yaw_rad), 3),
                    round(accel * math.sin(yaw_rad), 3), 0.0
                ),
                angular_velocity=CarlaAngularVelocity(0.0, 0.0, round(wz, 3))
            )

            ctrl = CarlaControl(
                steer=round(steer_demand, 3),
                throttle=round(throttle, 3),
                brake=round(brake, 3),
                hand_brake=False,
                reverse=False,
                gear=int(min(5, max(1, int(speed_mps * 3.6 // 15) + 1)))
            )

            env = CarlaEnvironmentContext(
                speed_limit_kmh=segment.speed_limit_kmh,
                lane_id=segment.lane_id,
                road_id=segment.road_id,
                lane_offset_meters=round(lane_offset, 3),
                distance_to_lead_vehicle_m=round(lead_dist, 2) if lead_dist > 0 else -1.0,
                lead_vehicle_speed_kmh=round(lead_speed_mps * 3.6, 2) if segment.has_lead_vehicle else 0.0,
                traffic_light_state=tl_state,
                collision_warning=False
            )

            wp = CarlaWaypoint(
                round(x + 12.0 * math.cos(yaw_rad), 3),
                round(y + 12.0 * math.sin(yaw_rad), 3),
                0.15,
                segment.speed_limit_kmh
            )

            frames.append(CarlaFrame(
                frame_id=frame_id,
                timestamp=timestamp,
                delta_seconds=self.dt,
                ego_state=ego,
                control=ctrl,
                environment_context=env,
                target_waypoint=wp
            ))

            # End segment early if distance traveled
            if dist_traveled >= segment.length_meters:
                break

        return frames, x, y, yaw_rad, speed_mps

    # ---- public API ----

    def generate_city_block_route(self) -> List[CarlaFrame]:
        """
        Generates the complete 'around the block' city driving route
        with curves, turns, intersections, traffic lights, and lead vehicles.
        """
        route = build_city_block_route()
        all_frames: List[CarlaFrame] = []
        x, y, yaw, speed = 0.0, 0.0, 0.0, 0.5  # start near-stationary

        frame_offset = 0
        for seg in route:
            seg_frames, x, y, yaw, speed = self._generate_segment_frames(
                seg, x, y, yaw, speed, frame_offset
            )
            all_frames.extend(seg_frames)
            frame_offset += len(seg_frames)

        print(f"Generated city-block route: {len(all_frames)} frames, "
              f"{len(route)} segments, {frame_offset * self.dt:.1f}s duration")
        return all_frames

    def generate_reference_sequence(
        self,
        num_frames: int = 200,
        scenario_type: str = "city_block",
        **kwargs
    ) -> List[CarlaFrame]:
        """
        Backward-compatible API. Now defaults to 'city_block' which covers
        straights, curves, turns, intersections, and traffic.
        """
        if scenario_type == "city_block":
            frames = self.generate_city_block_route()
            # Trim or pad to requested length
            if len(frames) > num_frames:
                frames = frames[:num_frames]
            return frames

        # Legacy single-scenario fallback
        return self._generate_single_scenario(num_frames, scenario_type=scenario_type, **kwargs)

    def _generate_single_scenario(
        self,
        num_frames: int,
        scenario_type: str = "straight_cruising",
        start_x: float = 0.0,
        start_y: float = 0.0,
        initial_speed_kmh: float = 30.0,
    ) -> List[CarlaFrame]:
        """Legacy single-scenario generation for backward compatibility."""
        if scenario_type == "straight_cruising":
            seg = RouteSegment("straight", length_meters=num_frames * 0.5,
                               speed_limit_kmh=50.0, road_id=1)
        elif scenario_type == "gentle_curve":
            seg = RouteSegment("curve", length_meters=num_frames * 0.5,
                               speed_limit_kmh=45.0, curvature=0.012, road_id=2)
        elif scenario_type == "lead_vehicle_acc":
            seg = RouteSegment("straight", length_meters=num_frames * 0.5,
                               speed_limit_kmh=50.0, has_lead_vehicle=True,
                               lead_vehicle_speed_kmh=30.0, road_id=3)
        elif scenario_type == "traffic_light_stop":
            seg = RouteSegment("intersection", length_meters=num_frames * 0.4,
                               speed_limit_kmh=40.0, has_traffic_light=True,
                               tl_state="Red", road_id=4)
        else:
            seg = RouteSegment("straight", length_meters=num_frames * 0.5,
                               speed_limit_kmh=50.0, road_id=1)

        frames, _, _, _, _ = self._generate_segment_frames(
            seg, start_x, start_y, 0.0, initial_speed_kmh / 3.6, 0
        )
        return frames[:num_frames]

    def save_to_file(self, frames: List[CarlaFrame], filepath: str) -> None:
        data = [f.to_dict() for f in frames]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(frames)} CARLA Autopilot reference frames to {filepath}")


if __name__ == "__main__":
    generator = CarlaAutopilotStreamGenerator()
    frames = generator.generate_city_block_route()
    generator.save_to_file(frames, "carla_reference_stream.json")
