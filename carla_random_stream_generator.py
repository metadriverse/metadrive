"""
CARLA Real-World Deviance Induction Stream Generator.
Injects scenario-aware, physically-bounded disturbances onto CARLA Autopilot reference streams.

Deviance modes adapt to the current driving context:
  - Straights:      crosswind drift, minor throttle lag, micro-steering wobble
  - Curves:         centrifugal understeer/oversteer, apex cutting, lateral-G induced drift
  - Block Turns:    turn-entry hesitation, mid-turn understeer, exit overshoot
  - Intersections:  stop-line creep, throttle hesitation, lane-marking dropout
  - Lead Vehicle:   delayed braking reaction, following-distance oscillation

Uses Ornstein-Uhlenbeck (mean-reverting correlated drift) as the core noise model,
with scenario-specific gain scheduling to keep disturbances realistically bounded.
"""

import copy
import random
import json
import math
from typing import List, Tuple, Optional
import numpy as np
from carla_schema import CarlaFrame


class OrnsteinUhlenbeckProcess:
    """
    Mean-reverting stochastic process for smooth, time-correlated real-world drift.
    dx_t = theta * (mu - x_t) * dt + sigma * dW_t
    """
    def __init__(self, theta: float = 0.15, mu: float = 0.0, sigma: float = 0.08, dt: float = 0.05):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.state = 0.0

    def reset(self, initial_state: float = 0.0):
        self.state = initial_state

    def step(self) -> float:
        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal()
        self.state += dx
        return self.state


class CarlaDevianceInductionGenerator:
    """
    Induces realistic, scenario-aware deviances on CARLA simulation streams.
    Noise intensity and character adapt to the current road geometry and driving context.
    """

    def __init__(self, seed: Optional[int] = 42):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Independent OU processes for each disturbance channel
        self.steer_drift = OrnsteinUhlenbeckProcess(theta=0.20, sigma=0.04)
        self.lane_drift = OrnsteinUhlenbeckProcess(theta=0.10, sigma=0.06)
        self.speed_drift = OrnsteinUhlenbeckProcess(theta=0.15, sigma=0.25)
        # Additional processes for curve/turn-specific phenomena
        self.centrifugal_drift = OrnsteinUhlenbeckProcess(theta=0.12, sigma=0.10)
        self.hesitation_drift = OrnsteinUhlenbeckProcess(theta=0.25, sigma=0.15)

    def _detect_scenario(self, frame: CarlaFrame) -> str:
        """
        Infers the current driving scenario from frame telemetry.
        Uses steering magnitude, angular velocity, speed, and traffic light state.
        """
        abs_steer = abs(frame.control.steer)
        abs_wz = abs(frame.ego_state.angular_velocity.wz)
        speed_kmh = frame.ego_state.velocity.speed_kmh
        tl = frame.environment_context.traffic_light_state
        lead_dist = frame.environment_context.distance_to_lead_vehicle_m

        # Sharp turn detection: high steer + low speed
        if abs_steer > 0.15 and speed_kmh < 30.0:
            return "turn"

        # Curve detection: moderate steer + moderate angular velocity
        if abs_steer > 0.03 or abs_wz > 0.02:
            return "curve"

        # Intersection: traffic light active or very low speed near stop
        if tl in ("Red", "Yellow") or (speed_kmh < 5.0 and frame.control.brake > 0.1):
            return "intersection"

        # Lead vehicle following
        if 0 < lead_dist < 40.0:
            return "following"

        return "straight"

    def _induce_straight_deviance(
        self, frame: CarlaFrame, s_bias: float, l_bias: float, v_bias: float
    ) -> CarlaFrame:
        """
        Straight-road deviance: crosswind / road-camber lateral drift,
        minor throttle fluctuation, micro-steering wobble.
        """
        deviated = CarlaFrame.from_dict(copy.deepcopy(frame.to_dict()))

        # Subtle steering wobble
        steer_pert = float(np.clip(s_bias * 0.3 + np.random.normal(0.0, 0.012), -0.10, 0.10))
        deviated.control.steer = round(float(np.clip(deviated.control.steer + steer_pert, -1.0, 1.0)), 3)

        # Minor throttle fluctuation
        throttle_pert = float(np.clip(np.random.normal(0.0, 0.025), -0.06, 0.06))
        deviated.control.throttle = round(float(np.clip(deviated.control.throttle + throttle_pert, 0.0, 1.0)), 3)

        # Speed variance (+/- ~3 km/h max)
        speed_delta = float(np.clip(v_bias * 0.4 + np.random.normal(0.0, 0.08), -0.9, 0.9))
        new_speed = max(0.0, deviated.ego_state.velocity.speed_mps + speed_delta)
        deviated.ego_state.velocity.speed_mps = round(new_speed, 3)
        deviated.ego_state.velocity.speed_kmh = round(new_speed * 3.6, 2)

        # Crosswind / camber lateral drift
        lane_pert = float(np.clip(l_bias * 1.5 + np.random.normal(0.0, 0.015), -0.35, 0.35))
        deviated.environment_context.lane_offset_meters = round(
            frame.environment_context.lane_offset_meters + lane_pert, 3
        )

        # Yaw consistent with drift
        yaw_pert = float(np.clip(lane_pert * 3.0, -3.5, 3.5))
        deviated.ego_state.transform.rotation.yaw = round(
            deviated.ego_state.transform.rotation.yaw + yaw_pert, 2
        )

        return deviated

    def _induce_curve_deviance(
        self, frame: CarlaFrame, s_bias: float, l_bias: float, v_bias: float, cf_bias: float
    ) -> CarlaFrame:
        """
        Curve deviance:
        - Centrifugal push: lateral drift toward outside of curve (proportional to v^2 * curvature)
        - Understeer/oversteer: steering phase lag
        - Apex cutting: inward drift when speed is too low
        - Speed mismatch entering curves
        """
        deviated = CarlaFrame.from_dict(copy.deepcopy(frame.to_dict()))
        speed_mps = deviated.ego_state.velocity.speed_mps
        wz = deviated.ego_state.angular_velocity.wz  # current turn rate

        # Centrifugal outward push: F_centrifugal ∝ v^2 * curvature
        # Approximate curvature from angular velocity / speed
        curvature_est = abs(wz) / max(0.5, speed_mps)
        centrifugal_offset = float(np.clip(
            cf_bias * 0.8 + speed_mps ** 2 * curvature_est * 0.003 * np.sign(wz),
            -0.40, 0.40
        ))

        # Steering delay / phase lag (understeer)
        steer_lag = float(np.clip(s_bias * 0.5 + np.random.normal(0.0, 0.02), -0.12, 0.12))
        # On curves, the deviance biases toward understeer (lagging behind required steer)
        understeer_bias = -np.sign(frame.control.steer) * 0.025
        deviated.control.steer = round(float(np.clip(
            deviated.control.steer + steer_lag + understeer_bias, -1.0, 1.0
        )), 3)

        # Speed: tendency to enter curves too fast
        speed_delta = float(np.clip(v_bias * 0.3 + np.random.normal(0.0, 0.15), -1.0, 1.5))
        new_speed = max(0.0, speed_mps + speed_delta)
        deviated.ego_state.velocity.speed_mps = round(new_speed, 3)
        deviated.ego_state.velocity.speed_kmh = round(new_speed * 3.6, 2)

        # Lateral offset: combine centrifugal push + normal drift
        lane_pert = float(np.clip(
            l_bias * 1.2 + centrifugal_offset + np.random.normal(0.0, 0.02),
            -0.50, 0.50
        ))
        deviated.environment_context.lane_offset_meters = round(
            frame.environment_context.lane_offset_meters + lane_pert, 3
        )

        # Yaw perturbation consistent with steering lag
        yaw_pert = float(np.clip((steer_lag + centrifugal_offset * 0.5) * 5.0, -6.0, 6.0))
        deviated.ego_state.transform.rotation.yaw = round(
            deviated.ego_state.transform.rotation.yaw + yaw_pert, 2
        )

        if abs(deviated.environment_context.lane_offset_meters) > 0.45:
            deviated.environment_context.collision_warning = True

        return deviated

    def _induce_turn_deviance(
        self, frame: CarlaFrame, s_bias: float, l_bias: float, v_bias: float, h_bias: float
    ) -> CarlaFrame:
        """
        Block-turn deviance:
        - Entry hesitation: deceleration delay before the turn
        - Mid-turn understeer: insufficient steering input
        - Exit overshoot: over-correction on turn exit
        - Lane tracking dropout: brief loss of lane reference in intersection box
        """
        deviated = CarlaFrame.from_dict(copy.deepcopy(frame.to_dict()))

        # Steering deviance: larger envelope for turns
        steer_pert = float(np.clip(
            s_bias * 0.6 + h_bias * 0.3 + np.random.normal(0.0, 0.03),
            -0.18, 0.18
        ))
        deviated.control.steer = round(float(np.clip(
            deviated.control.steer + steer_pert, -1.0, 1.0
        )), 3)

        # Hesitation: throttle drop or brake flutter
        if abs(h_bias) > 0.1:
            throttle_drop = float(np.clip(-abs(h_bias) * 0.15, -0.10, 0.0))
            deviated.control.throttle = round(float(np.clip(
                deviated.control.throttle + throttle_drop, 0.0, 1.0
            )), 3)
            # Phantom light brake tap
            if np.random.rand() < 0.20:
                deviated.control.brake = round(float(np.clip(
                    deviated.control.brake + abs(h_bias) * 0.08, 0.0, 0.3
                )), 3)

        # Speed variance at turns
        speed_delta = float(np.clip(v_bias * 0.25 + np.random.normal(0.0, 0.12), -0.8, 0.8))
        new_speed = max(0.0, deviated.ego_state.velocity.speed_mps + speed_delta)
        deviated.ego_state.velocity.speed_mps = round(new_speed, 3)
        deviated.ego_state.velocity.speed_kmh = round(new_speed * 3.6, 2)

        # Lateral offset: wider drift envelope at intersections where lane markings are absent
        lane_pert = float(np.clip(l_bias * 2.5 + np.random.normal(0.0, 0.03), -0.55, 0.55))
        deviated.environment_context.lane_offset_meters = round(
            frame.environment_context.lane_offset_meters + lane_pert, 3
        )

        yaw_pert = float(np.clip(steer_pert * 6.0, -8.0, 8.0))
        deviated.ego_state.transform.rotation.yaw = round(
            deviated.ego_state.transform.rotation.yaw + yaw_pert, 2
        )

        if abs(deviated.environment_context.lane_offset_meters) > 0.45:
            deviated.environment_context.collision_warning = True

        return deviated

    def _induce_intersection_deviance(
        self, frame: CarlaFrame, s_bias: float, l_bias: float, v_bias: float, h_bias: float
    ) -> CarlaFrame:
        """
        Intersection deviance:
        - Stop-line creep: inching forward after initial stop
        - Throttle hesitation on green: delayed acceleration response
        - Brake timing error: stopping too early or too late
        """
        deviated = CarlaFrame.from_dict(copy.deepcopy(frame.to_dict()))
        tl = frame.environment_context.traffic_light_state
        speed_mps = deviated.ego_state.velocity.speed_mps

        if tl == "Red":
            # Stop-line creep: tiny forward inching
            creep_throttle = float(np.clip(abs(h_bias) * 0.04, 0.0, 0.05))
            if speed_mps < 1.0:
                deviated.control.throttle = round(creep_throttle, 3)
                deviated.control.brake = round(float(np.clip(
                    deviated.control.brake - 0.05, 0.0, 1.0
                )), 3)
            # Brake timing error
            brake_pert = float(np.clip(np.random.normal(0.0, 0.04), -0.08, 0.08))
            deviated.control.brake = round(float(np.clip(
                deviated.control.brake + brake_pert, 0.0, 1.0
            )), 3)
        elif tl == "Green":
            # Hesitation on green: delayed throttle ramp-up
            throttle_delay = float(np.clip(h_bias * 0.2, -0.12, 0.0))
            deviated.control.throttle = round(float(np.clip(
                deviated.control.throttle + throttle_delay, 0.0, 1.0
            )), 3)

        # Minor lateral drift at intersection (no lane markings to follow)
        lane_pert = float(np.clip(l_bias * 1.8 + np.random.normal(0.0, 0.02), -0.40, 0.40))
        deviated.environment_context.lane_offset_meters = round(
            frame.environment_context.lane_offset_meters + lane_pert, 3
        )

        # Speed variance
        speed_delta = float(np.clip(v_bias * 0.2 + np.random.normal(0.0, 0.06), -0.5, 0.5))
        new_speed = max(0.0, speed_mps + speed_delta)
        deviated.ego_state.velocity.speed_mps = round(new_speed, 3)
        deviated.ego_state.velocity.speed_kmh = round(new_speed * 3.6, 2)

        steer_pert = float(np.clip(s_bias * 0.25 + np.random.normal(0.0, 0.01), -0.08, 0.08))
        deviated.control.steer = round(float(np.clip(
            deviated.control.steer + steer_pert, -1.0, 1.0
        )), 3)

        yaw_pert = float(np.clip(lane_pert * 3.5, -4.0, 4.0))
        deviated.ego_state.transform.rotation.yaw = round(
            deviated.ego_state.transform.rotation.yaw + yaw_pert, 2
        )

        return deviated

    def _induce_following_deviance(
        self, frame: CarlaFrame, s_bias: float, l_bias: float, v_bias: float
    ) -> CarlaFrame:
        """
        Lead-vehicle following deviance:
        - Delayed braking reaction
        - Following distance oscillation
        - Throttle overshoot when lead accelerates
        """
        deviated = CarlaFrame.from_dict(copy.deepcopy(frame.to_dict()))

        # Brake reaction delay: late deceleration when lead is close
        lead_dist = deviated.environment_context.distance_to_lead_vehicle_m
        if 0 < lead_dist < 25.0:
            # Delayed brake: reduced braking force
            brake_lag = float(np.clip(np.random.normal(0.0, 0.05), -0.10, 0.10))
            deviated.control.brake = round(float(np.clip(
                deviated.control.brake + brake_lag, 0.0, 1.0
            )), 3)
            # Throttle overshoot if not braking enough
            if deviated.control.brake < 0.1:
                throttle_over = float(np.clip(v_bias * 0.15, 0.0, 0.08))
                deviated.control.throttle = round(float(np.clip(
                    deviated.control.throttle + throttle_over, 0.0, 1.0
                )), 3)

        # Straight-like lateral drift (on the road behind lead vehicle)
        steer_pert = float(np.clip(s_bias * 0.25 + np.random.normal(0.0, 0.01), -0.08, 0.08))
        deviated.control.steer = round(float(np.clip(deviated.control.steer + steer_pert, -1.0, 1.0)), 3)

        lane_pert = float(np.clip(l_bias * 1.2 + np.random.normal(0.0, 0.015), -0.30, 0.30))
        deviated.environment_context.lane_offset_meters = round(
            frame.environment_context.lane_offset_meters + lane_pert, 3
        )

        speed_delta = float(np.clip(v_bias * 0.3 + np.random.normal(0.0, 0.08), -0.7, 0.7))
        new_speed = max(0.0, deviated.ego_state.velocity.speed_mps + speed_delta)
        deviated.ego_state.velocity.speed_mps = round(new_speed, 3)
        deviated.ego_state.velocity.speed_kmh = round(new_speed * 3.6, 2)

        yaw_pert = float(np.clip(lane_pert * 3.0, -3.5, 3.5))
        deviated.ego_state.transform.rotation.yaw = round(
            deviated.ego_state.transform.rotation.yaw + yaw_pert, 2
        )

        return deviated

    # ---- public API ----

    def induce_deviance(
        self,
        reference_frame: CarlaFrame,
        steer_bias: float = 0.0,
        lane_bias: float = 0.0,
        speed_bias: float = 0.0,
        centrifugal_bias: float = 0.0,
        hesitation_bias: float = 0.0,
    ) -> CarlaFrame:
        """
        Applies scenario-aware deviance to a reference frame.
        Automatically detects current driving context and applies appropriate noise model.
        """
        scenario = self._detect_scenario(reference_frame)

        if scenario == "curve":
            return self._induce_curve_deviance(
                reference_frame, steer_bias, lane_bias, speed_bias, centrifugal_bias
            )
        elif scenario == "turn":
            return self._induce_turn_deviance(
                reference_frame, steer_bias, lane_bias, speed_bias, hesitation_bias
            )
        elif scenario == "intersection":
            return self._induce_intersection_deviance(
                reference_frame, steer_bias, lane_bias, speed_bias, hesitation_bias
            )
        elif scenario == "following":
            return self._induce_following_deviance(
                reference_frame, steer_bias, lane_bias, speed_bias
            )
        else:  # straight
            return self._induce_straight_deviance(
                reference_frame, steer_bias, lane_bias, speed_bias
            )

    def generate_deviated_sequence(
        self,
        reference_frames: List[CarlaFrame]
    ) -> List[CarlaFrame]:
        """
        Takes a sequence of golden autopilot reference frames and simulates a continuous,
        physically-consistent realistic drive with time-correlated, scenario-aware deviances.
        """
        self.steer_drift.reset()
        self.lane_drift.reset()
        self.speed_drift.reset()
        self.centrifugal_drift.reset()
        self.hesitation_drift.reset()

        deviated_sequence: List[CarlaFrame] = []
        for ref_frame in reference_frames:
            s_bias = self.steer_drift.step()
            l_bias = self.lane_drift.step()
            v_bias = self.speed_drift.step()
            cf_bias = self.centrifugal_drift.step()
            h_bias = self.hesitation_drift.step()

            dev_frame = self.induce_deviance(
                ref_frame,
                steer_bias=s_bias * 0.5,
                lane_bias=l_bias * 2.0,
                speed_bias=v_bias,
                centrifugal_bias=cf_bias,
                hesitation_bias=h_bias,
            )
            deviated_sequence.append(dev_frame)

        return deviated_sequence

    def generate_paired_dataset(
        self,
        reference_frames: List[CarlaFrame],
        variations: int = 6
    ) -> List[Tuple[CarlaFrame, CarlaFrame]]:
        """
        Produces paired (deviated_frame, golden_reference_frame) training examples
        across multiple random re-rolls of the OU processes.
        """
        pairs: List[Tuple[CarlaFrame, CarlaFrame]] = []
        for v in range(variations):
            # Re-seed OU processes with different random state for each variation
            self.steer_drift.reset(np.random.normal(0.0, 0.01))
            self.lane_drift.reset(np.random.normal(0.0, 0.01))
            self.speed_drift.reset(np.random.normal(0.0, 0.05))
            self.centrifugal_drift.reset(np.random.normal(0.0, 0.02))
            self.hesitation_drift.reset(np.random.normal(0.0, 0.03))

            dev_seq = self.generate_deviated_sequence(reference_frames)
            for dev_frame, ref_frame in zip(dev_seq, reference_frames):
                pairs.append((dev_frame, ref_frame))
        return pairs

    def save_to_file(self, frames: List[CarlaFrame], filepath: str) -> None:
        data = [f.to_dict() for f in frames]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(frames)} realistically deviated CARLA stream frames to {filepath}")


# Alias for backward compatibility
CarlaRandomStreamGenerator = CarlaDevianceInductionGenerator


if __name__ == "__main__":
    from carla_autopilot_generator import CarlaAutopilotStreamGenerator
    auto_gen = CarlaAutopilotStreamGenerator()
    ref_frames = auto_gen.generate_city_block_route()
    dev_gen = CarlaDevianceInductionGenerator()
    deviated_stream = dev_gen.generate_deviated_sequence(ref_frames)
    dev_gen.save_to_file(deviated_stream, "carla_random_stream.json")
