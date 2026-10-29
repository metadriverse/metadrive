"""
CARLA Simulation Data Stream Schema and Variable Definitions.
Defines standardized structures for telemetry, ego controls, and environmental context
matching CARLA Autonomous Driving Autopilot / TrafficManager format.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json
import numpy as np


@dataclass
class CarlaVector3D:
    x: float
    y: float
    z: float

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]


@dataclass
class CarlaRotation:
    pitch: float
    yaw: float
    roll: float

    def to_list(self) -> List[float]:
        return [self.pitch, self.yaw, self.roll]


@dataclass
class CarlaTransform:
    location: CarlaVector3D
    rotation: CarlaRotation


@dataclass
class CarlaVelocity:
    vx: float
    vy: float
    vz: float
    speed_mps: float
    speed_kmh: float

    def to_list(self) -> List[float]:
        return [self.vx, self.vy, self.vz, self.speed_mps, self.speed_kmh]


@dataclass
class CarlaAcceleration:
    ax: float
    ay: float
    az: float

    def to_list(self) -> List[float]:
        return [self.ax, self.ay, self.az]


@dataclass
class CarlaAngularVelocity:
    wx: float
    wy: float
    wz: float

    def to_list(self) -> List[float]:
        return [self.wx, self.wy, self.wz]


@dataclass
class CarlaEgoState:
    transform: CarlaTransform
    velocity: CarlaVelocity
    acceleration: CarlaAcceleration
    angular_velocity: CarlaAngularVelocity


@dataclass
class CarlaControl:
    steer: float       # [-1.0, 1.0]
    throttle: float    # [0.0, 1.0]
    brake: float       # [0.0, 1.0]
    hand_brake: bool
    reverse: bool
    gear: int


@dataclass
class CarlaEnvironmentContext:
    speed_limit_kmh: float
    lane_id: int
    road_id: int
    lane_offset_meters: float            # lateral deviation from lane center
    distance_to_lead_vehicle_m: float     # -1.0 if no lead vehicle
    lead_vehicle_speed_kmh: float
    traffic_light_state: str             # "Green", "Yellow", "Red", "None"
    collision_warning: bool


@dataclass
class CarlaWaypoint:
    x: float
    y: float
    z: float
    target_speed_kmh: float


@dataclass
class CarlaFrame:
    frame_id: int
    timestamp: float
    delta_seconds: float
    ego_state: CarlaEgoState
    control: CarlaControl
    environment_context: CarlaEnvironmentContext
    target_waypoint: CarlaWaypoint

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CarlaFrame":
        loc = CarlaVector3D(**data["ego_state"]["transform"]["location"])
        rot = CarlaRotation(**data["ego_state"]["transform"]["rotation"])
        transform = CarlaTransform(location=loc, rotation=rot)

        vel = CarlaVelocity(**data["ego_state"]["velocity"])
        acc = CarlaAcceleration(**data["ego_state"]["acceleration"])
        ang_vel = CarlaAngularVelocity(**data["ego_state"]["angular_velocity"])
        ego_state = CarlaEgoState(
            transform=transform,
            velocity=vel,
            acceleration=acc,
            angular_velocity=ang_vel
        )

        ctrl = CarlaControl(**data["control"])
        env = CarlaEnvironmentContext(**data["environment_context"])
        wp = CarlaWaypoint(**data["target_waypoint"])

        return cls(
            frame_id=data["frame_id"],
            timestamp=data["timestamp"],
            delta_seconds=data["delta_seconds"],
            ego_state=ego_state,
            control=ctrl,
            environment_context=env,
            target_waypoint=wp
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CarlaFrame":
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_feature_vector(self) -> np.ndarray:
        """
        Extracts numerical state and context representation for neural network consumption.
        Dimension: 18 features
        """
        ego = self.ego_state
        env = self.environment_context
        ctrl = self.control

        yaw_rad = np.radians(ego.transform.rotation.yaw)
        tl_red = 1.0 if env.traffic_light_state == "Red" else 0.0
        tl_yellow = 1.0 if env.traffic_light_state == "Yellow" else 0.0

        lead_dist = env.distance_to_lead_vehicle_m if env.distance_to_lead_vehicle_m >= 0 else 100.0

        vec = [
            ego.transform.location.x,
            ego.transform.location.y,
            yaw_rad,
            ego.velocity.vx,
            ego.velocity.vy,
            ego.velocity.speed_mps,
            ego.acceleration.ax,
            ego.acceleration.ay,
            ego.angular_velocity.wz,
            ctrl.steer,
            ctrl.throttle,
            ctrl.brake,
            env.lane_offset_meters,
            env.speed_limit_kmh / 3.6,
            lead_dist,
            env.lead_vehicle_speed_kmh / 3.6,
            tl_red,
            tl_yellow
        ]
        return np.array(vec, dtype=np.float32)

    def extract_action_vector(self) -> np.ndarray:
        """Extracts control actions [steer, throttle, brake]."""
        return np.array([
            self.control.steer,
            self.control.throttle,
            self.control.brake
        ], dtype=np.float32)

    def apply_action_vector(self, actions: np.ndarray) -> None:
        """Updates ego control in-place from action vector."""
        self.control.steer = float(np.clip(actions[0], -1.0, 1.0))
        self.control.throttle = float(np.clip(actions[1], 0.0, 1.0))
        self.control.brake = float(np.clip(actions[2], 0.0, 1.0))
