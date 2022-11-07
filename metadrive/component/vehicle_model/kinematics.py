import math
from typing import List

import numpy as np

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.manager.traffic_manager import PGTrafficManager
from metadrive.utils import get_np_random, random_string
from metadrive.utils.utils import deprecation_warning


class Vehicle:
    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    COLLISIONS_ENABLED = True
    """ Enable collision detection between vehicles """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_SPEEDS = [23, 25]
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 40.
    """ Maximum reachable speed [m/s] """
    def __init__(
        self,
        traffic_mgr: PGTrafficManager,
        position: List,
        heading: float = 0,
        speed: float = 0,
        np_random: np.random.RandomState = None,
        name: str = None
    ):
        raise DeprecationWarning("We don't use kinematics from highway now")
        deprecation_warning("Vehicle", "Policy Class", error=False)

        self.name = random_string() if name is None else name
        self.traffic_mgr = traffic_mgr
        self._position = np.array(position).astype('float')
        self.heading = heading
        self.speed = speed
        # self.lane_index, _ = self.traffic_mgr.current_map.road_network.get_closest_lane_index(
        #     self.position
        # ) if self.traffic_mgr else (np.nan, np.nan)
        # self.lane = self.traffic_mgr.current_map.road_network.get_lane(self.lane_index) if self.traffic_mgr else None
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        # self.log = []
        # self.history = deque(maxlen=30)
        self.np_random = np_random if np_random else get_np_random()

    # def update_lane_index(self, lane_index, lane):
    #     raise ValueError("Deprecated!")
    #     self.lane_index = lane_index
    #     self.lane = lane

    @property
    def position(self):
        if self._position is None:
            return np.array([np.nan, np.nan])
        else:
            return self._position.copy()

    def set_position(self, pos):
        self._position = np.asarray(pos).copy()

    # @classmethod
    # def make_on_lane(cls, traffic_manager: TrafficManager, lane_index: LaneIndex, longitudinal: float, speed: float = 0):
    #     """
    #     Create a vehicle on a given lane at a longitudinal position.
    #
    #     :param traffic_manager: the road where the vehicle is driving
    #     :param lane_index: index of the lane where the vehicle is located
    #     :param longitudinal: longitudinal position along the lane
    #     :param speed: initial speed in [m/s]
    #     :return: A vehicle with at the specified position
    #     """
    #     lane = traffic_manager.current_map.road_network.get_lane(lane_index)
    #     if speed is None:
    #         speed = lane.speed_limit
    #     return cls(traffic_manager, lane.position(longitudinal, 0), lane.heading_at(longitudinal), speed)

    @classmethod
    def create_random(
        cls,
        traffic_mgr: PGTrafficManager,
        lane: AbstractLane,
        longitude: float,
        speed: float = None,
        random_seed=None
    ):
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param longitude: the longitude on lane
        :param lane: the lane where the vehicle is spawn
        :param traffic_mgr: the traffic_manager where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :return: A vehicle with random position and/or speed
        """
        if speed is None:
            speed = traffic_mgr.np_random.uniform(Vehicle.DEFAULT_SPEEDS[0], Vehicle.DEFAULT_SPEEDS[1])
        v = cls(
            traffic_mgr,
            list(lane.position(longitude, 0)),
            lane.heading_theta_at(longitude),
            speed,
            # random_seed=get_np_random(random_seed)
            # np_random=get_np_random(random_seed)
        )
        return v

    @classmethod
    def create_from(cls, vehicle: "Vehicle") -> "Vehicle":
        """
        Create a new vehicle from an existing one.

        Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.traffic_mgr, vehicle.position, vehicle.heading, vehicle.speed)
        return v

    # def act(self, action: Union[dict, str] = None) -> None:
    #     """
    #     Store an action to be repeated.

    def step(self, dt: float, action) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        assert isinstance(action, dict)
        # self.action = action
        action = self.clip_actions(action)
        delta_f = action['steering']
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array([math.cos(self.heading + beta), math.sin(self.heading + beta)])

        self._position += v * dt

        self.heading += self.speed * math.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += action['acceleration'] * dt
        # for performance reason,
        # self.on_state_update()

    def clip_actions(self, action) -> None:
        if self.crashed:
            action['steering'] = 0
            action['acceleration'] = -1.0 * self.speed
        action['steering'] = float(action['steering'])
        action['acceleration'] = float(action['acceleration'])
        if self.speed > self.MAX_SPEED:
            action['acceleration'] = min(action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))
        elif self.speed < -self.MAX_SPEED:
            action['acceleration'] = max(action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))
        return action

    @property
    def direction(self) -> np.ndarray:
        return np.array([math.cos(self.heading), math.sin(self.heading)])

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction

    @property
    def heading_theta(self):
        return self.heading

    # def on_state_update(self) -> None:
    #     new_l_index, _ = self.traffic_manager.current_map.road_network.get_closest_lane_index(self.position)
    #     self.lane_index = new_l_index
    #     self.lane = self.traffic_manager.current_map.road_network.get_lane(self.lane_index)

    # def lane_distance_to(self, vehicle: "Vehicle", lane: AbstractLane = None) -> float:
    #     """
    #     Compute the signed distance to another vehicle along a lane.
    #
    #     :param vehicle: the other vehicle
    #     :param lane: a lane
    #     :return: the distance to the other vehicle [m]
    #     """
    #     deprecation_warning(1, 2, True)
    #     if not vehicle:
    #         return np.nan
    #     if not lane:
    #         lane = self.lane
    #     return lane.local_coordinates(vehicle.position)[0] - lane.local_coordinates(self.position)[0]

    # def check_collision(self, other: Union['Vehicle', 'TrafficSign']) -> None:
    #     """
    #     Check for collision with another vehicle.
    #
    #     :param other: the other vehicle or object
    #     """
    #     if self.crashed or other is self:
    #         return
    #
    #     if isinstance(other, Vehicle):
    #         if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED:
    #             return
    #
    #         if self._is_colliding(other):
    #             self.speed = other.speed = min([self.speed, other.speed], key=abs)
    #             self.crashed = other.crashed = True
    #     elif isinstance(other, TrafficSign):
    #         if not self.COLLISIONS_ENABLED:
    #             return
    #
    #         if self._is_colliding(other):
    #             self.speed = min([self.speed, 0], key=abs)
    #             self.crashed = other.hit = True
    #     elif isinstance(other, TrafficSign):
    #         if self._is_colliding(other):
    #             other.hit = True

    # def _is_colliding(self, other):
    #     # Fast spherical pre-check
    #     if distance_greater(other.position, self.position, self.LENGTH):
    #         return False
    #     # Accurate rectangular check
    #     return utils.rotated_rectangles_intersect(
    #         (self.position, 0.9 * self.LENGTH, 0.9 * self.WIDTH, self.heading),
    #         (other.position, 0.9 * other.LENGTH, 0.9 * other.WIDTH, other.heading)
    #     )

    # @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane = self.traffic_mgr.current_map.road_network.get_lane(self.route[-1])
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    # #
    # @property
    # def destination_direction(self) -> np.ndarray:
    #     if (self.destination != self.position).any():
    #         return (self.destination - self.position) / norm(*(self.destination - self.position))
    #     else:
    #         return np.zeros((2, ))

    # @property
    # def on_road(self) -> bool:
    #     """ Is the vehicle on its current lane, or off-traffic_manager ? """
    #     return self.lane.on_lane(self.position)
    #
    # def front_distance_to(self, other: "Vehicle") -> float:
    #     return self.direction.dot(other.position - self.position)

    def to_dict(self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True) -> dict:
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity[0],
            'vy': self.velocity[1],
            'cos_h': self.direction[0],
            'sin_h': self.direction[1],
            # 'cos_d': self.destination_direction[0],
            # 'sin_d': self.destination_direction[1]
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    def destroy(self):
        self.traffic_mgr = None
        self._position = None
        self.heading = None
        self.speed = None
        self.lane_index = None
        self.lane = None
        self.action = None
        self.crashed = False
        self.log = None
        self.history = None
        self.np_random = None

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()
