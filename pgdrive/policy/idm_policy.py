import copy
import math
from typing import Tuple

import numpy as np

import pgdrive.utils.math_utils as utils
from pgdrive.component.highway_vehicle.controller import ControlledVehicle, Vehicle
from pgdrive.component.lane.abs_lane import AbstractLane
from pgdrive.component.static_object import BaseStaticObject
# from pgdrive.component.highway_vehicle.kinematics import Vehicle
from pgdrive.component.vehicle.base_vehicle import BaseVehicle
from pgdrive.constants import Route, LaneIndex
from pgdrive.manager.traffic_manager import TrafficManager
from pgdrive.policy.base_policy import BasePolicy
from pgdrive.utils import norm
from pgdrive.engine.engine_utils import get_engine
from pgdrive.utils.math_utils import clip
from pgdrive.utils.scene_utils import ray_localization


class IDMPolicy(BasePolicy):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    STEERING_PARAMETERS = [ControlledVehicle.KP_HEADING, ControlledVehicle.KP_HEADING * ControlledVehicle.KP_LATERAL]

    # Longitudinal policy parameters
    ACC_MAX = 20.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 10.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -10.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 2.0  # []
    """Exponent of the velocity term."""

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]
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

    TAU_A = 0.6  # [s]
    TAU_DS = 0.2  # [s]
    PURSUIT_TAU = 1.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 3 * KP_HEADING  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    def __init__(
        self,
        vehicle: BaseVehicle,
        traffic_manager: TrafficManager,
        # position: List,
        delay_time: float,
        # heading: float = 0,
        # speed: float = 0,
        target_lane_index: int = None,
        target_speed: float = 0.01,
        route: Route = None,
        enable_lane_change: bool = True,
        random_seed=None
        # np_random: np.random.RandomState = None,
    ):
        super().__init__(random_seed=random_seed)
        self.enable_lane_change = enable_lane_change
        self.delay_time = delay_time

        self.traffic_manager = traffic_manager
        # self._position = np.array(position).astype('float')
        # self.heading = heading
        # self.speed = speed
        self.target_speed = target_speed

        self.vehicle = vehicle
        self.lane_index, _ = self.traffic_manager.current_map.road_network.get_closest_lane_index(
            self.vehicle.position
        ) if self.traffic_manager else (np.nan, np.nan)
        self.lane = self.traffic_manager.current_map.road_network.get_lane(
            self.lane_index
        ) if self.traffic_manager else None
        self.target_lane_index = target_lane_index or self.lane_index

        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        # self.history = deque(maxlen=30)
        self.route = route

    # @property
    # def lane_index(self):
    #     TODO(pzh): Please change this by the ray_localization!!!
    # lane_index, _ = self.traffic_manager.current_map.road_network.get_closest_lane_index(
    #     self.vehicle.position
    # ) if self.traffic_manager else (np.nan, np.nan)
    # return lane_index

    # @property
    # def lane(self):
    #     return self.traffic_manager.current_map.road_network.get_lane(self.lane_index) if self.traffic_manager else None

    def update_lane_index(self, lane_index, lane):
        self.lane_index = lane_index
        self.lane = lane

    def before_step(self, vehicle: BaseVehicle, front_vehicle, rear_vehicle, current_map):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.
        """

        # if vehicle.crashed:
        #     return
        # TODO: If vehicle is crashed, then we should return everything.

        action = {}
        # front_vehicle, rear_vehicle = traffic_manager.neighbour_vehicles(self)
        # Lateral: MOBIL
        self.follow_road(current_map)

        if self.enable_lane_change:
            self.change_lane_policy()

        action['steering'] = self.steering_control(self.target_lane_index)
        action['steering'] = clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # Longitudinal: IDM
        action['acceleration'] = self.acceleration(ego_vehicle=vehicle, front_vehicle=front_vehicle)

        # TODO(pzh): @LQY why we remove this line?
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])

        action['acceleration'] = clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)

        self.action = action

    def step(self, dt):
        self.delay_time += dt
        if self.action['acceleration'] < 0 and self.speed <= 0:
            self.action['acceleration'] = -self.speed / dt

        # TODO: This is a workaround.
        # self.vehicle.vehicle_node.kinematic_model.step(dt, action=self.action)
        # self.vehicle.step(dt, self.action)
        return copy.copy(self.action)

        # self.clip_actions()
        # delta_f = self.action['steering']
        # beta = np.arctan(1 / 2 * np.tan(delta_f))
        # # v = self.speed * np.array([math.cos(self.heading + beta), math.sin(self.heading + beta)])
        # # self._position += v * dt
        # self.heading += self.speed * math.sin(beta) / (self.LENGTH / 2) * dt
        # self.speed += self.action['acceleration'] * dt

    def after_step(self, *args, **kwargs):
        engine = get_engine()
        dir = np.array([math.cos(self.heading), math.sin(self.heading)])
        lane, lane_index = ray_localization(dir, self.position, engine)
        if lane is not None:
            self.update_lane_index(lane_index, lane)
        # self.lane_index = lane_index
        # self.lane = lane
        # self.out_of_road = not self.lane.on_lane(self.position, margin=2)

    def follow_road(self, current_map):
        """At the end of a lane, automatically switch to a next one."""
        if current_map.road_network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = current_map.road_network.next_lane(
                self.target_lane_index, route=self.route, position=self.position, np_random=self.np_random
            )

    # ============================================+
    # TODO(pzh): Currently, this is the _position of the kinematics vehicle!
    @property
    def position(self):
        return self.vehicle.position

    @property
    def speed(self):
        return self.vehicle.speed

    @property
    def heading(self):
        return self.vehicle.heading

    # ============================================+

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Linear controller with respect to parameters.

        Overrides the non-linear controller ControlledVehicle.steering_control()

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        return float(np.dot(np.array(self.STEERING_PARAMETERS), self.steering_features(target_lane_index)))

    def lane_distance_to(self, vehicle: "Vehicle", lane: AbstractLane = None) -> float:
        # TODO(pzh): This should be a utility function! Instead of a function inside policy!
        #  besides, we copied the same code multiple times! In kinematics and basevehicle! This is not good!
        """
        Compute the signed distance to another vehicle along a lane.

        :param vehicle: the other vehicle
        :param lane: a lane
        :return: the distance to the other vehicle [m]
        """
        if not vehicle:
            return np.nan
        if not lane:
            lane = self.lane
        ret = lane.local_coordinates(vehicle.position)[0] - lane.local_coordinates(self.position)[0]
        if ret == 0:
            print('Fuck you!')
        return ret

    def acceleration(
        # self, ego_vehicle: ControlledVehicle, front_vehicle: Vehicle = None, rear_vehicle: Vehicle = None
        self,
        ego_vehicle,
        front_vehicle
    ) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or isinstance(ego_vehicle, BaseStaticObject):
            return 0

        ego_target_speed = utils.not_zero(self.target_speed)

        # TODO(pzh): We should not required the self.speed! We should get the speed from the vehicle.
        acceleration = self.COMFORT_ACC_MAX * (1 - np.power(max(self.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            # if isinstance(ego_vehicle, TrafficVehicle):
            d = self.lane_distance_to(front_vehicle)

            if d > 0:
                print('1111111111111')

            # else:
            #     d = ego_vehicle.lane_distance_to(front_vehicle)

            acceleration -= self.COMFORT_ACC_MAX * \
                            np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        if acceleration < 0 and self.speed < 0:
            acceleration = -self.speed / 0.2

        if abs(acceleration) > 10_0000:
            print("Your acceleration {} is too large?".format(acceleration))

        return acceleration

    # def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None) -> float:
    def desired_gap(self, ego_vehicle, front_vehicle) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    # def maximum_speed(self, front_vehicle: Vehicle = None) -> Tuple[float, float]:
    def maximum_speed(self, front_vehicle) -> Tuple[float, float]:
        """
        Compute the maximum allowed speed to avoid Inevitable Collision States.

        Assume the front vehicle is going to brake at full deceleration and that
        it will be noticed after a given delay, and compute the maximum speed
        which allows the ego-vehicle to brake enough to avoid the collision.

        :param front_vehicle: the preceding vehicle
        :return: the maximum allowed speed, and suggested acceleration
        """
        if not front_vehicle:
            return self.target_speed
        d0 = self.DISTANCE_WANTED
        a0 = self.COMFORT_ACC_MIN
        a1 = self.COMFORT_ACC_MIN
        tau = self.TIME_WANTED
        d = max(self.lane_distance_to(front_vehicle) - self.LENGTH / 2 - front_vehicle.LENGTH / 2 - d0, 0)
        v1_0 = front_vehicle.speed
        delta = 4 * (a0 * a1 * tau)**2 + 8 * a0 * (a1**2) * d + 4 * a0 * a1 * v1_0**2
        v_max = -a0 * tau + np.sqrt(delta) / (2 * a1)

        # Speed control
        self.target_speed = min(self.maximum_speed(front_vehicle), self.target_speed)
        acceleration = self.speed_control(self.target_speed)

        return v_max, acceleration

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.traffic_manager.vehicles:

                    e = get_engine()
                    p = e.policy_manager.get_policy(v.name)
                    if p is None:
                        continue
                    v_target_lane = p.target_lane_index

                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v_target_lane == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.delay_time):
            return
        self.delay_time = 0

        # decide to make a lane change
        for lane_index in self.traffic_manager.current_map.road_network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.traffic_manager.current_map.road_network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.traffic_manager.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.traffic_manager.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self.vehicle, front_vehicle=new_preceding)
        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] -
                                                                             self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self.vehicle, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (
                new_following_pred_a - new_following_a + old_following_pred_a - old_following_a
            )
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        print('Change road!!!!')
        return True

    def recover_from_stop(self, acceleration: float) -> float:
        """
        If stopped on the wrong lane, try a reversing maneuver.

        :param acceleration: desired acceleration from IDM
        :return: suggested acceleration to recover from being stuck
        """
        stopped_speed = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.speed < stopped_speed:
            _, rear = self.traffic_manager.neighbour_vehicles(self)
            _, new_rear = self.traffic_manager.neighbour_vehicles(
                self, self.traffic_manager.current_map.road_network.get_lane(self.target_lane_index)
            )
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2
        return acceleration

    def steering_features(self, target_lane_index: LaneIndex) -> np.ndarray:
        """
        A collection of features used to follow a lane

        :param target_lane_index: index of the lane to follow
        :return: a array of features
        """
        lane = self.traffic_manager.current_map.road_network.get_lane(target_lane_index)
        lane_coords = lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.PURSUIT_TAU
        lane_future_heading = lane.heading_at(lane_next_coords)
        features = np.array(
            [
                utils.wrap_to_pi(lane_future_heading - self.heading) * self.LENGTH / utils.not_zero(self.speed),
                -lane_coords[1] * self.LENGTH / (utils.not_zero(self.speed)**2)
            ]
        )
        return features

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):

            last_lane = self.traffic_manager.current_map.road_network.get_lane(self.route[-1])
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / norm(*(self.destination - self.position))
        else:
            return np.zeros((2, ))

    def reset(self):
        # self.vehicle_node.reset(self._initial_state)
        # self.out_of_road = False
        # print('111')
        self.lane_index, _ = self.traffic_manager.current_map.road_network.get_closest_lane_index(
            self.vehicle.position
        ) if self.traffic_manager else (np.nan, np.nan)
        self.lane = self.traffic_manager.current_map.road_network.get_lane(
            self.lane_index
        ) if self.traffic_manager else None
