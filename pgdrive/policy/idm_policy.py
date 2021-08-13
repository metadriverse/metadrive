from pgdrive.policy.base_policy import BasePolicy
from pgdrive.utils.scene_utils import is_same_lane_index, is_following_lane_index
import numpy as np
from pgdrive.utils.math_utils import not_zero, wrap_to_pi
from pgdrive.component.vehicle_module.PID_controller import PIDController


class IDMPolicy(BasePolicy):
    """
    We implement this policy based on the HighwayEnv code base.
    """
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.3  # [s]
    TAU_LATERAL = 0.8  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 5.0
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front v"""

    DELTA = 2.0  # []
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self, control_object, random_seed):
        super(IDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.target_speed = 30
        self.target_lane = None

        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)

    def act(self):
        if self.target_lane is None:
            self.target_lane = self.control_object.lane
        elif self.control_object.lane is not self.target_lane:
            index = self.target_lane.index[-1]
            self.target_lane = self.control_object.navigation.current_ref_lanes[index]
        steering = self.steering_control(ego_vehicle=self.control_object)
        front_obj, dist = self.find_front_obj()
        acc = self.acceleration(self.control_object, front_obj, dist)
        return [steering, acc]

    def steering_control(self, ego_vehicle) -> float:
        # heading control following a lateral distance control
        target_lane = self.target_lane
        long, lat = target_lane.local_coordinates(ego_vehicle.position)
        lane_heading = target_lane.heading_at(long + 1)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)

    def acceleration(self, ego_vehicle, front_obj, dist_to_front) -> float:
        ego_target_speed = not_zero(self.target_speed, 0)
        acceleration = self.COMFORT_ACC_MAX * (1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))
        if front_obj:
            d = dist_to_front
            acceleration -= self.COMFORT_ACC_MAX * \
                            np.power(self.desired_gap(ego_vehicle, front_obj) / not_zero(d), 2)
        return acceleration

    def desired_gap(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_obj.velocity, ego_vehicle.heading) if projected \
            else ego_vehicle.speed - front_obj.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def find_front_obj(self):
        objs = self.control_object.lidar.get_surrounding_objects(self.control_object)
        min_long = 1000
        ret = None
        find_in_current_lane = False
        current_long = self.control_object.lane.local_coordinates(self.control_object.position)[0]
        left_long = self.control_object.lane.length - current_long

        for obj in objs:
            if is_same_lane_index(obj.lane_index, self.control_object.lane_index):
                long = self.control_object.lane.local_coordinates(obj.position)[0] - current_long
                if min_long > long > 0:
                    min_long = long
                    ret = obj
                    find_in_current_lane = True
            elif not find_in_current_lane and is_following_lane_index(self.control_object.lane_index, obj.lane_index):
                long = obj.lane.local_coordinates(obj.position)[0] + left_long
                if min_long > long > 0:
                    min_long = long
                    ret = obj
        return ret, min_long

    def reset(self):
        self.heading_pid.reset()
        self.lateral_pid.reset()
