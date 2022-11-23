import logging

import numpy as np
from metadrive.component.vehicle_module.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.utils.math_utils import not_zero, wrap_to_pi


class FrontBackObjects:
    def __init__(self, front_ret, back_ret, front_dist, back_dist):
        self.front_objs = front_ret
        self.back_objs = back_ret
        self.front_dist = front_dist
        self.back_dist = back_dist

    def left_lane_exist(self):
        return True if self.front_dist[0] is not None else False

    def right_lane_exist(self):
        return True if self.front_dist[-1] is not None else False

    def has_front_object(self):
        return True if self.front_objs[1] is not None else False

    def has_back_object(self):
        return True if self.back_objs[1] is not None else False

    def has_left_front_object(self):
        return True if self.front_objs[0] is not None else False

    def has_left_back_object(self):
        return True if self.back_objs[0] is not None else False

    def has_right_front_object(self):
        return True if self.front_objs[-1] is not None else False

    def has_right_back_object(self):
        return True if self.back_objs[-1] is not None else False

    def front_object(self):
        return self.front_objs[1]

    def left_front_object(self):
        return self.front_objs[0]

    def right_front_object(self):
        return self.front_objs[-1]

    def back_object(self):
        return self.back_objs[1]

    def left_back_object(self):
        return self.back_objs[0]

    def right_back_object(self):
        return self.back_objs[-1]

    def left_front_min_distance(self):
        assert self.left_lane_exist(), "left lane doesn't exist"
        return self.front_dist[0]

    def right_front_min_distance(self):
        assert self.right_lane_exist(), "right lane doesn't exist"
        return self.front_dist[-1]

    def front_min_distance(self):
        return self.front_dist[1]

    def left_back_min_distance(self):
        assert self.left_lane_exist(), "left lane doesn't exist"
        return self.back_dist[0]

    def right_back_min_distance(self):
        assert self.right_lane_exist(), "right lane doesn't exist"
        return self.back_dist[-1]

    def back_min_distance(self):
        return self.back_dist[1]

    @classmethod
    def get_find_front_back_objs(cls, objs, lane, position, max_distance, ref_lanes=None):
        """
        Find objects in front of/behind the lane and its left lanes/right lanes, return objs, dist.
        If ref_lanes is None, return filter results of this lane
        """
        if ref_lanes is not None:
            assert lane in ref_lanes
        idx = lane.index[-1] if ref_lanes is not None else None
        left_lane = ref_lanes[idx - 1] if ref_lanes is not None and idx > 0 else None
        right_lane = ref_lanes[idx + 1] if ref_lanes is not None and idx + 1 < len(ref_lanes) else None
        lanes = [left_lane, lane, right_lane]

        min_front_long = [max_distance if lane is not None else None for lane in lanes]
        min_back_long = [max_distance if lane is not None else None for lane in lanes]

        front_ret = [None, None, None]
        back_ret = [None, None, None]

        find_front_in_current_lane = [False, False, False]
        find_back_in_current_lane = [False, False, False]

        current_long = [lane.local_coordinates(position)[0] if lane is not None else None for lane in lanes]
        left_long = [lane.length - current_long[idx] if lane is not None else None for idx, lane in enumerate(lanes)]

        for i, lane in enumerate(lanes):
            if lane is None:
                continue
            for obj in objs:
                if obj.lane is lane:
                    long = lane.local_coordinates(obj.position)[0] - current_long[i]
                    if min_front_long[i] > long > 0:
                        min_front_long[i] = long
                        front_ret[i] = obj
                        find_front_in_current_lane[i] = True
                    if long < 0 and abs(long) < min_back_long[i]:
                        min_back_long[i] = abs(long)
                        back_ret[i] = obj
                        find_back_in_current_lane[i] = True

                elif not find_front_in_current_lane[i] and lane.is_previous_lane_of(obj.lane):
                    long = obj.lane.local_coordinates(obj.position)[0] + left_long[i]
                    if min_front_long[i] > long > 0:
                        min_front_long[i] = long
                        front_ret[i] = obj
                elif not find_back_in_current_lane[i] and obj.lane.is_previous_lane_of(lane):
                    long = obj.lane.length - obj.lane.local_coordinates(obj.position)[0] + current_long[i]
                    if min_back_long[i] > long:
                        min_back_long[i] = long
                        back_ret[i] = obj

        return cls(front_ret, back_ret, min_front_long, min_back_long)

    @classmethod
    def get_find_front_back_objs_waymo(cls, objs, lane, position, max_distance):
        """
        Find objects in front of/behind the lane and its left lanes/right lanes, return objs, dist.
        If ref_lanes is None, return filter results of this lane
        """
        lanes = [None, lane, None]

        min_front_long = [max_distance if lane is not None else None for lane in lanes]
        min_back_long = [max_distance if lane is not None else None for lane in lanes]

        front_ret = [None, None, None]
        back_ret = [None, None, None]

        find_front_in_current_lane = [False, False, False]
        # find_back_in_current_lane = [False, False, False]

        current_long = [lane.local_coordinates(position)[0] if lane is not None else None for lane in lanes]

        for i, lane in enumerate(lanes):
            if lane is None:
                continue
            for obj in objs:
                if np.linalg.norm(obj.position - position) > max_distance:
                    continue
                long, lat = lane.local_coordinates(obj.position)
                if abs(lat) > lane.width:
                    continue
                long = long - current_long[i]
                if min_front_long[i] > long > 0:
                    min_front_long[i] = long
                    front_ret[i] = obj
                    find_front_in_current_lane[i] = True

        return cls(front_ret, back_ret, min_front_long, min_back_long)


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

    DISTANCE_WANTED = 10.0
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front v"""

    DELTA = 10.0  # []
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    # Lateral policy parameters
    LANE_CHANGE_FREQ = 50  # [step]
    LANE_CHANGE_SPEED_INCREASE = 10
    SAFE_LANE_CHANGE_DISTANCE = 15
    MAX_LONG_DIST = 30
    MAX_SPEED = 100

    # Normal speed
    NORMAL_SPEED = 30

    # Creep Speed
    CREEP_SPEED = 5

    # acc factor
    ACC_FACTOR = 1.0
    DEACC_FACTOR = -5

    def __init__(self, control_object, random_seed):
        super(IDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.target_speed = self.NORMAL_SPEED
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        self.enable_lane_change = self.engine.global_config.get("enable_idm_lane_change", True)
        self.disable_idm_deceleration = self.engine.global_config.get("disable_idm_deceleration", False)
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)

    def act(self, *args, **kwargs):
        # concat lane
        sucess = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        try:
            if sucess and self.enable_lane_change:
                # perform lane change due to routing
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                # can not find routing target lane
                surrounding_objects = FrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object.position,
                    max_distance=self.MAX_LONG_DIST
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
        except:
            # error fallback
            acc_front_obj = None
            acc_front_dist = 5
            steering_target_lane = self.routing_target_lane
            # logging.warning("IDM bug! fall back")
            # print("IDM bug! fall back")

        # control by PID and IDM
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)
        return [steering, acc]

    def move_to_next_road(self):
        # routing target lane is in current ref lanes
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.routing_target_lane is None:
            self.routing_target_lane = self.control_object.lane
            return True if self.routing_target_lane in current_lanes else False
        routing_network = self.control_object.navigation.map.road_network
        if self.routing_target_lane not in current_lanes:
            for lane in current_lanes:
                if self.routing_target_lane.is_previous_lane_of(lane) or \
                        routing_network.has_connection(self.routing_target_lane.index, lane.index):
                    # two lanes connect
                    self.routing_target_lane = lane
                    return True
                    # lane change for lane num change
            return False
        elif self.control_object.lane in current_lanes and self.routing_target_lane is not self.control_object.lane:
            # lateral routing lane change
            self.routing_target_lane = self.control_object.lane
            self.overtake_timer = self.np_random.randint(0, int(self.LANE_CHANGE_FREQ / 2))
            return True
        else:
            return True

    def steering_control(self, target_lane) -> float:
        # heading control following a lateral distance control
        ego_vehicle = self.control_object
        long, lat = target_lane.local_coordinates(ego_vehicle.position)
        lane_heading = target_lane.heading_theta_at(long + 1)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)

    def acceleration(self, front_obj, dist_to_front) -> float:
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(self.target_speed, 0)
        acceleration = self.ACC_FACTOR * (1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))
        if front_obj and (not self.disable_idm_deceleration):
            d = dist_to_front
            speed_diff = self.desired_gap(ego_vehicle, front_obj) / not_zero(d)
            acceleration -= self.ACC_FACTOR * (speed_diff**2)
        return acceleration

    def desired_gap(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.ACC_FACTOR * self.DEACC_FACTOR
        dv = np.dot(ego_vehicle.velocity - front_obj.velocity, ego_vehicle.heading) if projected \
            else ego_vehicle.speed - front_obj.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def reset(self):
        self.heading_pid.reset()
        self.lateral_pid.reset()
        self.target_speed = self.NORMAL_SPEED
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)

    def lane_change_policy(self, all_objects):
        current_lanes = self.control_object.navigation.current_ref_lanes
        surrounding_objects = FrontBackObjects.get_find_front_back_objs(
            all_objects, self.routing_target_lane, self.control_object.position, self.MAX_LONG_DIST, current_lanes
        )
        self.available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = self.control_object.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0

        # We have to perform lane changing because the number of lanes in next road is less than current road
        if lane_num_diff > 0:
            # lane num decreasing happened in left road or right road
            if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                index_range = [i for i in range(len(next_lanes))]
            else:
                index_range = [i for i in range(lane_num_diff, len(current_lanes))]
            self.available_routing_index_range = index_range
            if self.routing_target_lane.index[-1] not in index_range:
                # not on suitable lane do lane change !!!
                if self.routing_target_lane.index[-1] > index_range[-1]:
                    # change to left
                    if surrounding_objects.left_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.left_front_min_distance() < 5:
                        # creep to wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane
                    else:
                        # it is time to change lane!
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] - 1]
                else:
                    # change to right
                    if surrounding_objects.right_back_min_distance(
                    ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.right_front_min_distance() < 5:
                        # unsafe, creep and wait
                        self.target_speed = self.CREEP_SPEED
                        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                        ), self.routing_target_lane,
                    else:
                        # change lane
                        self.target_speed = self.NORMAL_SPEED
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                               current_lanes[self.routing_target_lane.index[-1] + 1]

        # lane follow or active change lane/overtake for high driving speed
        if abs(self.control_object.speed - self.NORMAL_SPEED) > 3 and surrounding_objects.has_front_object(
        ) and abs(surrounding_objects.front_object().speed -
                  self.NORMAL_SPEED) > 3 and self.overtake_timer > self.LANE_CHANGE_FREQ:
            # may lane change
            right_front_speed = surrounding_objects.right_front_object().speed if surrounding_objects.has_right_front_object() else self.MAX_SPEED \
                if surrounding_objects.right_lane_exist() and surrounding_objects.right_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.right_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE else None
            front_speed = surrounding_objects.front_object().speed if surrounding_objects.has_front_object(
            ) else self.MAX_SPEED
            left_front_speed = surrounding_objects.left_front_object().speed if surrounding_objects.has_left_front_object() else self.MAX_SPEED \
                if surrounding_objects.left_lane_exist() and surrounding_objects.left_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.left_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE else None
            if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                # left overtake has a high priority
                expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                if expect_lane_idx in self.available_routing_index_range:
                    return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                           current_lanes[expect_lane_idx]
            if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                if expect_lane_idx in self.available_routing_index_range:
                    return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                           current_lanes[expect_lane_idx]

        # fall back to lane follow
        self.target_speed = self.NORMAL_SPEED
        self.overtake_timer += 1
        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane


class ManualControllableIDMPolicy(IDMPolicy):
    def __init__(self, *args, **kwargs):
        super(ManualControllableIDMPolicy, self).__init__(*args, **kwargs)
        self.manual_control_policy = ManualControlPolicy(*args, **kwargs)

    def act(self, agent_id):
        if self.control_object is self.engine.current_track_vehicle and self.engine.global_config["manual_control"] \
                and not self.engine.current_track_vehicle.expert_takeover:
            return self.manual_control_policy.act(agent_id)
        else:
            return super(ManualControllableIDMPolicy, self).act(agent_id)


class WaymoIDMPolicy(IDMPolicy):
    NORMAL_SPEED = 40
    WAYMO_IDM_MAX_DIST = 20

    # DEACC_FACTOR = -20
    # ACC_FACTOR = 0.5
    # TIME_WANTED = 80
    # DISTANCE_WANTED = 55.0

    def __init__(self, control_object, random_seed, traj_to_follow, policy_index):
        super(WaymoIDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.policy_index = policy_index
        self.traj_to_follow = traj_to_follow
        self.target_speed = self.NORMAL_SPEED
        self.routing_target_lane = self.traj_to_follow
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        self.enable_lane_change = False

        self.heading_pid = PIDController(1.2, 0.1, 3.5)
        self.lateral_pid = PIDController(0.3, .0, 0.0)

        self.last_action = [0, 0]

    def steering_control(self, target_lane) -> float:
        # heading control following a lateral distance control
        ego_vehicle = self.control_object
        long, lat = target_lane.local_coordinates(ego_vehicle.position)
        lane_heading = target_lane.heading_theta_at(long + 1)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(wrap_to_pi(lane_heading - v_heading))
        # steering += self.lateral_pid.get_result(-lat)
        return float(steering)

    def act(self, do_speed_control, *args, **kwargs):
        # concat lane
        try:
            if do_speed_control:
                all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
                # can not find routing target lane
                surrounding_objects = FrontBackObjects.get_find_front_back_objs_waymo(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object.position,
                    max_distance=self.WAYMO_IDM_MAX_DIST
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()

                acc = self.acceleration(acc_front_obj, acc_front_dist)
            else:
                acc = self.last_action[-1]
        except:
            acc = 0
            print("WaymoIDM Longitudinal Planning failed, acceleration fall back to 0")

        # if self.policy_index % 2 == 0:
        steering_target_lane = self.routing_target_lane
        # control by PID and IDM
        steering = self.steering_control(steering_target_lane)
        # else:
        #     steering = self.last_action[0]
        self.last_action = [steering, acc]
        return [steering, acc]


class EgoWaymoIDMPolicy(IDMPolicy):
    """
    This policy is for Vehicles using navigation module, especially, ego car
    """
    NORMAL_SPEED = 30

    def __init__(self, control_object, random_seed):
        super(EgoWaymoIDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.target_speed = self.NORMAL_SPEED
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        self.enable_lane_change = False

        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .0, 0.0)

    def steering_control(self, target_lane) -> float:
        # heading control following a lateral distance control
        ego_vehicle = self.control_object
        long, lat = target_lane.local_coordinates(ego_vehicle.position)
        lane_heading = target_lane.heading_theta_at(long + 1)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(wrap_to_pi(lane_heading - v_heading))
        # steering += self.lateral_pid.get_result(-lat)
        return float(steering)

    def move_to_next_road(self):
        # routing target lane is in current ref lanes
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.routing_target_lane is None:
            self.routing_target_lane = self.control_object.lane
            return True if self.routing_target_lane in current_lanes else False
        if self.routing_target_lane not in current_lanes:
            for lane in current_lanes:
                self.routing_target_lane = lane
                return True
                # lane change for lane num change
            self.routing_target_lane = self.control_object.navigation.map.road_network.get_lane(
                self.control_object.navigation.next_checkpoint_lane_index
            )
            return True
        return False
