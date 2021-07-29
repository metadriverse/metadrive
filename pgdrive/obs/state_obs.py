import gym
import numpy as np

from pgdrive.obs.observation_base import ObservationBase
from pgdrive.component.vehicle_module.routing_localization import RoutingLocalizationModule
from pgdrive.utils.math_utils import clip, norm


class StateObservation(ObservationBase):
    ego_state_obs_dim = 6
    """
    Use vehicle state info, navigation info and lidar point clouds info as input
    """
    def __init__(self, config):
        super(StateObservation, self).__init__(config)

    @property
    def observation_space(self):
        # Navi info + Other states
        shape = self.ego_state_obs_dim + RoutingLocalizationModule.navigation_info_dim + self.get_side_detector_dim()
        return gym.spaces.Box(-0.0, 1.0, shape=(shape, ), dtype=np.float32)

    def observe(self, vehicle):
        """
        Ego states: [
                    [Distance to left yellow Continuous line,
                    Distance to right Side Walk], if NOT use lane_line detector else:
                    [Side_detector cloud_points]

                    Difference of heading between ego vehicle and current lane,
                    Current speed,
                    Current steering,
                    Throttle/brake of last frame,
                    Steering of last frame,
                    Yaw Rate,

                     [Lateral Position on current lane.], if use lane_line detector, else:
                     [lane_line_detector cloud points]
                    ], dim >= 9
        Navi info: [
                    Projection of distance between ego vehicle and checkpoint on ego vehicle's heading direction,
                    Projection of distance between ego vehicle and checkpoint on ego vehicle's side direction,
                    Radius of the lane whose end node is the checkpoint (0 if this lane is straight),
                    Clockwise (1) or anticlockwise (0) (0 if lane is straight),
                    Angle of the lane (0 if lane is straight)
                   ] * 2, dim = 10
        Since agent observes current lane info and next lane info, and two checkpoints exist, the dimension of navi info
        is 10.
        :param vehicle: BaseVehicle
        :return: Vehicle State + Navigation information
        """
        navi_info = vehicle.routing_localization.get_navi_info()
        ego_state = self.vehicle_state(vehicle)
        return np.concatenate([ego_state, navi_info])

    @staticmethod
    def vehicle_state(vehicle):
        """
        Wrap vehicle states to list
        """
        # update out of road
        info = []
        if hasattr(vehicle, "side_detector") and vehicle.side_detector is not None:
            info += vehicle.side_detector.get_cloud_points()
        else:
            lateral_to_left, lateral_to_right, = vehicle.dist_to_left_side, vehicle.dist_to_right_side
            total_width = float(
                (vehicle.routing_localization.get_current_lane_num() + 1) *
                vehicle.routing_localization.get_current_lane_width()
            )
            lateral_to_left /= total_width
            lateral_to_right /= total_width
            info += [clip(lateral_to_left, 0.0, 1.0), clip(lateral_to_right, 0.0, 1.0)]

        # print("Heading Diff: ", [
        #     vehicle.heading_diff(current_reference_lane)
        #     for current_reference_lane in vehicle.routing_localization.current_ref_lanes
        # ])

        current_reference_lane = vehicle.routing_localization.current_ref_lanes[-1]
        info += [
            vehicle.heading_diff(current_reference_lane),
            # Note: speed can be negative denoting free fall. This happen when emergency brake.
            clip((vehicle.speed + 1) / (vehicle.max_speed + 1), 0.0, 1.0),
            clip((vehicle.steering / vehicle.max_steering + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[0][0] + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[0][1] + 1) / 2, 0.0, 1.0)
        ]
        heading_dir_last = vehicle.last_heading_dir
        heading_dir_now = vehicle.heading
        cos_beta = heading_dir_now.dot(heading_dir_last) / (norm(*heading_dir_now) * norm(*heading_dir_last))
        beta_diff = np.arccos(clip(cos_beta, 0.0, 1.0))
        # print(beta)
        yaw_rate = beta_diff / 0.1
        # print(yaw_rate)
        info.append(clip(yaw_rate, 0.0, 1.0))

        if vehicle.lane_line_detector is not None:
            info += vehicle.lane_line_detector.get_cloud_points()
        else:
            _, lateral = vehicle.lane.local_coordinates(vehicle.position)
            info.append(
                clip((lateral * 2 / vehicle.routing_localization.get_current_lane_width() + 1.0) / 2.0, 0.0, 1.0)
            )

        return info

    def get_side_detector_dim(self):
        dim = 0
        dim += 2 if self.config["side_detector"]["num_lasers"] == 0 else \
            self.config["side_detector"]["num_lasers"]
        dim += 1 if self.config["lane_line_detector"]["num_lasers"] == 0 else \
            self.config["lane_line_detector"]["num_lasers"]
        return dim


class LidarStateObservation(ObservationBase):
    def __init__(self, vehicle_config):
        self.state_obs = StateObservation(vehicle_config)
        super(LidarStateObservation, self).__init__(vehicle_config)

    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        if self.config["lidar"]["num_lasers"] > 0 and self.config["lidar"]["distance"] > 0:
            # Number of lidar rays and distance should be positive!
            shape[0] += self.config["lidar"]["num_lasers"] + self.config["lidar"]["num_others"] * 4
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def observe(self, vehicle):
        """
        State observation + Navi info + 4 * closest vehicle info + Lidar points ,
        Definition of State Observation and Navi information can be found in **class StateObservation**
        Other vehicles' info: [
                              Projection of distance between ego and another vehicle on ego vehicle's heading direction,
                              Projection of distance between ego and another vehicle on ego vehicle's side direction,
                              Projection of speed between ego and another vehicle on ego vehicle's heading direction,
                              Projection of speed between ego and another vehicle on ego vehicle's side direction,
                              ] * 4, dim = 16

        Lidar points: 240 lidar points surrounding vehicle, starting from the vehicle head in clockwise direction

        :param vehicle: BaseVehicle
        :return: observation in 9 + 10 + 16 + 240 dim
        """
        state = self.state_observe(vehicle)
        other_v_info = self.lidar_observe(vehicle)
        return np.concatenate((state, np.asarray(other_v_info)))

    def state_observe(self, vehicle):
        return self.state_obs.observe(vehicle)

    def lidar_observe(self, vehicle):
        other_v_info = []
        if vehicle.lidar is not None:
            if self.config["lidar"]["num_others"] > 0:
                other_v_info += vehicle.lidar.get_surrounding_vehicles_info(vehicle, self.config["lidar"]["num_others"])
            other_v_info += self._add_noise_to_cloud_points(
                vehicle.lidar.get_cloud_points(),
                gaussian_noise=self.config["lidar"]["gaussian_noise"],
                dropout_prob=self.config["lidar"]["dropout_prob"]
            )
        return other_v_info

    def _add_noise_to_cloud_points(self, points, gaussian_noise, dropout_prob):
        if gaussian_noise > 0.0:
            points = np.asarray(points)
            points = np.clip(points + np.random.normal(loc=0.0, scale=gaussian_noise, size=points.shape), 0.0, 1.0)

        if dropout_prob > 0.0:
            assert dropout_prob <= 1.0
            points = np.asarray(points)
            points[np.random.uniform(0, 1, size=points.shape) < dropout_prob] = 0.0

        return list(points)
