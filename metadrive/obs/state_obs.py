import gym
import numpy as np

from metadrive.component.vehicle_navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.obs.observation_base import ObservationBase
from metadrive.utils.math_utils import clip, norm


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
        shape = self.ego_state_obs_dim + NodeNetworkNavigation.navigation_info_dim + self.get_line_detector_dim()
        if self.config["random_agent_model"]:
            shape += 2
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
        navi_info = vehicle.navigation.get_navi_info()
        ego_state = self.vehicle_state(vehicle)
        ret = np.concatenate([ego_state, navi_info])
        return ret.astype(np.float32)

    def vehicle_state(self, vehicle):
        """
        Wrap vehicle states to list
        """
        # update out of road
        info = []
        if self.config["random_agent_model"]:

            # The length of the target vehicle
            info.append(clip(vehicle.LENGTH / vehicle.MAX_LENGTH, 0.0, 1.0))

            # The width of the target vehicle
            info.append(clip(vehicle.WIDTH / vehicle.MAX_WIDTH, 0.0, 1.0))

        if hasattr(vehicle, "side_detector") and vehicle.side_detector.available:

            # If side detector (a Lidar scanning road borders) is turn on, then add the cloud points of side detector
            info += vehicle.side_detector.perceive(vehicle, vehicle.engine.physics_world.static_world).cloud_points

        else:

            # If the side detector is turn off, then add the distance to left and right road borders as state.
            lateral_to_left, lateral_to_right, = vehicle.dist_to_left_side, vehicle.dist_to_right_side
            total_width = float((vehicle.navigation.map.MAX_LANE_NUM + 1) * vehicle.navigation.map.MAX_LANE_WIDTH)
            lateral_to_left /= total_width
            lateral_to_right /= total_width
            info += [clip(lateral_to_left, 0.0, 1.0), clip(lateral_to_right, 0.0, 1.0)]

        current_reference_lane = vehicle.navigation.current_ref_lanes[-1]
        info += [

            # The angular difference between vehicle's heading and the lane heading at this location.
            vehicle.heading_diff(current_reference_lane),

            # The velocity of target vehicle
            clip((vehicle.speed + 1) / (vehicle.max_speed + 1), 0.0, 1.0),

            # Current steering
            clip((vehicle.steering / vehicle.MAX_STEERING + 1) / 2, 0.0, 1.0),

            # The normalized actions at last steps
            clip((vehicle.last_current_action[0][0] + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[0][1] + 1) / 2, 0.0, 1.0)
        ]

        # Current angular acceleration (yaw rate)
        heading_dir_last = vehicle.last_heading_dir
        heading_dir_now = vehicle.heading
        cos_beta = heading_dir_now.dot(heading_dir_last) / (norm(*heading_dir_now) * norm(*heading_dir_last))
        beta_diff = np.arccos(clip(cos_beta, 0.0, 1.0))
        yaw_rate = beta_diff / 0.1
        info.append(clip(yaw_rate, 0.0, 1.0))

        if vehicle.lane_line_detector.available:

            # If lane line detector (a Lidar scanning current lane borders) is turn on,
            # then add the cloud points of lane line detector
            info += vehicle.lane_line_detector.perceive(vehicle, vehicle.engine.physics_world.static_world).cloud_points

        else:

            # If the lane line detector is turn off, then add the offset of current position
            # against the central of current lane to the state. If vehicle is centered in the lane, then the offset
            # is 0 and vice versa.
            _, lateral = vehicle.lane.local_coordinates(vehicle.position)
            info.append(clip((lateral * 2 / vehicle.navigation.map.MAX_LANE_WIDTH + 1.0) / 2.0, 0.0, 1.0))

        return info

    def get_line_detector_dim(self):
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
        self.cloud_points = None
        self.detected_objects = None

    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        if self.config["lidar"]["num_lasers"] > 0 and self.config["lidar"]["distance"] > 0:
            # Number of lidar rays and distance should be positive!
            lidar_dim = self.config["lidar"]["num_lasers"] + self.config["lidar"]["num_others"] * 4
            if self.config["lidar"]["add_others_navi"]:
                lidar_dim += self.config["lidar"]["num_others"] * 4
            shape[0] += lidar_dim
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
        self.current_observation = np.concatenate((state, np.asarray(other_v_info)))
        ret = self.current_observation
        return ret.astype(np.float32)

    def state_observe(self, vehicle):
        return self.state_obs.observe(vehicle)

    def lidar_observe(self, vehicle):
        other_v_info = []
        if vehicle.lidar.available:
            cloud_points, detected_objects = vehicle.lidar.perceive(vehicle, )
            if self.config["lidar"]["num_others"] > 0:
                other_v_info += vehicle.lidar.get_surrounding_vehicles_info(
                    vehicle, detected_objects, self.config["lidar"]["num_others"],
                    self.config["lidar"]["add_others_navi"]
                )
            other_v_info += self._add_noise_to_cloud_points(
                cloud_points,
                gaussian_noise=self.config["lidar"]["gaussian_noise"],
                dropout_prob=self.config["lidar"]["dropout_prob"]
            )
            self.cloud_points = cloud_points
            self.detected_objects = detected_objects
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
