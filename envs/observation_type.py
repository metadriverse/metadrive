from abc import ABC
from typing import Dict
import matplotlib.pyplot as plt
import gym
import numpy as np

PERCEIVE_DIST = 50


class ObservationType(ABC):
    def __init__(self, config):
        self.config = config
        self.observation_shape = self.get_obs_shape(self.config)
        self.observation_space = gym.spaces.Box(-0.0, 1.0, shape=self.observation_shape, dtype=np.float32)

    def get_obs_shape(self, config: Dict):
        raise NotImplementedError

    def observe(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def vehicle_to_left_right(vehicle):
        from scene_creator.lanes.circular_lane import CircularLane
        current_reference_lane = vehicle.routing_localization.current_ref_lanes[-1]
        lane_num = len(vehicle.routing_localization.current_ref_lanes)
        _, lateral_to_reference = current_reference_lane.local_coordinates(vehicle.position)

        if isinstance(current_reference_lane, CircularLane) \
                and lane_num == 1 and current_reference_lane.direction == -1:

            lateral_to_right = abs(lateral_to_reference) + vehicle.routing_localization.map.lane_width * (
                    vehicle.routing_localization.map.lane_num - 0.5) if lateral_to_reference < 0 \
                else vehicle.routing_localization.map.lane_width * vehicle.routing_localization.map.lane_num - \
                     vehicle.routing_localization.map.lane_width / 2 - lateral_to_reference

        else:
            lateral_to_right = abs(
                lateral_to_reference) + vehicle.routing_localization.map.lane_width / 2 if lateral_to_reference < 0 \
                else vehicle.routing_localization.map.lane_width / 2 - abs(lateral_to_reference)

        lateral_to_left = vehicle.routing_localization.map.lane_width * vehicle.routing_localization.map.lane_num - lateral_to_right
        return lateral_to_left, lateral_to_right


class ArrayObservationType(ObservationType):
    def __init__(self, config):
        super(ArrayObservationType, self).__init__(config)

    def get_obs_shape(self, config: Dict):
        # lidar + other scalar obs
        from scene_creator.ego_vehicle.base_vehicle import BaseVehicle
        from scene_creator.ego_vehicle.vehicle_module.routing_localization import RoutingLocalizationModule
        shape = BaseVehicle.Ego_state_obs_dim + RoutingLocalizationModule.Navi_obs_dim
        shape += self.config["lidar"][0] + self.config["lidar"][2] * 4
        return (shape, )

    def observe(self, vehicle):
        navi_info = vehicle.routing_localization.get_navi_info()
        ego_state = self.vehicle_state(vehicle)
        other_v_info = []
        other_v_info = vehicle.lidar.get_surrounding_vehicles_info(vehicle, self.config["lidar"][2])
        other_v_info += vehicle.lidar.get_cloud_points()
        return np.asarray(ego_state + navi_info + other_v_info, dtype=np.float32)

    @staticmethod
    def vehicle_state(vehicle):
        """
        Used to train
        """
        from utils.math_utils import clip
        current_reference_lane = vehicle.routing_localization.current_ref_lanes[-1]
        lateral_to_left, lateral_to_right = ObservationType.vehicle_to_left_right(vehicle)
        if lateral_to_left < 0 or lateral_to_right < 0:
            vehicle.out_of_road = True
        total_width = float(
            (vehicle.routing_localization.map.lane_num + 1) * vehicle.routing_localization.map.lane_width
        )
        info = [
            clip(lateral_to_left / total_width, 0.0, 1.0),
            clip(lateral_to_right / total_width, 0.0, 1.0),
            vehicle.heading_diff(current_reference_lane),
            # Note: speed can be negative denoting free fall. This happen when emergency brake.
            clip((vehicle.speed + 1) / (vehicle.max_speed + 1), 0.0, 1.0),
            clip((vehicle.steering / vehicle.max_steering + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[0][0] + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[0][1] + 1) / 2, 0.0, 1.0)
        ]
        heading_dir_last = vehicle.last_heading_dir
        heading_dir_now = vehicle.heading
        cos_beta = heading_dir_now.dot(heading_dir_last
                                       ) / (np.linalg.norm(heading_dir_now) * np.linalg.norm(heading_dir_last))

        beta_diff = np.arccos(clip(cos_beta, 0.0, 1.0))

        # print(beta)
        yaw_rate = beta_diff / 0.1
        # print(yaw_rate)
        info.append(clip(yaw_rate, 0.0, 1.0))
        _, lateral = vehicle.lane.local_coordinates(vehicle.position)
        info.append(clip((lateral * 2 / vehicle.routing_localization.map.lane_width + 1.0) / 2.0, 0.0, 1.0))
        return info


class GrayScaleObservation(ObservationType):
    STACK_SIZE = 3  # use continuous 4 image as the input

    def __init__(self, config):
        super(GrayScaleObservation, self).__init__(config)
        self.rgb_clip = True if "rgb_clip" in config and config["rgb_clip"] else False
        if not self.rgb_clip:
            self.observation_space = gym.spaces.Box(0, 255, shape=self.observation_shape, dtype=np.uint8)
        self.state = np.zeros(self.observation_shape)

    def get_obs_shape(self, config: Dict):
        shape = (84, 84) + (self.STACK_SIZE, )
        return shape

    def observe(self, vehicle):
        new_obs = vehicle.front_cam.get_gray_pixels_array(self.rgb_clip)
        # self.show_gray_scale_array(new_obs)
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[:, :, -1] = new_obs

        # update out of road
        lateral_to_left, lateral_to_right = ObservationType.vehicle_to_left_right(vehicle)
        if lateral_to_left < 0 or lateral_to_right < 0:
            vehicle.out_of_road = True
        return self.state

    @staticmethod
    def resize_img(array, dim_1, dim_2):
        """
        resize to (84, 84)
        """
        x_step = int(dim_1 / 84)
        y_step = int(dim_2 / 84 / 2)
        res = []
        for x in range(0, dim_1, x_step):
            d = []
            for y in range(dim_2 - 1, 0, -y_step):
                d.append(array[x][y])
                if len(d) > 84:
                    break
            res.append(d[:84])
        res = res[:84]
        return np.asarray(res, dtype=np.float32)

    @staticmethod
    def show_gray_scale_array(obs):
        import matplotlib.pyplot as plt
        img = np.rot90(obs)
        img = img[::-1]
        plt.plot()
        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()
