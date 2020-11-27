from abc import ABC
from typing import Dict
from pg_drive.world.image_buffer import ImageBuffer
import gym
import numpy as np

PERCEIVE_DIST = 50


class ObservationType(ABC):
    def __init__(self, config):
        self.config = config

    @property
    def observation_space(self):
        raise NotImplementedError

    def observe(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def vehicle_state(vehicle):
        """
        Wrap vehicle states to list
        """
        # update out of road
        from pg_drive.utils.math_utils import clip
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

    @staticmethod
    def vehicle_to_left_right(vehicle):
        from pg_drive.scene_creator.lanes.circular_lane import CircularLane
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


class StateObservation(ObservationType):
    """
    Use vehicle state info, navigation info and lidar point clouds info as input
    """
    def __init__(self, config):
        super(StateObservation, self).__init__(config)

    @property
    def observation_space(self):
        # lidar + other scalar obs
        from pg_drive.scene_creator.ego_vehicle.base_vehicle import BaseVehicle
        from pg_drive.scene_creator.ego_vehicle.vehicle_module.routing_localization import RoutingLocalizationModule
        shape = BaseVehicle.Ego_state_obs_dim + RoutingLocalizationModule.Navi_obs_dim
        return gym.spaces.Box(-0.0, 1.0, shape=(shape, ), dtype=np.float32)

    def observe(self, vehicle):
        navi_info = vehicle.routing_localization.get_navi_info()
        ego_state = self.vehicle_state(vehicle)
        return np.asarray(ego_state + navi_info, dtype=np.float32)


class ImageObservation(ObservationType):
    """
    Use only image info as input
    """
    STACK_SIZE = 3  # use continuous 4 image as the input

    def __init__(self, config, image_buffer_name: str):
        self.image_buffer_name = image_buffer_name
        super(ImageObservation, self).__init__(config)
        self.rgb_clip = True if "rgb_clip" in config and config["rgb_clip"] else False
        self.state = np.zeros(self.observation_space.shape)

    @property
    def observation_space(self):
        shape = tuple(self.config[self.image_buffer_name][0:2]) + (self.STACK_SIZE, )
        if self.rgb_clip:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        else:
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def observe(self, image_buffer: ImageBuffer):
        new_obs = image_buffer.get_gray_pixels_array(self.rgb_clip)
        # self.show_gray_scale_array(new_obs)
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[:, :, -1] = new_obs
        return self.state


class LidarStateObservation(ObservationType):
    def __init__(self, config):
        self.state_obs = StateObservation(config)
        super(LidarStateObservation, self).__init__(config)

    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        shape[0] += self.config["lidar"][0] + self.config["lidar"][2] * 4
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def observe(self, vehicle):
        state = self.state_obs.observe(vehicle)
        other_v_info = []
        other_v_info += vehicle.lidar.get_surrounding_vehicles_info(vehicle, self.config["lidar"][2])
        other_v_info += vehicle.lidar.get_cloud_points()
        return np.concatenate((state, np.asarray(other_v_info)))


class ImageStateObservation(ObservationType):
    """
    Use ego state info, navigation info and front cam image/top down image as input
    The shape needs special handling
    """
    IMAGE = "image"
    STATE = "state"

    def __init__(self, config, image_buffer_name: str):
        super(ImageStateObservation, self).__init__(config)
        self.img_obs = ImageObservation(config, image_buffer_name)
        self.state_obs = StateObservation(config)

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                self.IMAGE: self.img_obs.observation_space,
                self.STATE: self.state_obs.observation_space
            }
        )

    def observe(self, vehicle):
        if self.img_obs.image_buffer_name == "front_cam":
            image_buffer = vehicle.front_cam
        elif self.img_obs.image_buffer_name == "mini_map":
            image_buffer = vehicle.mini_map
        else:
            raise ValueError("No such as module named {} in vehicle".format(self.img_obs.image_buffer_name))
        return {self.IMAGE: self.img_obs.observe(image_buffer), self.STATE: self.state_obs.observe(vehicle)}
