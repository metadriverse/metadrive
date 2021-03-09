from abc import ABC

import gym
import numpy as np
from pgdrive.scene_creator.ego_vehicle.base_vehicle import BaseVehicle
from pgdrive.scene_creator.ego_vehicle.vehicle_module.routing_localization import RoutingLocalizationModule
from pgdrive.utils import import_pygame
from pgdrive.utils.math_utils import clip
from pgdrive.world.image_buffer import ImageBuffer

PERCEIVE_DIST = 50


class ObservationType(ABC):
    def __init__(self, config, env=None):
        self.config = config
        self.env = env

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
        current_reference_lane = vehicle.routing_localization.current_ref_lanes[-1]
        lateral_to_left, lateral_to_right = vehicle.dist_to_left, vehicle.dist_to_right
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
        import matplotlib.pyplot as plt  # Lazy import
        img = np.moveaxis(obs, -1, 0)
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
        # Navi info + Other states
        shape = BaseVehicle.Ego_state_obs_dim + RoutingLocalizationModule.Navi_obs_dim
        return gym.spaces.Box(-0.0, 1.0, shape=(shape, ), dtype=np.float32)

    def observe(self, vehicle):
        """
        Ego states: [
                    Distance to left yellow Continuous line,
                    Distance to right Side Walk,
                    Difference of heading between ego vehicle and current lane,
                    Current speed,
                    Current steering,
                    Throttle/brake of last frame,
                    Steering of last frame,
                    Yaw Rate,
                    Lateral Position on current lane.
                    ], dim = 9
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
        return np.asarray(ego_state + navi_info, dtype=np.float32)


class ImageObservation(ObservationType):
    """
    Use only image info as input
    """
    STACK_SIZE = 3  # use continuous 3 image as the input

    def __init__(self, config, image_source: str, clip_rgb: bool):
        self.image_source = image_source
        super(ImageObservation, self).__init__(config)
        self.rgb_clip = clip_rgb
        self.state = np.zeros(self.observation_space.shape)

    @property
    def observation_space(self):
        shape = tuple(self.config[self.image_source][0:2]) + (self.STACK_SIZE, )
        if self.rgb_clip:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        else:
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def observe(self, image_buffer: ImageBuffer):
        new_obs = image_buffer.get_pixels_array(self.rgb_clip)
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[:, :, -1] = new_obs
        return self.state

    def get_image(self):
        return self.state.copy()[:, :, -1]


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

    def __init__(self, config, image_source: str, clip_rgb: bool):
        super(ImageStateObservation, self).__init__(config)
        self.img_obs = ImageObservation(config, image_source, clip_rgb)
        self.state_obs = StateObservation(config)

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                self.IMAGE: self.img_obs.observation_space,
                self.STATE: self.state_obs.observation_space
            }
        )

    def observe(self, vehicle: BaseVehicle):
        image_buffer = vehicle.image_sensors[self.img_obs.image_source]
        return {self.IMAGE: self.img_obs.observe(image_buffer), self.STATE: self.state_obs.observe(vehicle)}


class TopDownObservation(ObservationType):
    def __init__(self, config, env, clip_rgb: bool):
        super(TopDownObservation, self).__init__(config, env)
        self.rgb_clip = clip_rgb
        self.num_stacks = 3
        self.obs_shape = (64, 64)

        self.pygame = import_pygame()

    @property
    def observation_space(self):
        shape = self.obs_shape + (self.num_stacks, )
        if self.rgb_clip:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        else:
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def observe(self, vehicle: BaseVehicle):
        self.env.pg_world.highway_render.render()
        surface = self.env.pg_world.highway_render.get_observation_window()
        img = self.pygame.surfarray.array3d(surface)
        if self.rgb_clip:
            img = img.astype(np.float32) / 255
        else:
            img = img.astype(np.uint8)
        return np.transpose(img, (1, 0, 2))
