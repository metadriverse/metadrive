import gym
import numpy as np
from pgdrive.obs.observation_type import ObservationType
from pgdrive.obs.state_obs import StateObservation
from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle
from pgdrive.world.image_buffer import ImageBuffer


class ImageStateObservation(ObservationType):
    """
    Use ego state info, navigation info and front cam image/top down image as input
    The shape needs special handling
    """
    IMAGE = "image"
    STATE = "state"

    def __init__(self, vehicle_config):
        config = vehicle_config
        super(ImageStateObservation, self).__init__(config)
        self.img_obs = ImageObservation(config, config["image_source"], config["rgb_clip"])
        self.state_obs = StateObservation(config)

    @property
    def observation_space(self):
        # TODO it should be specified by different vehicle
        return gym.spaces.Dict(
            {
                self.IMAGE: self.img_obs.observation_space,
                self.STATE: self.state_obs.observation_space
            }
        )

    def observe(self, vehicle: BaseVehicle):
        image_buffer = vehicle.image_sensors[self.img_obs.image_source]
        return {self.IMAGE: self.img_obs.observe(image_buffer), self.STATE: self.state_obs.observe(vehicle)}


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

    def reset(self, env, vehicle=None):
        """
        Clear stack
        :param env: PGDrive
        :param vehicle: BaseVehicle
        :return: None
        """
        self.state = np.zeros(self.observation_space.shape)
