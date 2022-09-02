import gym
import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.obs.observation_base import ObservationBase
from metadrive.obs.state_obs import StateObservation


class ImageStateObservation(ObservationBase):
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
        return gym.spaces.Dict(
            {
                self.IMAGE: self.img_obs.observation_space,
                self.STATE: self.state_obs.observation_space
            }
        )

    def observe(self, vehicle: BaseVehicle):
        return {self.IMAGE: self.img_obs.observe(vehicle), self.STATE: self.state_obs.observe(vehicle)}


class ImageObservation(ObservationBase):
    """
    Use only image info as input
    """
    STACK_SIZE = 3  # use continuous 3 image as the input

    def __init__(self, config, image_source: str, clip_rgb: bool):
        self.STACK_SIZE = config["stack_size"]
        self.image_source = image_source
        super(ImageObservation, self).__init__(config)
        self.rgb_clip = clip_rgb
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)

    @property
    def observation_space(self):
        shape = (self.config[self.image_source][1], self.config[self.image_source][0]
                 ) + ((self.STACK_SIZE, ) if self.config["rgb_to_grayscale"] else (3, self.STACK_SIZE))
        if self.rgb_clip:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        else:
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def observe(self, vehicle):
        new_obs = vehicle.image_sensors[self.image_source].get_pixels_array(vehicle, self.rgb_clip)
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[..., -1] = new_obs
        return self.state

    def get_image(self):
        return self.state.copy()[:, :, -1]

    def reset(self, env, vehicle=None):
        """
        Clear stack
        :param env: MetaDrive
        :param vehicle: BaseVehicle
        :return: None
        """
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
