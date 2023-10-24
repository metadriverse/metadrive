import gymnasium as gym
from metadrive.component.sensors.base_camera import BaseCamera
import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.obs.observation_base import ObservationBase
from metadrive.obs.state_obs import StateObservation

_cuda_enable = True
try:
    import cupy as cp
except ImportError:
    _cuda_enable = False


class ImageStateObservation(ObservationBase):
    """
    Use ego state info, navigation info and front cam image/top down image as input
    The shape needs special handling
    """
    IMAGE = "image"
    STATE = "state"

    def __init__(self, config):
        super(ImageStateObservation, self).__init__(config)
        self.img_obs = ImageObservation(config, config["vehicle_config"]["image_source"], config["rgb_clip"])
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
        self.enable_cuda = config["image_on_cuda"]
        if self.enable_cuda:
            assert _cuda_enable, "CuPy is not enabled"
        self.STACK_SIZE = config["stack_size"]
        self.image_source = image_source
        super(ImageObservation, self).__init__(config)
        self.rgb_clip = clip_rgb
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32 if self.rgb_clip else np.uint8)
        if self.enable_cuda:
            self.state = cp.asarray(self.state)

    @property
    def observation_space(self):
        sensor_cls = self.config["sensors"][self.image_source][0]
        assert sensor_cls == "MainCamera" or issubclass(sensor_cls, BaseCamera), "Sensor should be BaseCamera"
        channel = sensor_cls.num_channels if sensor_cls != "MainCamera" else 3
        shape = (self.config["sensors"][self.image_source][2], self.config["sensors"][self.image_source][1]
                 ) + ((self.STACK_SIZE, ) if self.config["rgb_to_grayscale"] else (channel, self.STACK_SIZE))
        if self.rgb_clip:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        else:
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def observe(self, vehicle):
        new_obs = self.engine.get_sensor(self.image_source).perceive(vehicle, self.rgb_clip)
        self.state = cp.roll(self.state, -1, axis=-1) if self.enable_cuda else np.roll(self.state, -1, axis=-1)
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
        if self.enable_cuda:
            self.state = cp.asarray(self.state)
