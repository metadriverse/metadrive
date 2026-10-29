from abc import ABC
import numpy as np
import gymnasium as gym
from copy import deepcopy
from metadrive.engine.logger import get_logger
from metadrive.utils.config import Config

logger = get_logger()


class BaseObservation(ABC):
    """
    BaseObservation Class. Observation should implement all abstracted methods
    """
    def __init__(self, config):
        # assert not engine_initialized(), "Observations can not be created after initializing the simulation"
        self.config = deepcopy(config)
        self.current_observation = None

    @property
    def engine(self):
        from metadrive.engine.engine_utils import get_engine
        return get_engine()

    @property
    def observation_space(self):
        raise NotImplementedError

    def observe(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, env, vehicle=None):
        pass

    def destroy(self):
        """
        Clear allocated memory
        """
        pass
        # Config.clear_nested_dict(self.config)
        # self.config = None


class DummyObservation(BaseObservation):
    """
    Fake Observation class, can be used as placeholder
    """
    def __init__(self, config=None):
        super(DummyObservation, self).__init__(config)
        logger.warning("You are using DummyObservation which doesn't collect information from the environment.")

    @property
    def observation_space(self):
        return gym.spaces.Box(-0.0, 1.0, shape=(1, ), dtype=np.float32)

    def observe(self, *args, **kwargs):
        return np.array([0])
