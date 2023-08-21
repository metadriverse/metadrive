from abc import ABC
from copy import deepcopy


class ObservationBase(ABC):
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
