from abc import ABC


class ObservationBase(ABC):
    def __init__(self, config, env=None):
        self.config = config
        self.env = env

    @property
    def observation_space(self):
        raise NotImplementedError

    def observe(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, env, vehicle=None):
        pass
