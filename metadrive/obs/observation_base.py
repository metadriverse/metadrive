from abc import ABC


class ObservationBase(ABC):
    def __init__(self, config):
        self.config = config
        self.current_observation = None

    @property
    def observation_space(self):
        raise NotImplementedError

    def observe(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, env, vehicle=None):
        pass
