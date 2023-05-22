import gymnasium
from gymnasium.spaces import Box

from metadrive.constants import TerminationState
from metadrive.envs.base_env import BaseEnv


class GymnasiumEnvWrapper:
    def __init__(self, *args, **kwargs):
        super(GymnasiumEnvWrapper, self).__init__(*args, **kwargs)
        self._skip_env_checking = True

    def step(self, actions):
        o, r, d, i = super(GymnasiumEnvWrapper, self).step(actions)
        truncated = True if i[TerminationState.MAX_STEP] else False
        return o, r, d, truncated, i

    @property
    def observation_space(self):
        obs_space = super(GymnasiumEnvWrapper, self).observation_space
        return Box(low=obs_space.low, high=obs_space.high, shape=obs_space.shape)

    @property
    def action_space(self):
        space = super(GymnasiumEnvWrapper, self).action_space
        return Box(low=space.low, high=space.high, shape=space.shape)

    def reset(self, *, seed=None, options=None):
        return super(GymnasiumEnvWrapper, self).reset(force_seed=seed), {}

    @classmethod
    def build(cls, base_class):
        assert issubclass(base_class, BaseEnv), "The base class should be the subclass of BaseEnv!"
        return type("{}({})".format(cls.__name__, base_class.__name__), (cls, base_class), {})


if __name__ == '__main__':
    from metadrive.envs.metadrive_env import MetaDriveEnv

    env = GymnasiumEnvWrapper.build(MetaDriveEnv)()
    o, i = env.reset()
    assert isinstance(env.observation_space, gymnasium.Space)
    assert isinstance(env.action_space, gymnasium.Space)
    while True:
        o, r, d, t, i = env.step([0, 0])
        if d:
            env.reset()
