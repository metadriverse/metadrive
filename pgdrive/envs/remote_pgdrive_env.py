"""
This file provide a RemotePGDrive environment which can be easily ran in single process!
"""
import gym
from pgdrive.envs.pgdrive_env import PGDriveEnv

try:
    import ray
except ImportError:
    ray = None


def get_remote_pgdrive():
    assert ray is not None

    @ray.remote
    class _RemotePGDrive(PGDriveEnv):
        pass

    return _RemotePGDrive


class RemotePGDrive(gym.Env):
    def __init__(self, env_config):
        assert ray is not None, "Please install ray via: pip install ray " \
                                "if you wish to use multiple PGDrive in single process."
        # Temporary environment
        tmp = PGDriveEnv(env_config)
        self.action_space = tmp.action_space
        self.observation_space = tmp.observation_space
        self.reward_range = tmp.reward_range
        del tmp
        self.env = None
        self.env_config = env_config

    def step(self, *args, **kwargs):
        ret = ray.get(self.env.step.remote(*args, **kwargs))
        return ret

    def reset(self, *args, **kwargs):
        if self.env is None:
            if not ray.is_initialized():
                ray.init()
            self.env = get_remote_pgdrive().remote(self.env_config)

        return ray.get(self.env.reset.remote(*args, **kwargs))

    def close(self):
        if ray.is_initialized():
            self.env.close.remote()
            ray.shutdown()
        del self.env
        self.env = None

    def render(self, *args, **kwargs):
        raise NotImplementedError("Not implemented for remote PGDrive!")


if __name__ == '__main__':
    # Test and also show cases!
    envs = [RemotePGDrive(dict(map=7)) for _ in range(3)]
    ret = [env.reset() for env in envs]
    print(ret)
    ret = [env.step(env.action_space.sample()) for env in envs]
    print(ret)
    [env.reset() for env in envs]
    [env.close() for env in envs]
    print('Success!')
