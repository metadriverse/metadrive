import gym
from pgdrive import PGDriveEnv


def _a(env, action):
    assert env.action_space.contains(action)
    obs, reward, done, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def _step(env):
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        for _ in range(5):
            _a(env, env.action_space.sample())
    finally:
        env.close()


def test_naive_multi_agent_pgdrive():
    env = PGDriveEnv(config={"num_agents": 1})
    assert isinstance(env.action_space, gym.spaces.Box)
    _step(env)

    env = PGDriveEnv(config={"num_agents": 10})
    assert isinstance(env.action_space, gym.spaces.Dict)
    obs = env.reset()
    assert isinstance(obs, dict)
    a = env.action_space.sample()
    assert isinstance(a, dict)
    o, r, d, i = env.step(a)
    assert isinstance(o, dict)
    assert isinstance(r, dict)
    assert isinstance(d, dict)
    assert isinstance(i, dict)
    _step(env)


if __name__ == '__main__':
    test_naive_multi_agent_pgdrive()
