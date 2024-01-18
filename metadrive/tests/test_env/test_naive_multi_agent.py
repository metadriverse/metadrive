import gymnasium as gym
import numpy as np

from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive


def _a(env, action):
    assert env.action_space.contains(action)
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def _step(env):
    try:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        for _ in range(5):
            _a(env, env.action_space.sample())
    finally:
        env.close()


def test_naive_multi_agent_metadrive():
    # env = MetaDriveEnv(config={"num_agents": 1})
    # assert isinstance(env.action_space, gym.spaces.Box)
    # _step(env)
    # env.close()

    env = MultiAgentMetaDrive(
        config={
            "map": "SSS",
            "num_agents": 4,
            "agent_configs": {"agent{}".format(i): {
                "spawn_longitude": i * 5
            }
                              for i in range(4)}
        }
    )
    try:
        assert isinstance(env.action_space, gym.spaces.Dict)
        obs, _ = env.reset()
        assert isinstance(obs, dict)
        env.action_space.seed(0)
        for step in range(100):
            a = env.action_space.sample()
            assert isinstance(a, dict)
            o, r, tm, tc, i = env.step(a)
            if len(o) > 2:
                obses = list(o.values())
                assert not np.isclose(obses[0], obses[1], rtol=1e-3, atol=1e-3).all()

            pos_z_list = [v.chassis.getNode(0).transform.pos[2] for v in env.agents.values()]
            for p in pos_z_list:
                assert p < 5.0 or step <= 10

            assert isinstance(o, dict)
            assert isinstance(r, dict)
            assert isinstance(tm, dict)
            assert isinstance(tc, dict)
            assert isinstance(i, dict)
            if tm["__all__"]:
                break

        _step(env)
    finally:
        env.close()


if __name__ == '__main__':
    test_naive_multi_agent_metadrive()
