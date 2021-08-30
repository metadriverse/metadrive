import gym

from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive


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


def test_naive_multi_agent_metadrive():
    # env = MetaDriveEnv(config={"num_agents": 1})
    # assert isinstance(env.action_space, gym.spaces.Box)
    # _step(env)
    # env.close()

    env = MultiAgentMetaDrive(
        config={
            "map": "SSS",
            "num_agents": 4,
            "target_vehicle_configs": {"agent{}".format(i): {
                "spawn_longitude": i * 5
            }
                                       for i in range(4)}
        }
    )
    try:
        assert isinstance(env.action_space, gym.spaces.Dict)
        obs = env.reset()
        assert isinstance(obs, dict)
        env.action_space.seed(0)
        for step in range(100):
            a = env.action_space.sample()
            assert isinstance(a, dict)
            o, r, d, i = env.step(a)

            pos_z_list = [v.chassis.getNode(0).transform.pos[2] for v in env.vehicles.values()]
            for p in pos_z_list:
                assert p < 5.0 or step <= 10

            assert isinstance(o, dict)
            assert isinstance(r, dict)
            assert isinstance(d, dict)
            assert isinstance(i, dict)
            if d["__all__"]:
                break

        _step(env)
    finally:
        env.close()


if __name__ == '__main__':
    test_naive_multi_agent_metadrive()
