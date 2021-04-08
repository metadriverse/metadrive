import numpy as np

from pgdrive.envs.pgdrive_env_v2_minimal import PGDriveEnvV2Minimal


def test_pgdrive_env_v2_minimal():
    def _act(env, action):
        assert env.action_space.contains(action)
        obs, reward, done, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isscalar(reward)
        assert isinstance(info, dict)

    def _test(env):
        try:
            obs = env.reset()
            assert env.observation_space.contains(obs)
            _act(env, env.action_space.sample())
            env.reset()
            for c in range(100):
                print(c)
                _act(env, [0, 1])
        finally:
            env.close()

    _test(PGDriveEnvV2Minimal({"num_others": 4, "use_extra_state": True, "traffic_density": 0.5}))
    _test(PGDriveEnvV2Minimal({"num_others": 0, "use_extra_state": True, "traffic_density": 0.5}))
    _test(PGDriveEnvV2Minimal({"num_others": 4, "use_extra_state": False, "traffic_density": 0.5}))
    _test(PGDriveEnvV2Minimal({"num_others": 0, "use_extra_state": False, "traffic_density": 0.5}))


def test_pgdrive_env_v2_minimal_long_run():
    try:
        for m in ["X", "O", "C", "S", "R", "r", "T"]:
            env = PGDriveEnvV2Minimal(
                {
                    "map": m,
                    "fast": True,
                    "use_render": False,
                    "debug": True,
                    "camera_height": 100,
                    "vehicle_config": {
                        "show_lidar": True
                    }
                }
            )
            o = env.reset()
            for c in range(300):
                print(c)
                if c > 100:
                    print(c)
                assert env.observation_space.contains(o)
                o, r, d, i = env.step([0, 1])
                if d:
                    break
            env.close()
    finally:
        if "env" in locals():
            env = locals()["env"]
            env.close()


if __name__ == '__main__':
    # test_pgdrive_env_v2_minimal()
    test_pgdrive_env_v2_minimal_long_run()
