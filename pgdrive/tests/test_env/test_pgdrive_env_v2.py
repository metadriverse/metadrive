from pgdrive.envs.pgdrive_env_v2 import PGDriveEnvV2
from pgdrive.tests.test_env.test_pgdrive_env import _act


def test_pgdrive_env_v2():
    env = PGDriveEnvV2({"vehicle_config": {"wheel_friction": 1.2}})
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
                # print('finish {}! \n'.format((x, y)))
    finally:
        env.close()


def test_pgdrive_env_v2_long_run():
    try:
        for m in ["X", "O", "C", "S", "R", "r", "T"]:
            env = PGDriveEnvV2({"map": m})
            o = env.reset()
            for _ in range(300):
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
    test_pgdrive_env_v2()
    # test_pgdrive_env_v2_long_run()
