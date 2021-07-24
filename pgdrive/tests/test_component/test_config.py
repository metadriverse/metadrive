from pgdrive.utils import PGConfig, recursive_equal

from pgdrive.envs.pgdrive_env import PGDriveEnv


def test_config_unchangeable():
    c = PGConfig({"aaa": 100}, unchangeable=True)
    try:
        c['aaa'] = 1000
    except ValueError as e:
        print('Great! ', e)
    assert c['aaa'] == 100


def test_config_sync():
    """
    The config in PGDriveEngine should be the same as env.config, if PGDriveEngine exists in process
    """
    env = PGDriveEnv({"vehicle_config": dict(
        max_engine_force=500,
        max_brake_force=40,
        max_steering=40,
    )})
    env.reset()
    recursive_equal(env.config, env.pgdrive_engine.global_config)
    env.config.update({"vehicle_config": dict(max_engine_force=0.1, max_brake_force=0.1, max_steering=0.1)})
    recursive_equal(env.config, env.pgdrive_engine.global_config)
    env.close()
    env.reset()
    recursive_equal(env.config, env.pgdrive_engine.global_config)
    env.pgdrive_engine.global_config.update(
        {"vehicle_config": dict(
            max_engine_force=50,
            max_brake_force=4,
            max_steering=4,
        )}
    )
    recursive_equal(env.config, env.pgdrive_engine.global_config)
    env.close()


if __name__ == '__main__':
    test_config_unchangeable()
