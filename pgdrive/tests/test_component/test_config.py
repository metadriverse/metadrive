from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import Config, recursive_equal


def test_config_unchangeable():
    c = Config({"aaa": 100}, unchangeable=True)
    try:
        c['aaa'] = 1000
    except ValueError as e:
        print('Great! ', e)
    assert c['aaa'] == 100


def test_config_sync():
    """
    The config in BaseEngine should be the same as env.config, if BaseEngine exists in process
    """
    try:
        env = PGDriveEnv({"vehicle_config": dict(
            max_engine_force=500,
            max_brake_force=40,
            max_steering=40,
        )})
        env.reset()
        recursive_equal(env.config, env.engine.global_config)
        env.config.update({"vehicle_config": dict(max_engine_force=0.1, max_brake_force=0.1, max_steering=0.1)})
        recursive_equal(env.config, env.engine.global_config)
        env.close()
        env.reset()
        recursive_equal(env.config, env.engine.global_config)
        env.engine.global_config.update(
            {"vehicle_config": dict(
                max_engine_force=50,
                max_brake_force=4,
                max_steering=4,
            )}
        )
        recursive_equal(env.config, env.engine.global_config)
    finally:
        env.close()


if __name__ == '__main__':
    test_config_unchangeable()
