from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.engine.base_engine import BaseEngine
from metadrive.engine.engine_utils import get_global_config, initialize_global_config
from metadrive.utils import Config, recursive_equal


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
        env = MetaDriveEnv({"vehicle_config": dict(show_lidar=False, show_navi_mark=False)})
        # assert recursive_equal(env.config, env.engine.global_config)
        assert recursive_equal(env.config, get_global_config())
        assert recursive_equal(env.config, BaseEngine.global_config)
        assert env.config is BaseEngine.global_config
        env.reset()
        assert env.config is env.engine.global_config is BaseEngine.global_config
        assert recursive_equal(env.config, env.engine.global_config)
        assert recursive_equal(env.config, get_global_config())
        assert recursive_equal(env.config, BaseEngine.global_config)
        env.config.update({"vehicle_config": dict(show_lidar=True, show_navi_mark=True)})
        assert recursive_equal(env.config, env.engine.global_config)
        assert recursive_equal(env.config, get_global_config())
        assert recursive_equal(env.config, BaseEngine.global_config)
        env.close()
        # assert recursive_equal(env.config, get_global_config())
        # assert recursive_equal(env.config, BaseEngine.global_config)
        env.reset()
        assert recursive_equal(env.config, env.engine.global_config)
        assert recursive_equal(env.config, get_global_config())
        assert recursive_equal(env.config, BaseEngine.global_config)
        env.engine.global_config.update({"vehicle_config": dict(show_lidar=False, show_navi_mark=False)})
        assert recursive_equal(env.config, env.engine.global_config)
        assert recursive_equal(env.config, get_global_config())
        assert recursive_equal(env.config, BaseEngine.global_config)

        camera_shape = (128, 876)
        env.config["vehicle_config"]["rgb_camera"] = camera_shape
        assert recursive_equal(env.config, env.engine.global_config)
        assert recursive_equal(env.config, get_global_config())
        assert recursive_equal(env.config, BaseEngine.global_config)
        assert env.engine.global_config["vehicle_config"]["rgb_camera"] == camera_shape
        env.reset()
        assert env.engine.global_config["vehicle_config"]["rgb_camera"] == camera_shape
        assert recursive_equal(env.config, env.engine.global_config)
        assert recursive_equal(env.config, get_global_config())
        assert recursive_equal(env.config, BaseEngine.global_config)

    finally:
        env.close()


def test_config_set():
    """
    The config in BaseEngine should be the same as env.config, if BaseEngine exists in process
    """
    try:
        env = MetaDriveEnv({"vehicle_config": dict(show_lidar=False, show_navi_mark=False)})
        initialize_global_config(None)
        env = None
        env = MetaDriveEnv({"vehicle_config": dict(show_lidar=False, show_navi_mark=False)})
        initialize_global_config(None)
        env.reset()
        env.close()

        env = MetaDriveEnv({"vehicle_config": dict(show_lidar=False, show_navi_mark=False)})
        # initialize_global_config(None)
        env.reset()

        old_cfg = BaseEngine.global_config
        env.close()
        assert env.config is old_cfg
        env.reset()
        old_cfg_2 = BaseEngine.global_config
        assert old_cfg is old_cfg_2 is env.config
        env.close()
    finally:
        env.close()


def _test_config_set_conce():
    """
    The config in BaseEngine should be the same as env.config, if BaseEngine exists in process
    """
    try:
        env = MetaDriveEnv({"vehicle_config": dict(show_lidar=False, show_navi_mark=False)})
        test_pass = False
        try:
            initialize_global_config({})
            env.reset()
        except AssertionError:
            test_pass = True
        assert test_pass, "Test Fail"
    finally:
        env.close()


if __name__ == '__main__':
    test_config_sync()
