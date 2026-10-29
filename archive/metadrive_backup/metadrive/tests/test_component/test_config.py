from metadrive.envs.metadrive_env import MetaDriveEnv, METADRIVE_DEFAULT_CONFIG
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

        env.config["vehicle_config"]["max_engine_force"] = 100
        assert recursive_equal(env.config, env.engine.global_config)
        assert recursive_equal(env.config, get_global_config())
        assert recursive_equal(env.config, BaseEngine.global_config)
        assert env.engine.global_config["vehicle_config"]["max_engine_force"] == 100
        env.reset()
        assert env.engine.global_config["vehicle_config"]["max_engine_force"] == 100
        assert recursive_equal(env.config, env.engine.global_config)
        assert recursive_equal(env.config, get_global_config())
        assert recursive_equal(env.config, BaseEngine.global_config)

    finally:
        env.close()


def test_config_set_unchange():
    """
    The config in BaseEngine should be the same as env.config, if BaseEngine exists in process
    """
    try:
        env = MetaDriveEnv({"vehicle_config": dict(show_lidar=False, show_navi_mark=False)})
        evn_cfg = env.config
        assert evn_cfg is BaseEngine.global_config
        initialize_global_config(None)
        env = None
        env = MetaDriveEnv({"vehicle_config": dict(show_lidar=False, show_navi_mark=False)})
        assert env.config is not evn_cfg
        env_cfg = env.config
        initialize_global_config(None)
        assert BaseEngine.global_config is None
        env.reset()
        assert env.config is env_cfg
        assert env_cfg is BaseEngine.global_config
        env.close()
        assert env.config is env_cfg
        assert BaseEngine.global_config is None
        env.reset()
        assert env.config is env_cfg
        assert BaseEngine.global_config is env_cfg
        env.close()

        env = MetaDriveEnv({"vehicle_config": dict(show_lidar=False, show_navi_mark=False)})
        assert env.config is not env_cfg
        assert BaseEngine.global_config is env.config
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


def test_config_two_env():
    """
    The config in BaseEngine should be the same as env.config, if BaseEngine exists in process
    """
    cfg_1 = dict(show_lidar=False, show_navi_mark=False)
    env_1 = MetaDriveEnv({"vehicle_config": cfg_1})
    assert env_1.config is BaseEngine.global_config
    cfg_2 = dict(show_lidar=False, show_navi_mark=True)
    env_2 = MetaDriveEnv({"vehicle_config": cfg_2})
    try:
        assert get_global_config() is BaseEngine.global_config is env_2.config
        env_1.reset()
        assert get_global_config() is BaseEngine.global_config is env_1.config
        env_1.close()
        assert get_global_config() is None and BaseEngine.global_config is None
        env_2.reset()
        assert env_1.config is not env_2.config and (env_2.config is BaseEngine.global_config is get_global_config())
        env_2.close()
        assert get_global_config() is None and BaseEngine.global_config is None
    finally:
        env_1.close()
        env_2.close()


def test_config_overwrite():
    """
    The config in BaseEngine should be the same as env.config, if BaseEngine exists in process
    """
    METADRIVE_DEFAULT_CONFIG["map"] = "S"
    env = MetaDriveEnv({"vehicle_config": dict(show_lidar=False, show_navi_mark=False)})
    try:
        env.reset()
        assert env.current_map.blocks[-1].ID == "S" and len(env.current_map.blocks) == 2
        env.close()
        env = MetaDriveEnv({"map": "C", "vehicle_config": dict(show_lidar=False, show_navi_mark=False)})
        env.reset()
        assert env.current_map.blocks[-1].ID == "C" and len(env.current_map.blocks) == 2
        env.close()
    finally:
        env.close()


if __name__ == '__main__':
    test_config_sync()
