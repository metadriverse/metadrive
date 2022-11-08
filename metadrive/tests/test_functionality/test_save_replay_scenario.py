import pickle

from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import setup_logger
from metadrive.utils.math_utils import norm


def assert_equal_pos(dict_1, dict_2):
    for name, pos in dict_1.items():
        assert name in dict_2, "No such vehicle in dict 2: {}".format(name)
        pos_2 = dict_2[name]
        difference = pos - pos_2
        diff = norm(difference[0], difference[1])
        assert diff < 1, "pos mismatch for vehicle: {}, distance: {}".format(name, diff)


def test_save_recreate_scenario(vis=False):
    setup_logger(True)
    cfg = {
        "accident_prob": 0.8,
        "environment_num": 1,
        "traffic_density": 0.1,
        "start_seed": 1000,
        "manual_control": True,
        "debug": False,
        "use_render": vis,
        "agent_policy": IDMPolicy,
        "traffic_mode": TrafficMode.Trigger,
        "record_episode": True,
        "map_config": {
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            BaseMap.GENERATE_CONFIG: "CrCSX",
            BaseMap.LANE_WIDTH: 3.5,
            BaseMap.LANE_NUM: 3,
        }
    }
    env = SafeMetaDriveEnv(cfg)
    try:
        positions_1 = []
        o = env.reset()
        epi_info = env.engine.record_manager.get_episode_metadata()
        for i in range(1, 100000 if vis else 2000):
            o, r, d, info = env.step([0, 1])
            positions_1.append({v.name: v.position for v in env.engine.traffic_manager.spawned_objects.values()})
            if d:
                break
        env.close()
        env = SafeMetaDriveEnv(cfg)
        env.config["replay_episode"] = epi_info
        env.config["record_episode"] = False
        env.config["only_reset_when_replay"] = True
        o = env.reset()
        positions_1.reverse()
        for i in range(0, 100000 if vis else 2000):
            o, r, d, info = env.step([0, 1])
            position = positions_1.pop()
            position = {env.engine.replay_manager.record_name_to_current_name[key]: v for key, v in position.items()}
            current_position = {v.name: v.position for v in env.engine.traffic_manager.spawned_objects.values()}
            assert_equal_pos(position, current_position)
            if d:
                break
    finally:
        env.close()


if __name__ == "__main__":
    test_save_recreate_scenario(vis=True)
