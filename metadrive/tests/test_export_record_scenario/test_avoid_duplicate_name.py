from metadrive import SafeMetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.tests.test_export_record_scenario.test_save_replay_via_policy import assert_equal_pos
from metadrive.utils import setup_logger


def test_save_recreate_scenario_respawn_traffic(vis=False):
    setup_logger(True)
    cfg = {
        "accident_prob": 0.,
        "num_scenarios": 1,
        "traffic_density": 0.2,
        "start_seed": 1000,
        "manual_control": False,
        "debug": False,
        "use_render": vis,
        # "agent_policy": IDMPolicy,
        "traffic_mode": TrafficMode.Respawn,
        "record_episode": True,
        "map_config": {
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            BaseMap.GENERATE_CONFIG: "X",
            BaseMap.LANE_WIDTH: 3.5,
            BaseMap.LANE_NUM: 3,
        }
    }
    env = SafeMetaDriveEnv(cfg)
    try:
        positions_1 = []
        o, _ = env.reset()

        for i in range(1, 1000):
            o, r, tm, tc, info = env.step([0, 0])
            positions_1.append({v.name: v.position for v in env.engine.traffic_manager.spawned_objects.values()})
        epi_info = env.engine.record_manager.get_episode_metadata()
        env.close()
        env = SafeMetaDriveEnv(cfg)
        env.config["replay_episode"] = epi_info
        env.config["record_episode"] = False
        o, _ = env.reset()
        positions_1.reverse()
        for i in range(0, 1000):
            o, r, tm, tc, info = env.step([0, 1])
            position = positions_1.pop()
            position = {env.engine.replay_manager.record_name_to_current_name[key]: v for key, v in position.items()}
            current_position = {v.name: v.position for v in env.engine.replay_manager.spawned_objects.values()}
            assert_equal_pos(position, current_position)
            if info["replay_done"]:
                break
    finally:
        env.close()


if __name__ == "__main__":
    test_save_recreate_scenario_respawn_traffic(True)
