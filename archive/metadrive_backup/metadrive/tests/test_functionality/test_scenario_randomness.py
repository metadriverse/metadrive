from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils import setup_logger
from metadrive.utils.math import norm


def assert_equal_pos(data_1, data_2):
    assert len(data_1) == len(data_2)
    for i in range(len(data_1)):
        difference = data_1[i] - data_2[i]
        diff = norm(data_1[i][0] - data_2[i][0], data_1[i][1] - data_2[i][1])
        assert diff < 0.00001, "pos mismatch for vehicle: {}, distance: {}".format(i, diff)


def test_scenario_randomness(vis=False):
    setup_logger(True)
    cfg = {
        "accident_prob": 0.8,
        "num_scenarios": 1,
        "traffic_density": 0.1,
        "start_seed": 1000,
        "manual_control": False,
        "use_render": vis,
        "agent_policy": IDMPolicy,
        "traffic_mode": TrafficMode.Trigger,
        "record_episode": False,
        "map_config": {
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            BaseMap.GENERATE_CONFIG: "CrCSC",
            BaseMap.LANE_WIDTH: 3.5,
            BaseMap.LANE_NUM: 3,
        }
    }

    env = SafeMetaDriveEnv(cfg)
    try:
        positions_1 = []
        o, _ = env.reset()
        positions_1.append([env.agent.position] + [v.position for v in env.engine.traffic_manager.traffic_vehicles])
        for i in range(1, 100000 if vis else 2000):
            o, r, tm, tc, info = env.step([0, 1])
            positions_1.append([env.agent.position] + [v.position for v in env.engine.traffic_manager.traffic_vehicles])
            if tm or tc:
                break
        env.close()
        positions_1.reverse()
        env = SafeMetaDriveEnv(cfg)
        o, _ = env.reset()
        old_position = positions_1.pop()
        new_position = [env.agent.position] + [v.position for v in env.engine.traffic_manager.traffic_vehicles]
        assert_equal_pos(old_position, new_position)
        for i in range(1, 100000 if vis else 2000):
            o, r, tm, tc, info = env.step([0, 1])
            old_position = positions_1.pop()
            new_position = [env.agent.position] + [v.position for v in env.engine.traffic_manager.traffic_vehicles]
            assert_equal_pos(old_position, new_position)
            if tm or tc:
                break
    finally:
        env.close()


if __name__ == "__main__":
    test_scenario_randomness(vis=True)
