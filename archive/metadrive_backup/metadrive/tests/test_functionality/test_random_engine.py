import numpy as np

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import recursive_equal, norm


def test_seeding():
    env = MetaDriveEnv({"num_scenarios": 1000})
    try:
        env.reset()
        env.seed(999)
        # assert env.engine is None
        assert env.current_seed == 999
        env.reset(seed=992)
        assert env.current_seed == 992
        # assert env.engine is not None
    finally:
        env.close()


def test_map_random_seeding():
    cfg_1 = {
        "num_scenarios": 1,
        "start_seed": 5,
        "random_lane_width": True,
        "random_lane_num": True,
    }
    cfg_2 = {
        "num_scenarios": 10,
        "start_seed": 5,
        "random_lane_width": True,
        "random_lane_num": True,
    }
    cfg_3 = {
        "num_scenarios": 100,
        "start_seed": 5,
        "random_lane_width": True,
        "random_lane_num": True,
    }
    cfg_4 = {
        "num_scenarios": 10,
        "start_seed": 0,
        "random_lane_width": True,
        "random_lane_num": True,
    }
    cfg_5 = {
        "num_scenarios": 3,
        "start_seed": 3,
        "random_lane_width": True,
        "random_lane_num": True,
    }
    from metadrive.component.pgblock.first_block import FirstPGBlock
    map_configs = []
    lane_width = []
    lane_num = []
    for cfg in [cfg_1, cfg_2, cfg_3, cfg_4, cfg_5]:
        env = MetaDriveEnv(cfg)
        try:
            env.reset()
            env.reset()
            env.reset(seed=5)
            map_configs.append(env.current_map.get_meta_data())
            lane_num.append(len(env.current_map.road_network.graph[FirstPGBlock.NODE_1][FirstPGBlock.NODE_2]))
            lane_width.append(
                env.current_map.road_network.graph[FirstPGBlock.NODE_1][FirstPGBlock.NODE_2][0].width_at(0)
            )
        finally:
            env.close()
    for idx, map_cfg in enumerate(map_configs[:-1]):
        nxt_map_cfg = map_configs[idx + 1]
        ret = recursive_equal(map_cfg, nxt_map_cfg)
        assert ret, "Error"
    assert np.std(lane_width) < 0.01 and np.std(lane_num) < 0.01, "random engine error"


def test_fixed_traffic():
    env = MetaDriveEnv({
        "random_traffic": False,
        "traffic_mode": "respawn",
        #  "use_render": True
    })
    try:
        last_pos = None
        for i in range(20):
            obs, _ = env.reset()
            assert env.engine.traffic_manager.random_seed == env.current_seed
            new_pos = [v.position for v in env.engine.traffic_manager.vehicles]
            if last_pos is not None and len(new_pos) == len(last_pos):
                assert sum(
                    [norm(lastp[0] - newp[0], lastp[1] - newp[1]) <= 1e-3 for lastp, newp in zip(last_pos, new_pos)]
                ), [(lastp, newp) for lastp, newp in zip(last_pos, new_pos)]
            last_pos = new_pos
    finally:
        env.close()


def test_random_traffic():
    env = MetaDriveEnv(
        {
            "random_traffic": True,
            "traffic_mode": "respawn",
            "traffic_density": 0.3,
            "start_seed": 5,

            #  "use_render": True
        }
    )
    has_traffic = False
    try:
        last_pos = None
        for i in range(10):
            obs, _ = env.reset(seed=5)
            assert env.engine.traffic_manager.random_traffic
            new_pos = [v.position for v in env.engine.traffic_manager.traffic_vehicles]
            if len(new_pos) > 0:
                has_traffic = True
            if last_pos is not None and len(new_pos) == len(last_pos):
                assert sum(
                    [norm(lastp[0] - newp[0], lastp[1] - newp[1]) >= 0.5 for lastp, newp in zip(last_pos, new_pos)]
                ), [(lastp, newp) for lastp, newp in zip(last_pos, new_pos)]
            last_pos = new_pos
        assert has_traffic
    finally:
        env.close()


def test_random_lane_width():
    env = MetaDriveEnv(
        {
            "num_scenarios": 5,
            "traffic_density": .2,
            "traffic_mode": "trigger",
            "start_seed": 12,
            "random_lane_width": True,
        }
    )
    try:
        o, _ = env.reset(seed=12)
        old_config_1 = env.agent.lane.width
        env.reset(seed=15)
        old_config_2 = env.agent.lane.width
        env.reset(seed=13)
        env.reset(seed=12)
        new_config = env.agent.lane.width
        assert old_config_1 == new_config
        env.reset(seed=15)
        new_config = env.agent.lane.width
        assert old_config_2 == new_config
        assert old_config_1 != old_config_2
    finally:
        env.close()


def test_random_lane_num():
    env = MetaDriveEnv(
        {
            "num_scenarios": 5,
            "traffic_density": .2,
            "traffic_mode": "trigger",
            "start_seed": 12,
            "random_lane_num": True,
        }
    )
    try:
        o, _ = env.reset(seed=12)
        old_config_1 = env.agent.navigation.get_current_lane_num()
        env.reset(seed=15)
        old_config_2 = env.agent.navigation.get_current_lane_num()
        env.reset(seed=13)
        env.reset(seed=12)
        new_config = env.agent.navigation.get_current_lane_num()
        assert old_config_1 == new_config
        env.reset(seed=15)
        new_config = env.agent.navigation.get_current_lane_num()
        assert old_config_2 == new_config
        env.close()
        env.reset(seed=12)
        assert old_config_1 == env.agent.navigation.get_current_lane_num()
        env.reset(seed=15)
        assert old_config_2 == env.agent.navigation.get_current_lane_num()
    finally:
        env.close()


def test_random_vehicle_parameter():
    env = MetaDriveEnv(
        {
            "num_scenarios": 5,
            "traffic_density": .2,
            "traffic_mode": "trigger",
            "start_seed": 12,
            "random_agent_model": True
        }
    )
    try:
        o, _ = env.reset(seed=12)
        old_config_1 = env.agent.get_config(True)
        env.reset(seed=15)
        old_config_2 = env.agent.get_config(True)
        env.reset(seed=13)
        env.reset(seed=12)
        new_config = env.agent.get_config(True)
        assert recursive_equal(old_config_1, new_config)
        env.reset(seed=15)
        new_config = env.agent.get_config(True)
        assert recursive_equal(old_config_2, new_config)
    finally:
        env.close()


if __name__ == '__main__':
    test_map_random_seeding()
    # test_seeding()
    # test_random_lane_width()
