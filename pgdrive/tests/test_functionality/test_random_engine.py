from pgdrive import PGDriveEnvV2
from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import recursive_equal, norm


def test_seeding():
    env = PGDriveEnv({"environment_num": 1000})
    try:
        env.seed(999)
        assert env.pgdrive_engine is None
        assert env.current_seed == 999
        env.reset(force_seed=999)
        assert env.current_seed == 999
        assert env.pgdrive_engine is not None
    finally:
        env.close()


def test_map_random_seeding():
    cfg_1 = {
        "environment_num": 1,
        "start_seed": 5,
    }
    cfg_2 = {
        "environment_num": 10,
        "start_seed": 5,
    }
    cfg_3 = {
        "environment_num": 100,
        "start_seed": 5,
    }
    cfg_4 = {
        "environment_num": 10,
        "start_seed": 0,
    }
    cfg_5 = {
        "environment_num": 3,
        "start_seed": 3,
    }
    map_configs = []
    for cfg in [cfg_1, cfg_2, cfg_3, cfg_4, cfg_5]:
        env = PGDriveEnv(cfg)
        try:
            env.reset(force_seed=5)
            map_configs.append(env.current_map.save_map)
        finally:
            env.close()
    for idx, map_cfg in enumerate(map_configs[:-1]):
        nxt_map_cfg = map_configs[idx + 1]
        recursive_equal(map_cfg, nxt_map_cfg)


def test_fixed_traffic():
    env = PGDriveEnvV2({
        "random_traffic": False,
        "traffic_mode": "respawn",
        # "fast": True, "use_render": True
    })
    try:
        last_pos = None
        for i in range(20):
            obs = env.reset()
            assert env.pgdrive_engine.traffic_manager.random_seed == env.current_seed
            new_pos = [v.position for v in env.pgdrive_engine.traffic_manager.vehicles]
            if last_pos is not None and len(new_pos) == len(last_pos):
                assert sum(
                    [norm(lastp[0] - newp[0], lastp[1] - newp[1]) <= 1e-3 for lastp, newp in zip(last_pos, new_pos)]
                ), [(lastp, newp) for lastp, newp in zip(last_pos, new_pos)]
            last_pos = new_pos
    finally:
        env.close()


def test_random_traffic():
    env = PGDriveEnvV2(
        {
            "random_traffic": True,
            "traffic_mode": "respawn",
            "traffic_density": 0.3,
            "start_seed": 5,

            # "fast": True, "use_render": True
        }
    )
    has_traffic = False
    try:
        last_pos = None
        for i in range(20):
            obs = env.reset(force_seed=5)
            assert env.pgdrive_engine.traffic_manager.random_traffic
            new_pos = [v.position for v in env.pgdrive_engine.traffic_manager.traffic_vehicles]
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


if __name__ == '__main__':
    test_map_random_seeding()
