from pgdrive import PGDriveEnv


def test_config_consistency():
    env = PGDriveEnv({"vehicle_config": {"lidar": {"num_lasers": 999}}})
    try:
        env.reset()
        assert env.vehicle.config["lidar"]["num_lasers"] == 999
    finally:
        env.close()


def test_config_consistency_2():
    # env = PGDriveEnv({"map_config": {"config": "OO"}})
    the_config = 11
    env = PGDriveEnv({"map_config": {"config": the_config}})
    try:
        env.reset()
        assert env.current_map.config["config"] == the_config
        assert all([v.config["config"] == the_config for v in env.maps.values()])
        assert env.config["map_config"]["config"] == the_config
    finally:
        env.close()

    the_config = 11
    env = PGDriveEnv({"map": the_config})
    try:
        env.reset()
        assert env.current_map.config["config"] == the_config
        assert all([v.config["config"] == the_config for v in env.maps.values()])
        assert env.config["map_config"]["config"] == the_config
    finally:
        env.close()

    the_config = "OO"
    env = PGDriveEnv({"map": the_config})
    try:
        env.reset()
        assert env.current_map.config["config"] == the_config
        assert all([v.config["config"] == the_config for v in env.maps.values()])
        assert env.config["map_config"]["config"] == the_config
    finally:
        env.close()

    the_config = "OO"
    env = PGDriveEnv({"map_config": {"config": the_config, "type": "block_sequence"}})
    try:
        env.reset()
        assert env.current_map.config["config"] == the_config
        assert all([v.config["config"] == the_config for v in env.maps.values()])
        assert env.config["map_config"]["config"] == the_config
    finally:
        env.close()


if __name__ == '__main__':
    # test_config_consistency()
    test_config_consistency_2()
