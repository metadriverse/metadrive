from pgdrive.envs.pgdrive_env import PGDriveEnv


def test_get_lane_index(use_render=False):
    env = PGDriveEnv(
        {
            "map": "rRCXSOTCR",
            "environment_num": 1,
            "traffic_density": 0.3,
            "traffic_mode": "reborn",
            "use_render": use_render
        }
    )

    o = env.reset()
    for i in range(1, 1000):
        o, r, d, info = env.step([0, 0])
        for v in env.scene_manager.traffic.vehicles:
            assert v.lane_index == env.current_map.road_network.get_closest_lane_index(v.position)[0]
    env.close()


if __name__ == "__main__":
    test_get_lane_index(True)
