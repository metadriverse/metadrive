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
    try:
        o = env.reset()
        for i in range(1, 1000):
            o, r, d, info = env.step([0, 0])
            for v in env.scene_manager.traffic_mgr.vehicles:
                old = env.current_map.road_network.get_closest_lane_index(v.position)
                if v.lane_index[1:] != old[0][1:]:
                    raise ValueError((v.lane_index, old[0]))
    finally:
        env.close()


if __name__ == "__main__":
    test_get_lane_index(False)
