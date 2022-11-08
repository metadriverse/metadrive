from metadrive.envs.metadrive_env import MetaDriveEnv


def test_get_lane_index(use_render=False):
    env = MetaDriveEnv(
        {
            "map": "SOTCR",
            "environment_num": 1,
            "traffic_density": 0.3,
            "traffic_mode": "trigger",
            "use_render": use_render
        }
    )
    try:
        o = env.reset()
        for i in range(1, 500):
            o, r, d, info = env.step([0, 0])
            for v in env.engine.traffic_manager.vehicles:
                old_res = env.current_map.road_network.get_closest_lane_index(v.position, True)
                old_lane_idx = [index[1] for index in old_res]

                if v.lane_index not in old_lane_idx:
                    raise ValueError((v.lane_index), old_lane_idx)
                else:
                    idx = old_lane_idx.index(v.lane_index)
                    if old_res[idx][0] > 4. and idx > 2:
                        raise ValueError("L1 dist:{} of {} is too large".format(old_res[idx][0], idx))
    finally:
        env.close()


if __name__ == "__main__":
    test_get_lane_index(False)
