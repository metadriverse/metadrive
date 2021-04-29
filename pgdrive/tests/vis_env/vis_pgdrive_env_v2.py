from pgdrive import PGDriveEnvV2
from pgdrive.utils import setup_logger

if __name__ == "__main__":
    setup_logger(True)
    env = PGDriveEnvV2(
        {
            "start_seed": 0,
            "environment_num": 5,
            "fast": True,
            "use_render": True,
            "manual_control": True,
            "vehicle_config": {
                "show_side_detector": True,
                "show_lane_line_detector": True,
                "show_navi_mark": True,
                "show_lidar": True,
            }
        }
    )

    o = env.reset()
    print("vehicle num", len(env.scene_manager.traffic_manager.vehicles))
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        info["fuel"] = env.vehicle.energy_consumption
        env.render(
            text={
                "left": env.vehicle.dist_to_left,
                "right": env.vehicle.dist_to_right,
                "white_lane_line": env.vehicle.on_white_continuous_line,
                "reward": r,
            }
        )
        if d:
            print("Reset")
            # env.reset()
    env.close()
