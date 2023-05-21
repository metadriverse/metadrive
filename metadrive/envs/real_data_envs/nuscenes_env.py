import time

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

NuScenesEnv = ScenarioEnv

if __name__ == "__main__":
    env = NuScenesEnv(
        {
            "use_render": False,
            "agent_policy": ReplayEgoCarPolicy,
            # "manual_control": True,
            "show_interface": False,
            "show_logo": False,
            "sequential_seed": True,
            "curriculum_sort": True,
            # "no_traffic": True,
            "reactive_traffic": False,
            "show_fps": False,
            "debug": False,
            # "pstats": True,
            # "render_pipeline": True,
            # "daytime": "22:01",
            # "no_traffic": True,
            # "no_light": False,
            # "debug":True,
            # Make video
            "window_size": (1600, 900),
            "camera_dist": 9,
            # "camera_height": 0.5,
            # "camera_pitch": np.pi / 3,
            # "camera_fov": 60,
            # "no_traffic":True,
            # "force_render_fps": 10,
            # "start_scenario_index": 192,
            # "start_scenario_index": 1000,
            "num_scenarios": 10,
            # "force_reuse_object_name": True,
            # "data_directory": "/home/shady/Downloads/test_processed",
            "horizon": 1000,
            # "no_static_vehicles": True,
            # "show_policy_mark": True,
            # "show_coordinates": True,
            "force_destroy": True,
            # "default_vehicle_in_traffic": True,
            "vehicle_config": dict(
                # light=True,
                # random_color=True,
                show_navi_mark=False,
                # no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
            "data_directory": AssetLoader.file_path("nuscenes", return_raw_style=False),
        }
    )

    # 0,1,3,4,5,6

    success = []
    reset_num = 0
    start = time.time()
    reset_used_time = 0
    s = 0
    while True:
        # for i in range(10):
        start_reset = time.time()
        env.reset(force_seed=env.current_seed if env.engine is not None else 7)
        reset_used_time += time.time() - start_reset
        reset_num += 1
        for t in range(10000):
            o, r, d, info = env.step([0, 0])
            s += 1
            if d and info["arrive_dest"]:
                print(
                    "Time elapse: {:.4f}. Average FPS: {:.4f}, AVG_Reset_time: {:.4f}".format(
                        time.time() - start, s / (time.time() - start - reset_used_time),
                        reset_used_time / reset_num
                    )
                )
                print("seed:{}, success".format(env.engine.global_random_seed))
                break
