from metadrive.engine.asset_loader import AssetLoader
import numpy as np
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.examples.generate_video_for_bev_and_interface import VideoRecorder

NuScenesEnv = ScenarioEnv

if __name__ == "__main__":
    env = NuScenesEnv(
        {
            "use_render": True,
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "show_interface": False,
            "show_logo": False,
            # "no_traffic": True,
            "reactive_traffic": False,
            "show_fps": False,
            "debug": False,
            # "pstats": True,
            "render_pipeline": True,
            "pstats": True,
            "daytime": "22:01",
            # "no_traffic": True,
            # "no_light": False,
            # "debug":True,
            # Make video
            "window_size": (1600, 900),
            "camera_dist": -2.5,
            "camera_height": 0.5,
            "camera_pitch": np.pi / 3,
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
            "vehicle_config": dict(
                light=True,
                show_navi_mark=False,
                no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
            "data_directory": AssetLoader.file_path("nuscenes", return_raw_style=False),
        }
    )

    # 0,1,3,4,5,6

    success = []
    while True:
        env.reset(force_seed=7)
        # env.engine.force_fps.disable()
        for t in range(10000):
            o, r, d, info = env.step([0, 0])
            # env.capture("nuscenes_{:03d}.png".format(t))
            if d and info["arrive_dest"]:
                print("seed:{}, success".format(env.engine.global_random_seed))
                break
