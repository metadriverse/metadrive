import tqdm

from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy


def test_first_frame_depth_cam():
    env = ScenarioEnv(
        {
            # To enable onscreen rendering, set this config to True.
            # "use_render": False,

            # !!!!! To enable offscreen rendering, set this config to True !!!!!
            "image_observation": True,
            # "render_pipeline": False,

            # ===== The scenario and MetaDrive config =====
            "agent_policy": ReplayEgoCarPolicy,
            "no_traffic": False,
            "sequential_seed": True,
            "reactive_traffic": False,
            "start_scenario_index": 0,
            "num_scenarios": 10,
            # "horizon": 1000,
            # "no_static_vehicles": False,
            "vehicle_config": dict(
                # show_navi_mark=False,
                # use_special_color=False,
                image_source="depth_camera",
                # lidar=dict(num_lasers=120, distance=50),
                # lane_line_detector=dict(num_lasers=0, distance=50),
                # side_detector=dict(num_lasers=12, distance=50)
            ),
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),

            # ===== Set some sensor and visualization configs =====
            # "daytime": "08:10",
            # "window_size": (800, 450),
            # "camera_dist": 0.8,
            # "camera_height": 1.5,
            # "camera_pitch": None,
            # "camera_fov": 60,

            # "interface_panel": ["semantic_camera"],
            # "show_interface": True,
            "sensors": dict(
                # semantic_camera=(SemanticCamera, 1600, 900),
                depth_camera=(DepthCamera, 80, 60),
                # rgb_camera=(RGBCamera, 800, 600),
            ),
        }
    )

    try:
        for ep in tqdm.trange(5):
            env.reset()
            for t in range(10000):

                img = env.engine.get_sensor("depth_camera").perceive(False)
                # img = env.engine.get_sensor("depth_camera").get_image(env.agent)

                assert not (img == 255).all()
                if t == 5:
                    break
                env.step([1, 0.88])

    finally:
        env.close()


if __name__ == '__main__':
    test_first_frame_depth_cam()
