import pathlib

import pygame
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

if __name__ == "__main__":
    env = ScenarioEnv(
        {
            # To enable onscreen rendering, set this config to True.
            "use_render": False,

            # !!!!! To enable offscreen rendering, set this config to True !!!!!
            "image_observation": True,

            # ===== The scenario and MetaDrive config =====
            "agent_policy": ReplayEgoCarPolicy,
            "no_traffic": False,
            "sequential_seed": True,
            "reactive_traffic": False,
            "start_scenario_index": 0,
            "num_scenarios": 10,
            "horizon": 1000,
            "no_static_vehicles": False,
            "vehicle_config": dict(
                show_navi_mark=False,
                use_special_color=False,
                image_source="semantic_camera",
                lidar=dict(num_lasers=120, distance=50),
                lane_line_detector=dict(num_lasers=0, distance=50),
                side_detector=dict(num_lasers=12, distance=50)
            ),
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),

            # ===== Set some sensor and visualization configs =====
            "daytime": "08:10",
            "window_size": (800, 450),
            "camera_dist": 0.8,
            "camera_height": 1.5,
            "camera_pitch": None,
            "camera_fov": 60,
            "interface_panel": ["semantic_camera"],
            "sensors": dict(
                semantic_camera=(SemanticCamera, 1600, 900),
                depth_camera=(DepthCamera, 800, 600),
                rgb_camera=(RGBCamera, 800, 600),
            ),

            # ===== Remove useless items in the images =====
            "show_logo": False,
            "show_fps": False,
            "show_interface": True,
        }
    )

    file_dir = pathlib.Path("images_offscreen")
    file_dir.mkdir(exist_ok=True)

    for ep in range(1):
        env.reset(seed=6)

        # Run it once to initialize the TopDownRenderer
        env.render(
            mode="topdown",
            screen_size=(1600, 900),
            film_size=(9000, 9000),
            target_vehicle_heading_up=True,
            semantic_map=True,
        )

        for t in range(10000):

            # We don't have interface in offscreen. So comment out:
            # env.capture("rgb_deluxe_{}_{}.jpg".format(env.current_seed, t))

            ret = env.render(
                mode="topdown",
                screen_size=(1600, 900),
                film_size=(9000, 9000),
                target_vehicle_heading_up=True,
                semantic_map=True,
                to_image=False
            )
            pygame.image.save(ret, str(file_dir / "bev_{}.png".format(t)))
            env.engine.get_sensor("depth_camera").save_image(env.agent, str(file_dir / "depth_{}.jpg".format(t)))
            env.engine.get_sensor("rgb_camera").save_image(env.agent, str(file_dir / "rgb_{}.jpg".format(t)))
            env.engine.get_sensor("semantic_camera").save_image(env.agent, str(file_dir / "semantic_{}.jpg".format(t)))
            print("Image at step {} is saved at: {}".format(t, file_dir))
            if t == 30:
                break
            env.step([1, 0.88])
