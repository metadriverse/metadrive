"""
Test example in metadrive/documentation/source/simgen_render.ipynb
"""
import os
import time

import cv2
import gymnasium as gym
import mediapy as media
import numpy as np
import tqdm
from PIL import Image
from PIL import ImageDraw, ImageFont
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.obs.image_obs import ImageObservation
from metadrive.obs.observation_base import BaseObservation
from metadrive.policy.replay_policy import ReplayEgoCarPolicy


def test_simgen():
    class SimGenObservation(BaseObservation):
        def __init__(self, config):
            super(SimGenObservation, self).__init__(config)
            assert config["norm_pixel"] is False
            assert config["stack_size"] == 1
            self.seg_obs = ImageObservation(config, "seg_camera", config["norm_pixel"])
            self.rgb_obs = ImageObservation(config, "rgb_camera", config["norm_pixel"])
            self.depth_obs = ImageObservation(config, "depth_camera", config["norm_pixel"])

        @property
        def observation_space(self):
            os = dict(
                rgb=self.rgb_obs.observation_space,
                seg=self.seg_obs.observation_space,
                depth=self.depth_obs.observation_space,
            )
            return gym.spaces.Dict(os)

        def observe(self, vehicle):
            ret = {}

            seg_cam = self.engine.get_sensor("seg_camera").cam
            agent = seg_cam.getParent()
            original_position = seg_cam.getPos()
            heading, pitch, roll = seg_cam.getHpr()
            seg_img = self.seg_obs.observe(agent, position=original_position, hpr=[heading, pitch, roll])
            assert seg_img.ndim == 4
            assert seg_img.shape[-1] == 1
            assert seg_img.dtype == np.uint8
            # Do some postprocessing here
            seg_img = seg_img[..., 0]
            before = seg_img.copy()
            # seg_img = postprocess_semantic_image(seg_img)
            seg_img = seg_img[..., ::-1]  # BGR -> RGB
            ret["seg"] = seg_img

            depth_cam = self.engine.get_sensor("depth_camera").cam
            agent = depth_cam.getParent()
            original_position = depth_cam.getPos()
            heading, pitch, roll = depth_cam.getHpr()
            depth_img = self.depth_obs.observe(agent, position=original_position, hpr=[heading, pitch, roll])
            assert depth_img.ndim == 4
            assert depth_img.shape[-1] == 1
            assert depth_img.dtype == np.uint8
            depth_img = depth_img[..., 0]
            # before = depth_img.copy()
            depth_img = cv2.bitwise_not(depth_img)
            depth_img = depth_img[..., None]
            ret["depth"] = depth_img

            rgb_cam = self.engine.get_sensor("rgb_camera").cam
            agent = rgb_cam.getParent()
            original_position = rgb_cam.getPos()
            heading, pitch, roll = rgb_cam.getHpr()
            rgb_img = self.rgb_obs.observe(agent, position=original_position, hpr=[heading, pitch, roll])
            assert rgb_img.ndim == 4
            assert rgb_img.shape[-1] == 1
            assert rgb_img.dtype == np.uint8
            rgb_img = rgb_img[..., 0]
            # Change the color from BGR to RGB
            rgb_img = rgb_img[..., ::-1]
            ret["rgb"] = rgb_img

            return ret

    # ===== MetaDrive Setup =====

    sensor_size = (80, 45)  #if os.getenv('TEST_DOC') else (800, 450)

    env = ScenarioEnv(
        {
            'agent_observation': SimGenObservation,

            # To enable onscreen rendering, set this config to True.
            "use_render": False,

            # !!!!! To enable offscreen rendering, set this config to True !!!!!
            "image_observation": True,
            "norm_pixel": False,
            "stack_size": 1,

            # ===== The scenario and MetaDrive config =====
            "agent_policy": ReplayEgoCarPolicy,
            "no_traffic": False,
            "sequential_seed": True,
            "reactive_traffic": False,
            "num_scenarios": 9,
            "horizon": 1000,
            "no_static_vehicles": False,
            "agent_configs": {
                "default_agent": dict(use_special_color=True, vehicle_model="varying_dynamics_bounding_box")
            },
            "vehicle_config": dict(
                show_navi_mark=False,
                show_line_to_dest=False,
                lidar=dict(num_lasers=120, distance=50),
                lane_line_detector=dict(num_lasers=0, distance=50),
                side_detector=dict(num_lasers=12, distance=50),
            ),
            # "use_bounding_box": True,
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
            "height_scale": 1,
            "set_static": True,

            # ===== Set some sensor and visualization configs =====
            "daytime": "08:10",
            "window_size": (800, 450),
            "camera_dist": 0.8,  # 0.8, 1.71
            "camera_height": 1.5,  # 1.5
            "camera_pitch": None,
            "camera_fov": 66,  # 60, 66
            "sensors": dict(
                depth_camera=(DepthCamera, sensor_size[0], sensor_size[1]),
                rgb_camera=(RGBCamera, sensor_size[0], sensor_size[1]),
                seg_camera=(SemanticCamera, sensor_size[0], sensor_size[1]),
            ),

            # ===== Remove useless items in the images =====
            "show_logo": False,
            "show_fps": False,
            "show_interface": True,
            "disable_collision": True,
            "force_destroy": True,
        }
    )

    skip_steps = 1
    fps = 10

    frames = []

    try:
        env.reset()
        scenario = env.engine.data_manager.current_scenario
        scenario_id = scenario['id']
        print(
            "Current scenario ID {}, dataset version {}, len: {}".format(
                scenario_id, scenario['version'], scenario['length']
            )
        )
        # horizon = scenario['length']
        horizon = 10

        for t in tqdm.trange(horizon):
            o, r, d, _, _ = env.step([1, 0.88])
            if t % skip_steps == 0:
                depth_img = Image.fromarray(o["depth"].repeat(3, axis=-1), mode="RGB")
                seg_img = Image.fromarray(o["seg"], mode="RGB")
                rgb_img = Image.fromarray(o["rgb"], mode="RGB")

                assert not (o["seg"] == 255).all()
                assert not (o["seg"] == 0).all()
                assert not (o["depth"] == 255).all()
                assert not (o["depth"] == 0).all()
                assert not (o["rgb"] == 255).all()
                assert not (o["rgb"] == 0).all()

    finally:
        env.close()

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
                depth_camera=(DepthCamera, 800, 600),
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
    test_simgen()
