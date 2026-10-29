"""
Issue from: https://github.com/metadriverse/metadrive/issues/775
Fixed by: https://github.com/metadriverse/metadrive/pull/788
"""

from metadrive.envs.metadrive_env import MetaDriveEnv
import cv2
import gymnasium as gym
import numpy as np
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.image_obs import ImageObservation
import os
from metadrive.utils import generate_gif
from IPython.display import Image
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.scenario_env import ScenarioEnv


class MultiviewObservation(BaseObservation):
    def __init__(self, config):
        super(MultiviewObservation, self).__init__(config)
        self.rgb = ImageObservation(config, "rgb_camera", config["norm_pixel"])
        self.num_view = 6

    @property
    def observation_space(self):
        os = {"image_{}".format(idx): self.rgb.observation_space for idx in range(self.num_view)}
        # os["top_down"] = self.rgb.observation_space
        return gym.spaces.Dict(os)

    def observe(self, vehicle):
        ret = {}
        for idx in range(self.num_view):
            ret["image_{}".format(idx)] = self.rgb.observe(
                vehicle._node_path_list[-1].parent,
                # render/world_np/VEHICLE/wheel ==parent==> render/world_np/VEHICLE
                hpr=[idx * 60, 0, 0]
            )
        return ret


sensor_size = (640, 360)

# cfg=dict(agent_observation=MultiviewObservation,
#          image_observation=True,
#          vehicle_config=dict(image_source="rgb_camera"),
#          sensors={"rgb_camera": (RGBCamera, *sensor_size)},
#          stack_size=3, # return image shape (H,W,C,stack_size)
#          agent_policy=IDMPolicy, # drive with IDM policy
#          image_on_cuda = True
#         )

# env=MetaDriveEnv(cfg)

# turn on this to enable 3D render. It only works when you have a screen
threeD_render = False
# Use the built-in datasets with simulator
nuscenes_data = AssetLoader.file_path(AssetLoader.asset_path, "nuscenes", unix_style=False)

env = ScenarioEnv(
    {
        "reactive_traffic": False,
        "use_render": threeD_render,
        "agent_policy": ReplayEgoCarPolicy,
        "data_directory": nuscenes_data,
        "num_scenarios": 3,
        "image_observation": True,
        "vehicle_config": dict(image_source="rgb_camera"),
        "sensors": {
            "rgb_camera": (RGBCamera, *sensor_size)
        },
        "stack_size": 3,  # return image shape (H,W,C,stack_size)
        "image_on_cuda": True,
        "agent_observation": MultiviewObservation,
    }
)

frames = []
try:
    env.reset(0)
    for _ in range(1 if os.getenv('TEST_DOC') else 10000):
        # simulation
        o, r, d, _, _ = env.step([0, 1])
        # o, r, tm, tc, info = env.step([1.0, 0.])

        # visualize image observation
        if not env.config['image_on_cuda']:
            ret = cv2.hconcat([v[..., -1] for _, v in o.items()]) * 255
        else:
            ret = cv2.hconcat([v[..., -1].get() for _, v in o.items()]) * 255
        ret = ret.astype(np.uint8)
        frames.append(ret[::2, ::2, ::-1])
        if d:
            break
    # generate_gif(frames if os.getenv('TEST_DOC') else frames[-100:])  # only show -100 frames

    # Save image to disk
    cv2.imwrite("multiview_observation_with_image_on_cuda_no_fix.png", frames[-1])

finally:
    env.close()
