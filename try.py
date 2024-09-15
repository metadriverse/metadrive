# visualization
# from IPython.display import Image as IImage
import pygame
import numpy as np
from PIL import Image

def make_GIF(frames, name="demo.gif"):
    print("Generate gif...")
    imgs = [frame for frame in frames]
    imgs = [Image.fromarray(img) for img in imgs]
    imgs[0].save(name, save_all=True, append_images=imgs[1:], duration=50, loop=0)

    #@title Make some configurations and import some modules
from metadrive.engine.engine_utils import close_engine
close_engine()
from metadrive.pull_asset import pull_asset
pull_asset(False)
# NOTE: usually you don't need the above lines. It is only for avoiding a potential bug when running on colab

from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.envs.metadrive_env import MetaDriveEnv
import os

threeD_render=False # turn on this to enable 3D render. It only works when you have a screen and not running on Colab.
threeD_render=threeD_render and not RunningInCOLAB
os.environ["SDL_VIDEODRIVER"] = "dummy" # Hide the pygame window
waymo_data =  AssetLoader.file_path(AssetLoader.asset_path, "waymo", unix_style=False) # Use the built-in datasets with simulator
nuscenes_data =  AssetLoader.file_path(AssetLoader.asset_path, "nuscenes", unix_style=False) # Use the built-in datasets with simulator

import pygame
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.engine.asset_loader import AssetLoader


sensor_size = (84, 60) if os.getenv('TEST_DOC') else (200, 100)
from metadrive.component.sensors.rgb_camera import RGBCamera
# env = ScenarioEnv(
#     {
#         # "manual_control": False,
#         # "reactive_traffic": False,
#         # "use_render": threeD_render,
#         "agent_policy": ReplayEgoCarPolicy,
#         "data_directory": waymo_data,
#         "num_scenarios": 1,
#         "image_observation":True, 
#         "vehicle_config":dict(image_source="rgb_camera"),
#         "sensors":{"rgb_camera": (RGBCamera, *sensor_size)},
#         "stack_size":3,
#     }
# )

cfg=dict(image_observation=True, 
         vehicle_config=dict(image_source="rgb_camera"),
         sensors={"rgb_camera": (RGBCamera, *sensor_size)},
         stack_size=3,
         agent_policy=IDMPolicy # drive with IDM policy
        )

env=MetaDriveEnv(cfg)

# @title Run Simulation

env.reset(seed=0)
frames = []
for i in range(1, 100000):
    o, r, tm, tc, info = env.step([1.0, 0.])
    frames.append(env.render(mode="topdown",film_size=(1200, 1200)))
    # ret=o["image"][..., -1]*255 # [0., 1.] to [0, 255]
    # ret=ret.astype(np.uint8)
    # frames.append(ret[..., ::-1])
    print(o)
    if tm or tc:
        break
env.close()

make_GIF(frames)
# visualization
Image(open("demo.gif", 'rb').read())