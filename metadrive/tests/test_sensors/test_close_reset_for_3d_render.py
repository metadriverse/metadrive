import logging
from panda3d.core import TexturePool, ModelPool

import psutil

from metadrive.envs.metadrive_env import MetaDriveEnv


def _test_close_reset_for_3d_render():
    """
    The assets should be closed as well
    """

    for i in range(30):
        ## offscreen
        # sensors = {"rgb_camera": (RGBCamera, 1920, 1080)}
        # env = MetaDriveEnv({"image_observation": True, "use_render": False, "force_destroy": True, "preload_models": False, "sensors": sensors, "log_level": logging.CRITICAL})
        ## onscreen
        env = MetaDriveEnv(
            {
                "use_render": True,
                "traffic_density": 0.0,
                "show_skybox": True,
                # "show_terrain": False,
                "show_sidewalk": True,
                # "show_terrain": False,
                "log_level": logging.CRITICAL
            }
        )
        obs, _ = env.reset()
        for _ in range(50):
            env.step(env.action_space.sample())
        env.close()

        del env

        mem = psutil.Process().memory_info().rss
        print(f"Memory usage ({i}): ", mem / (1024**2), "MB")


if __name__ == '__main__':
    _test_close_reset_for_3d_render()
