import logging

import psutil

from metadrive.envs.metadrive_env import MetaDriveEnv


def test_close_reset_for_3d_render():
    """
    The assets should be closed as well
    """

    for i in range(10):
        ## offscreen
        # sensors = {"rgb_camera": (RGBCamera, 1920, 1080)}
        # env = MetaDriveEnv({"image_observation": True, "use_render": False, "force_destroy": True, "preload_models": False, "sensors": sensors, "log_level": logging.CRITICAL})
        ## onscreen
        env = MetaDriveEnv({"use_render": True,
                            "debug_physics_world": True,
                            "log_level": logging.CRITICAL})
        obs, _ = env.reset()
        env.close()

        del env

        mem = psutil.Process().memory_info().rss
        print(f"Memory usage ({i}): ", mem / (1024 ** 2), "MB")


if __name__ == '__main__':
    test_close_reset_for_3d_render()
