import time
import cv2

import numpy as np
from metadrive import MetaDriveEnv


def _test_rgb_camera_as_obs(render=False):
    env = MetaDriveEnv(
        dict(
            environment_num=1,
            start_seed=1010,
            traffic_density=0.0,
            offscreen_render=True,
            use_render=False,
            vehicle_config=dict(image_source="main_camera"),
            show_interface=True,
            show_logo=False,
            show_fps=False,
        )
    )
    obs = env.reset()
    action = [0.0, 0.1]
    start = time.time()
    for i in range(20000):
        o, r, d, _ = env.step(action)
        if render:
            cv2.imshow("window", o["image"][..., -1])
            cv2.waitKey(1)
        if i % 100 == 0 and i != 0:
            print("FPS: {}".format(i / (time.time() - start)))


if __name__ == "__main__":
    _test_rgb_camera_as_obs(False)
