import time
import cv2

import numpy as np
from metadrive import MetaDriveEnv


def _test_rgb_camera_as_obs(render=False):
    env = MetaDriveEnv(
        dict(
            num_scenarios=1,
            start_seed=1010,
            traffic_density=0.0,
            image_observation=True,
            use_render=False,
            vehicle_config=dict(image_source="rgb_camera", rgb_camera=(1200, 800)),
            show_interface=False,
            show_logo=False,
            show_fps=False,
        )
    )
    obs, _ = env.reset()
    action = [0.0, 0.1]
    start = time.time()
    for s in range(20000):
        o, r, tm, tc, i = env.step(action)
        # engine = env.engine
        # if engine.episode_step <= 1:
        #     engine.graphicsEngine.renderFrame()
        # origin_img = engine.win.getDisplayRegion(0).getScreenshot()
        # v = memoryview(origin_img.getRamImage).tolist()
        # img = np.array(v)
        # img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), 4))
        # img = img[::-1]
        # img = img[..., :-1]
        # img = img/255
        if render:
            cv2.imshow("window", o["image"][..., -1])
            cv2.waitKey(1)
        #
        # if i % 100 == 0 and i != 0:
        # print("FPS: {}".format(i / (time.time() - start)))


if __name__ == "__main__":
    _test_rgb_camera_as_obs(True)
