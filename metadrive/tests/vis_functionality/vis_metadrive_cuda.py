import time

import cv2
from metadrive.utils.cuda import restore_channel_and_shape
from metadrive import MetaDriveEnv


def _test_rgb_camera_as_obs(render=False):
    env = MetaDriveEnv(
        dict(
            environment_num=1,
            start_seed=1010,
            traffic_density=0.0,
            offscreen_render=True,
            image_on_cuda=True,
            use_render=False,
            vehicle_config=dict(image_source="rgb_camera", rgb_camera=(800, 600)),
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
            ret = restore_channel_and_shape(o["image"].get()[..., -1]) if env.config["image_on_cuda"] else o["image"][..., -1]
            cv2.imshow("window", ret)
            cv2.waitKey(1)
        if i % 100 == 0 and i != 0:
            print("FPS: {}".format(i / (time.time() - start)))


if __name__ == "__main__":
    _test_rgb_camera_as_obs(True)
