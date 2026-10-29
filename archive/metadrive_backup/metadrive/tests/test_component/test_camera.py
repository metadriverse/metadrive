import numpy as np
import cv2

from metadrive import MetaDriveEnv


def _test_main_camera_as_obs(render):
    try:
        env = MetaDriveEnv(
            dict(
                num_scenarios=1000,
                start_seed=1010,
                traffic_density=0.05,
                image_observation=True,
                use_render=False,
                vehicle_config=dict(image_source="main_camera"),
                show_interface=False,
                show_logo=False,
                show_fps=False,
            )
        )
        obs, _ = env.reset()
        action = [0.0, 1.]
        for _ in range(3):
            env.reset()
            for s in range(20):
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
                assert np.sum(o["image"][..., -1]) > 10
                assert not env.engine.interface.need_interface
                if render:
                    cv2.imshow("window", o["image"][..., -1])
                    cv2.waitKey(1)
    finally:
        env.close()


def _test_rgb_camera_as_obs():
    try:
        env = MetaDriveEnv(
            dict(
                num_scenarios=1000,
                start_seed=1010,
                traffic_density=0.05,
                image_observation=True,
                use_render=False,
                vehicle_config=dict(image_source="rgb_camera"),
                show_interface=False,
                show_logo=False,
                show_fps=False,
            )
        )
        obs, _ = env.reset()
        action = [0.0, 1.]
        for _ in range(3):
            env.reset()
            for s in range(20):
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
                assert np.sum(o["image"][..., -1]) > 10
                # cv2.waitKey(1)
    finally:
        env.close()


if __name__ == "__main__":
    _test_main_camera_as_obs(True)
