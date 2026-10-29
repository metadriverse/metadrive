import time

import cv2

from metadrive import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy


def _test_depth_camera_as_obs(render=False):
    env = MetaDriveEnv(
        dict(
            num_scenarios=1,
            start_seed=1010,
            agent_policy=IDMPolicy,
            traffic_density=0.0,
            image_observation=True,
            image_on_cuda=True,
            use_render=False,
            vehicle_config=dict(image_source="depth_camera", depth_camera=(800, 600)),
            show_interface=True,
            show_logo=False,
            show_fps=False,
        )
    )
    obs, _ = env.reset()
    action = [0.0, 0.1]
    start = time.time()
    for i in range(20000):
        o, r, tm, tc, _ = env.step(action)
        if render:
            ret = o["image"].get()[..., -1] if env.config["image_on_cuda"] else o["image"][..., -1]
            cv2.imshow("window", ret)
            cv2.waitKey(1)
        # if d:
        # print("FPS: {}".format(i / (time.time() - start)))
        # env.reset()
        # break


def _test_main_rgb_camera_as_obs_with_interface(render=False):
    env = MetaDriveEnv(
        dict(
            num_scenarios=1,
            start_seed=1010,
            agent_policy=IDMPolicy,
            traffic_density=0.0,
            image_observation=True,
            image_on_cuda=True,
            use_render=False,
            vehicle_config=dict(image_source="main_camera", rgb_camera=(800, 600)),
            show_interface=True,
            show_logo=False,
            show_fps=False,
        )
    )
    obs, _ = env.reset()
    action = [0.0, 0.1]
    start = time.time()
    reset_time = 0
    for i in range(20000):
        o, r, tm, tc, _ = env.step(action)
        if render:
            ret = o["image"].get()[..., -1] if env.config["image_on_cuda"] else o["image"][..., -1]
            cv2.imshow("window", ret)
            cv2.waitKey(1)
        if tm or tc:
            current = time.time()
            # env.reset()
            # reset_time += time.time()-current
            # print("FPS: {}".format(i / (current - start - reset_time)))
            break


def _test_main_rgb_camera_no_interface(render=False):
    env = MetaDriveEnv(
        dict(
            num_scenarios=1,
            start_seed=1010,
            agent_policy=IDMPolicy,
            traffic_density=0.0,
            image_observation=True,
            image_on_cuda=True,
            use_render=False,
            vehicle_config=dict(image_source="main_camera", rgb_camera=(800, 600)),
            show_interface=False,
            show_logo=False,
            show_fps=False,
        )
    )
    obs, _ = env.reset()
    action = [0.0, 0.1]
    start = time.time()
    for i in range(20000):
        o, r, tm, tc, _ = env.step(action)
        if render:
            ret = o["image"].get()[..., -1] if env.config["image_on_cuda"] else o["image"][..., -1]
            cv2.imshow("window", ret)
            cv2.waitKey(1)
        if tm or tc:
            # print("FPS: {}".format(i / (time.time() - start)))
            # env.reset()
            break


def _test_rgb_camera_as_obs(render=False):
    env = MetaDriveEnv(
        dict(
            num_scenarios=1,
            start_seed=1010,
            agent_policy=IDMPolicy,
            traffic_density=0.0,
            image_observation=True,
            image_on_cuda=True,
            use_render=False,
            vehicle_config=dict(image_source="rgb_camera", rgb_camera=(1920, 1080)),
            show_interface=False,
            show_logo=False,
            show_fps=False,
        )
    )
    obs, _ = env.reset()
    action = [0.0, 0.1]
    start = time.time()
    for i in range(20000):
        o, r, tm, tc, _ = env.step(action)
        if render:
            ret = o["image"].get()[..., -1] if env.config["image_on_cuda"] else o["image"][..., -1]
            cv2.imshow("window", ret)
            cv2.waitKey(1)
        if tm or tc:
            # print("FPS: {}".format(i / (time.time() - start)))
            env.reset()
            # break


if __name__ == "__main__":
    # _test_depth_camera_as_obs(False)
    _test_rgb_camera_as_obs(True)
    # _test_main_rgb_camera_as_obs_with_interface(False)
    # _test_main_rgb_camera_no_interface(True)
