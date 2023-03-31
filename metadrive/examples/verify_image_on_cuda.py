import argparse
import time

import cv2
import torch
from torch.utils.dlpack import from_dlpack

from metadrive import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy


def _test_rgb_camera_as_obs(render=False, image_on_cuda=True):
    env = MetaDriveEnv(
        dict(
            num_scenarios=1,
            start_seed=1010,
            agent_policy=IDMPolicy,
            traffic_density=0.0,
            image_observation=True,
            image_on_cuda=True if image_on_cuda else False,
            use_render=False,
            vehicle_config=dict(image_source="main_camera"),
            show_interface=False,
            show_logo=False,
            show_fps=False,
        )
    )
    env.reset()
    action = [0.0, 0.1]
    start = time.time()
    print(
        "Use {} with resolution {}".format(
            env.config["vehicle_config"]["image_source"],
            (env.observation_space["image"].shape[0], env.observation_space["image"].shape[1])
        )
    )
    for i in range(20000):
        o, r, d, _ = env.step(action)
        if image_on_cuda:
            torch_tensor = from_dlpack(o["image"].toDlpack())
        else:
            torch_tensor = torch.Tensor(o["image"])
        if render:
            ret = o["image"].get()[..., -1] if env.config["image_on_cuda"] else o["image"][..., -1]
            cv2.imshow("window", ret)
            cv2.waitKey(1)
        if d:
            print("FPS: {}".format(i / (time.time() - start)))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    _test_rgb_camera_as_obs(args.render, image_on_cuda=not args.native)
    print("Test Successful !!")
