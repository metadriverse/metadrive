#!/usr/bin/env python
import argparse
import time

torch_available = True
import cv2
try:
    import torch
    from torch.utils.dlpack import from_dlpack
except ImportError:
    torch_available = False
    print("Can not find torch")
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive import MetaDriveEnv
import os
from metadrive.policy.idm_policy import IDMPolicy


def _test_rgb_camera_as_obs(render=False, image_on_cuda=True, debug=False, camera="main"):
    res = (800, 600)
    mapping = {
        "depth": {
            "depth_camera": (DepthCamera, *res)
        },
        "rgb": {
            "rgb_camera": (RGBCamera, *res)
        },
        "semantic": {
            "semantic_camera": (SemanticCamera, *res)
        },
        "main": {
            "main_camera": ()
        },
    }

    env = MetaDriveEnv(
        dict(
            show_terrain="METADRIVE_TEST_EXAMPLE" not in os.environ,
            num_scenarios=1,
            start_seed=1010,
            agent_policy=IDMPolicy,
            traffic_density=0.0,
            image_observation=True,
            image_on_cuda=True if image_on_cuda else False,
            use_render=False,
            vehicle_config=dict(image_source="{}_camera".format(camera)),
            sensors=mapping[camera],
            show_interface=False,
            show_logo=False,
            show_fps=False,
        )
    )
    print(
        "Use {} with resolution {}".format(
            env.config["vehicle_config"]["image_source"],
            (env.observation_space["image"].shape[0], env.observation_space["image"].shape[1])
        )
    )
    o, i = env.reset()

    # for debug
    if debug:
        ret = o["image"].get()[..., -1] if env.config["image_on_cuda"] else o["image"][..., -1]
        cv2.imwrite("reset_frame.png", ret * 255)

    action = [0.0, 0.1]

    fps_cal_start_frame = 5
    for i in range(20000):
        o, r, d, _, _ = env.step(action)
        if i == fps_cal_start_frame:
            # the first several frames may be slow. Ignore them when calculating FPS
            start = time.time()
        if image_on_cuda:
            torch_tensor = from_dlpack(o["image"].toDlpack())
        elif torch_available:
            torch_tensor = torch.Tensor(o["image"])

        if debug:
            cv2.imwrite(
                "{}_frame.png".format(i),
                (o["image"].get()[..., -1] if env.config["image_on_cuda"] else o["image"][..., -1]) * 255
            )

        if render:
            ret = o["image"].get()[..., -1] if env.config["image_on_cuda"] else o["image"][..., -1]
            cv2.imshow("window", ret)
            cv2.waitKey(1)
        if d:
            print("FPS: {}".format((i - fps_cal_start_frame) / (time.time() - start)))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--camera", default="main", choices=["main", "rgb", "depth", "semantic"])
    args = parser.parse_args()
    if args.cuda:
        assert torch_available, "You have to install torch to use CUDA"
    _test_rgb_camera_as_obs(args.render, image_on_cuda=args.cuda, debug=args.debug, camera=args.camera)
    print(
        "Test Successful !! The FPS should go beyond 400 FPS, if you are using CUDA in offscreen mode "
        "with GPUs better than RTX 3060."
    )
