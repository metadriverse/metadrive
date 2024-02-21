#!/usr/bin/env python
"""
This file will run environment for one episode and generate two videos:

example_video_RANDOM-STRING/0_bev.mp4
example_video_RANDOM-STRING/0_interface.mp4

This file demonstrates how to use API to generate high-resolution & temporal aligned videos for visualization.

Dependencies:
    pip install mediapy
    conda install ffmpeg
    https://itsfoss.com/install-h-264-decoder-ubuntu/  (h264 decoder in Linux, or you can install VLC)
"""
import os
from datetime import datetime

import mediapy
import pygame

from metadrive.envs import MetaDriveEnv

if __name__ == '__main__':
    num_scenarios = 1  # Will run each seed for 1 episode.
    start_seed = 100
    generate_video = True

    folder_name = "example_video_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    video_bev = []
    video_interface = []

    env = MetaDriveEnv(
        dict(
            show_terrain="METADRIVE_TEST_EXAMPLE" not in os.environ,
            num_scenarios=num_scenarios,
            start_seed=start_seed,
            random_traffic=False,
            use_render=True,
            window_size=(900, 600),
            crash_vehicle_done=False,
            manual_control=True,  # For using expert policy. You don't need to control it.
            horizon=100,
        )
    )

    ep_count = 0
    step_count = 0
    frame_count = 0

    o, _ = env.reset(seed=start_seed)
    env.agent.expert_takeover = True
    env.engine.force_fps.disable()

    while True:
        o, r, tm, tc, info = env.step(env.action_space.sample())

        img_interface = env.main_camera.perceive(to_float=False)
        # BGR to RGB
        img_interface = img_interface[..., ::-1]
        img_bev = env.render(
            mode="topdown",
            target_vehicle_heading_up=False,
            draw_target_vehicle_trajectory=True,
            film_size=(3000, 3000),
            screen_size=(800, 800),
        )

        if generate_video:
            img_bev = img_bev.swapaxes(0, 1)
            video_bev.append(img_bev)
            video_interface.append(img_interface)

        frame_count += 1
        step_count += 1

        if tm or tc or step_count > 1000:
            ep_count += 1
            step_count = 0

            env.engine.force_fps.disable()
            if generate_video:
                os.makedirs(folder_name, exist_ok=True)

                video_base_name = "{}/{}".format(folder_name, ep_count)
                video_name_bev = video_base_name + "_bev.mp4"
                print("BEV video should be saved at: ", video_name_bev)
                mediapy.write_video(video_name_bev, video_bev, fps=60)

                video_name_interface = video_base_name + "_interface.mp4"
                print("Interface video should be saved at: ", video_name_interface)
                mediapy.write_video(video_name_interface, video_interface, fps=60)

            if ep_count >= num_scenarios:
                break

            o, _ = env.reset(seed=ep_count + start_seed)
            env.agent.expert_takeover = True
            env.engine.force_fps.disable()
