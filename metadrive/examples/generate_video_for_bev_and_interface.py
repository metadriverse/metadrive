"""
This file will run environment for one episode and generate two videos:

example_video_RANDOM-STRING/0_bev.mp4
example_video_RANDOM-STRING/0_interface.mp4

This file demonstrates how to use API to generate high-resolution video for demo.
"""
import os
import uuid

import cv2
import matplotlib.pyplot as plt
import pygame

from metadrive.envs import MetaDriveEnv


def plot(d):
    # Can use this function to show images, if you wish to set breakpoints in this file.
    plt.imshow(d)
    plt.show()


class VideoRecorder:
    def __init__(self, video_name, code, height, width, fps=40):
        """
        code=mp4v, avc1, x264, h264 etc.
        """
        assert video_name.endswith(".mp4")
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*code), fps, (width, height))
        self.video_writer = video

    def write(self, image):
        self.video_writer.write(image[..., ::-1])

    def finish(self):
        self.video_writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print("Error happen: ", e)


if __name__ == '__main__':
    num_scenarios = 1  # Will run each seed for 1 episode.
    start_seed = 100
    generate_video = True

    if generate_video:
        folder_name = "example_video_{}".format(str(uuid.uuid4())[:6])

    env = MetaDriveEnv(
        dict(
            num_scenarios=num_scenarios,
            start_seed=start_seed,
            random_traffic=False,
            use_render=True,
            window_size=(1600, 1200),
            crash_vehicle_done=False,
            manual_control=True,  # For using expert policy. You don't need to control it.
        )
    )

    ep_count = 0
    step_count = 0
    frame_count = 0

    o = env.reset(force_seed=start_seed)
    env.vehicle.expert_takeover = True
    env.engine.force_fps.disable()

    if generate_video:
        os.makedirs(folder_name, exist_ok=True)

        video_base_name = "{}/{}".format(folder_name, ep_count)
        video_name_bev = video_base_name + "_bev.mp4"
        video_bev = VideoRecorder(video_name=video_name_bev, height=3000, width=3000, code="avc1")
        # video_list_bev = []
        video_name_interface = video_base_name + "_interface.mp4"
        # video_list_interface = []
        video_interface = VideoRecorder(video_name=video_name_interface, height=1200, width=1600, code="avc1")

    while True:
        o, r, d, info = env.step(env.action_space.sample())

        img_interface = env.render("rgb_array")
        img_bev = env.render(
            "topdown",
            track_target_vehicle=False,
            draw_target_vehicle_trajectory=True,
            film_size=(3000, 3000),
            screen_size=(3000, 3000),
            crash_vehicle_done=False,
        )

        if generate_video:
            # Option 1: Directly feed image to video stream. Can save IO time and memory.
            # Has the risk to generate broken video.
            img_bev = pygame.surfarray.array3d(img_bev)
            img_bev = img_bev.swapaxes(0, 1)
            video_bev.write(img_bev)
            video_interface.write(img_interface)

            # Option 2: Save image to local temp folder and generate video later. Waste IO time but save memory.
            # The most safe method.
            # pygame.image.save(img_bev, "{}/{}.png".format(tmp_folder_bev, frame_count))
            # env.capture(file_name="{}/{}.png".format(tmp_folder_interface, frame_count))

            # Note that the interface image can also be used like this:
            # pygame.image.save(img_interface, "{}/{}.png".format(tmp_folder_interface, frame_count))

        frame_count += 1
        step_count += 1

        if d or step_count > 1000:
            ep_count += 1
            step_count = 0

            env.engine.force_fps.disable()
            if generate_video:
                video_bev.finish()
                video_interface.finish()
                # image_list_to_video(video_name_bev, video_list_bev, code="avc1")
                print("BEV Video is saved at: ", video_name_bev)
                # image_list_to_video(video_name_interface, video_list_interface, code="avc1")
                print("Interface Video is saved at: ", video_name_interface)

                video_base_name = "{}/{}".format(folder_name, ep_count)
                video_name_bev = video_base_name + "_bev.mp4"
                # video_list_bev = []
                video_name_interface = video_base_name + "_interface.mp4"
                # video_list_interface = []

            if ep_count >= num_scenarios:
                break

            if generate_video:
                video_bev = VideoRecorder(video_name=video_name_bev, height=3000, width=3000, code="avc1")
                video_interface = VideoRecorder(video_name=video_name_interface, height=1200, width=1600, code="avc1")

            o = env.reset(force_seed=ep_count + start_seed)
            env.vehicle.expert_takeover = True
            env.engine.force_fps.disable()

    # if generate_video:
    # from metadrive.utils.image_to_video import image_list_to_video
    # import shutil
    # For Option 2: You need to send the temp folder to video recorder.
    # It will read all image files and generate video.
    # Then delete the temp folder.

    # image_list_to_video(video_name_interface, tmp_folder_interface, code="avc1")
    # print("Interface Video is saved at: ", video_name_interface)
    # shutil.rmtree(tmp_folder_interface)
    #
    # image_list_to_video(video_name_bev, tmp_folder_bev, code="avc1")
    # print("BEV Video is saved at: ", video_name_bev)
    # shutil.rmtree(tmp_folder_bev)
