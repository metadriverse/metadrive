import os
import os.path as osp
import uuid

import cv2
import matplotlib.pyplot as plt
import pygame


def plot(d):
    # Can use this function to show images, if you wish to set breakpoints in this file.
    plt.imshow(d)
    plt.show()


class VideoRecorder:

    def __init__(self, video_name, code, height, width):
        """
        code=mp4v, avc1, x264, h264 etc.
        """
        assert video_name.endswith(".mp4")
        # assert len(image_list) > 0
        # frame = cv2.imread(os.path.join(image_folder, images[0]))
        # frame = image_list[0]
        # height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*code), 40, (width, height))
        self.video_writer = video
        # for image in tqdm(image_list, desc="Writing Video"):

        # Change color
        # image = image[..., ::-1]

        # video.write(image)

    def write(self, image):
        self.video_writer.write(image[..., ::-1])

    def finish(self):
        self.video_writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print("Error happen: ", e)


if __name__ == '__main__':
    ckpt_path = osp.join(
        # "PPO_MetaDriveEnv_87e53_00005_5_environment_num=3,start_seed=5000,traffic_density=0.3000_2023-02-22_22-10-18/checkpoint_000460")
        "PPO_MetaDriveEnv_87e53_00001_1_environment_num=3,start_seed=5000,traffic_density=0.1000_2023-02-22_22-09-04/checkpoint_000500"
        # "PPO_MetaDriveEnv_87e53_00001_1_environment_num=3,start_seed=5000,traffic_density=0.1000_2023-02-22_22-09-04/checkpoint_000500"
        # "PPO_MetaDriveEnv_87e53_00006_6_environment_num=5,start_seed=5000,traffic_density=0.3000_2023-02-22_22-10-34/checkpoint_000500"
        # "PPO_MetaDriveEnv_87e53_00002_2_environment_num=5,start_seed=5000,traffic_density=0.1000_2023-02-22_22-09-22/checkpoint_000500"
        # "PPO_MetaDriveEnv_87e53_00007_7_environment_num=1000,start_seed=5000,traffic_density=0.3000_2023-02-22_22-10-50/checkpoint_000500"
        # "PPO_MetaDriveEnv_87e53_00000_0_environment_num=1,start_seed=5000,traffic_density=0.1000_2023-02-22_22-08-46/checkpoint_000500"
    )
    traffic_density = 0.2
    # traffic_density = 0.1
    environment_num = 5
    # environment_num = 100
    start_seed = 100
    # start_seed = 0
    # num_ep = 10
    generate_video = True

    if generate_video:
        # folder_name = "td0.3_training_visualization_video_{}".format(str(uuid.uuid4())[:6])
        # folder_name = "td0.3_test_visualization_video_{}".format(str(uuid.uuid4())[:6])
        # folder_name = "td0.1_test_visualization_video_{}".format(str(uuid.uuid4())[:6])
        # folder_name = "td0.1_training_video_{}".format(str(uuid.uuid4())[:6])
        # folder_name = "td0.3_env5_training_video_{}".format(str(uuid.uuid4())[:6])
        # folder_name = "td0.1_env5_training_video_{}".format(str(uuid.uuid4())[:6])
        # folder_name = "expert_training_video_{}".format(str(uuid.uuid4())[:6])
        # folder_name = "td0.1_env1_test_video_{}".format(str(uuid.uuid4())[:6])
        folder_name = "td0.1_env5_test_video_{}".format(str(uuid.uuid4())[:6])

    # ==========
    import torch
    from ray.rllib.algorithms.ppo import PPO
    from metadrive.envs import MetaDriveEnv

    agent = PPO(dict(

        # ===== Training Environment =====
        # Train the policies in scenario sets with different number of scenarios.
        env=MetaDriveEnv,
        env_config=dict(
            traffic_density=traffic_density,
            environment_num=environment_num,
            start_seed=start_seed,
            random_traffic=False,
            use_render=True,
            window_size=(1600, 1200),
            crash_vehicle_done=False,

            # manual_control=True,
        ),

        # ===== Framework =====
        framework="torch",

        num_workers=0,

        # ===== Resources Specification =====
        num_gpus=0.25 if torch.cuda.is_available() else 0,
        num_cpus_per_worker=0.2,
        num_cpus_for_driver=0.5
    ))
    agent.restore(ckpt_path)
    env = agent.workers.local_worker().env

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

                video_bev = VideoRecorder(video_name=video_name_bev, height=3000, width=3000, code="avc1")
                video_interface = VideoRecorder(video_name=video_name_interface, height=1200, width=1600, code="avc1")

            if ep_count >= environment_num:
                break

            o = env.reset(force_seed=ep_count + start_seed)
            env.vehicle.expert_takeover = True

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
