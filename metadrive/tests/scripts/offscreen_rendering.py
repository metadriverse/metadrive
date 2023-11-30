import time

import matplotlib.pyplot as plt
import numpy as np

from metadrive.envs.metadrive_env import MetaDriveEnv
import time

if __name__ == '__main__':
    W, H = 1920, 1200
    config = dict(

        # use_render=True,
        image_observation=True,
        manual_control=True,  # set false for external subscriber control
        traffic_density=0.0,
        num_scenarios=100,
        random_agent_model=True,
        random_lane_width=True,
        random_lane_num=True,
        vehicle_config=dict(image_source="rgb_camera", rgb_camera=(W, H), stack_size=1),
        map=4,  # seven block
        start_seed=0,
        window_size=(300, 200),
    )
    env = MetaDriveEnv(config)
    start = time.time()
    env.reset()
    frames = []
    for num_frames in range(100):
        o, r, tm, tc, info = env.step([0, 1])
        frame = o['image']
        frame = frame[..., 0]  # Original return frame is [1200, 1920, 3, 1] (float), so remove last dim.
        # frame = 1 - frame
        # frame *= 255
        # frame = frame.astype(np.uint8)
        # print(f"Finish {num_frames + 1} frames")
        # plt.imshow(frame)
        # plt.show()
        # print(
        #     "Finish {}/100 simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}".format(
        #         num_frames + 1,
        #         time.time() - start, (num_frames + 1) / (time.time() - start)
        #     )
        # )
    env.close()
