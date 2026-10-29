from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.top_down_obs import TopDownObservation
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.utils import Config


class TopDownSingleFrameMetaDriveEnv(MetaDriveEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = MetaDriveEnv.default_config()
        # config["vehicle_config"]["lidar"].update({"num_lasers": 0, "distance": 0})  # Remove lidar
        config.update(
            {
                "frame_skip": 5,
                "frame_stack": 3,
                "post_stack": 5,
                "norm_pixel": True,
                "resolution_size": 84,
                "distance": 30
            }
        )
        return config

    def get_single_observation(self, _=None):
        return TopDownObservation(
            self.config["vehicle_config"],
            self.config["norm_pixel"],
            onscreen=self.config["use_render"],
            max_distance=self.config["distance"]
        )


class TopDownMetaDrive(TopDownSingleFrameMetaDriveEnv):
    def get_single_observation(self, _=None):
        return TopDownMultiChannel(
            self.config["vehicle_config"],
            onscreen=self.config["use_render"],
            clip_rgb=self.config["norm_pixel"],
            frame_stack=self.config["frame_stack"],
            post_stack=self.config["post_stack"],
            frame_skip=self.config["frame_skip"],
            resolution=(self.config["resolution_size"], self.config["resolution_size"]),
            max_distance=self.config["distance"]
        )


class TopDownMetaDriveEnvV2(MetaDriveEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = MetaDriveEnv.default_config()
        config["vehicle_config"]["lidar"] = {"num_lasers": 0, "distance": 0}  # Remove lidar
        config.update(
            {
                "frame_skip": 5,
                "frame_stack": 3,
                "post_stack": 5,
                "norm_pixel": True,
                "resolution_size": 84,
                "distance": 30
            }
        )
        return config

    def get_single_observation(self, _=None):
        return TopDownMultiChannel(
            self.config["vehicle_config"],
            self.config["use_render"],
            self.config["norm_pixel"],
            frame_stack=self.config["frame_stack"],
            post_stack=self.config["post_stack"],
            frame_skip=self.config["frame_skip"],
            resolution=(self.config["resolution_size"], self.config["resolution_size"]),
            max_distance=self.config["distance"]
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Test single RGB frame
    # env = TopDownSingleFrameMetaDriveEnv(dict(use_render=True, num_scenarios=1, map="C", traffic_density=1.0))
    # env.reset()
    # for _ in range(20):
    #     o, *_ = env.step([0, 1])
    #     assert env.observation_space.contains(o)
    # for _ in range(200):
    #     o, *_ = env.step([0.01, 1])
    #     env.render()
    #     # plt.imshow(o, cmap="gray")
    #     # plt.show()
    #     # print(o.mean())
    # env.close()

    # Test multi-channel frames
    env = TopDownMetaDriveEnvV2(dict(num_scenarios=1, start_seed=5000, distance=30))
    # env = TopDownMetaDrive(dict(num_scenarios=1, map="XTO", traffic_density=0.1, frame_stack=5))
    # env = TopDownMetaDrive(dict(use_render=True, manual_control=True))
    env.reset()
    names = [
        "road_network", "navigation", "past_pos", "traffic t", "traffic t-1", "traffic t-2", "traffic t-3",
        "traffic t-4"
    ]
    for _ in range(60):
        o, *_ = env.step([-0.00, 0.2])
        assert env.observation_space.contains(o)
    for _ in range(10000):
        o, r, tm, tc, i = env.step([0.0, 1])
        print("Velocity: ", i["velocity"])

        fig, axes = plt.subplots(1, o.shape[-1], figsize=(15, 3))

        # o = env.observations[env.DEFAULT_AGENT].get_screen_window()
        # import numpy as np
        # import pygame
        # o = pygame.surfarray.array3d(o)
        # o = np.transpose(o, (1, 0, 2))
        # axes[0].imshow(o)

        for o_i in range(o.shape[-1]):
            axes[o_i].imshow(o[..., o_i], cmap="gray", vmin=0, vmax=1)
            axes[o_i].set_title(names[o_i])

        fig.suptitle("Multi-channel Top-down Observation")
        plt.show()
        print(o.mean())
    env.close()
