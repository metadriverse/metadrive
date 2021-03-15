from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.pg_config import PGConfig
from pgdrive.constants import DEFAULT_AGENT
from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle
from pgdrive.world.top_down_observation import TopDownMultiChannel, TopDownObservation


class TopDownSingleFramePGDriveEnv(PGDriveEnv):
    @classmethod
    def default_config(cls) -> PGConfig:
        config = PGDriveEnv.default_config()
        config["target_vehicle_configs"][DEFAULT_AGENT]["lidar"] = {"num_lasers": 0, "distance": 0}  # Remove lidar
        config.extend_config_with_unknown_keys({"frame_skip": 5, "frame_stack": 5, "rgb_clip": False})
        return config

    def initialize_observation(self):
        vehicle_config = BaseVehicle.get_vehicle_config(self.config["vehicle_config"])
        return TopDownObservation(vehicle_config, self, self.config["rgb_clip"])


class TopDownPGDriveEnv(TopDownSingleFramePGDriveEnv):
    def initialize_observation(self):
        vehicle_config = BaseVehicle.get_vehicle_config(self.config["vehicle_config"])
        return TopDownMultiChannel(
            vehicle_config,
            self,
            self.config["rgb_clip"],
            frame_stack=self.config["frame_stack"],
            post_stack=self.config["frame_stack"],
            frame_skip=self.config["frame_skip"],
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Test single RGB frame
    # frames = []
    # env = TopDownSingleFramePGDriveEnv(dict(environment_num=1, map="C", traffic_density=1.0))
    # env.reset()
    # for _ in range(20):
    #     o, *_ = env.step([0, 1])
    #     assert env.observation_space.contains(o)
    # for _ in range(200):
    #     o, *_ = env.step([0.01, 1])
    #     frames.append(np.array(o) * 255)
    #     plt.imshow(o, cmap="gray")
    #     plt.show()
    #     print(o.mean())
    # env.close()

    # Test multi-channel frames
    env = TopDownPGDriveEnv(dict(environment_num=1, map="XTO", traffic_density=0.1))
    env.reset()
    names = [
        "road_network", "navigation", "target_vehicle", "past_pos", "traffic t", "traffic t-1", "traffic t-2",
        "traffic t-3", "traffic t-4"
    ]
    for _ in range(20):
        o, *_ = env.step([-0.05, 1])
        assert env.observation_space.contains(o)
    for _ in range(100):
        o, *_ = env.step([1, 1])
        fig, axes = plt.subplots(1, o.shape[-1], figsize=(15, 3))
        for o_i in range(o.shape[-1]):
            axes[o_i].imshow(o[..., o_i], cmap="gray")
            axes[o_i].set_title(names[o_i])
        fig.suptitle("Multi-channel Top-down Observation")
        plt.show()
        print(o.mean())
    env.close()
