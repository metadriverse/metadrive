from pgdrive.envs.observation_type import TopDownObservation
from pgdrive.envs.pgdrive_env import PGDriveEnv

from pgdrive.pg_config import PGConfig
from pgdrive.scene_creator.ego_vehicle.base_vehicle import BaseVehicle


class TopDownPGDriveEnv(PGDriveEnv):
    @staticmethod
    def default_config() -> PGConfig:
        config = PGDriveEnv.default_config()
        config["use_topdown"] = True
        return config

    def initialize_observation(self):
        vehicle_config = BaseVehicle.get_vehicle_config(self.config["vehicle_config"])
        return TopDownObservation(vehicle_config, self, self.config["rgb_clip"])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    frames = []
    env = TopDownPGDriveEnv(dict(environment_num=1, map="C", traffic_density=1.0))
    import numpy as np
    env.reset()
    for _ in range(10):
        o, *_ = env.step([0, 1])
        # env.reset()
    for _ in range(10):
        o, *_ = env.step([-0.05, 1])
    for _ in range(200):
        o, *_ = env.step([0.01, 1])

        frames.append(np.array(o) * 255)
        plt.imshow(o)
        plt.show()
        print(o.mean())
    env.close()

    # TODO(PZH) remove this when merging the PR!
    # import visya
    # visya.generate_mp4(np.array(frames, dtype=np.uint8), "tmp.mp4")
