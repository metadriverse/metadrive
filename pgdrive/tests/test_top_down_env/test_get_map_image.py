import os

import matplotlib.pyplot as plt
from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import setup_logger


def test_save_map_image():
    os.makedirs("tmp_images", exist_ok=True)
    setup_logger(debug=True)
    env = PGDriveEnv(dict(environment_num=20, start_seed=0, map=10))
    try:
        for i in range(5):
            env.reset()
            surface = env.get_map(resolution=(128, 128))
            plt.imshow(surface, cmap="Greys")
            plt.savefig("tmp_images/map_{}.png".format(i))
        env.close()
    finally:
        env.close()


if __name__ == "__main__":
    test_save_map_image()
