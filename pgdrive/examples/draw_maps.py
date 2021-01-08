import random

import matplotlib.pyplot as plt

from pgdrive import PGDriveEnv

if __name__ == '__main__':
    env = PGDriveEnv(config=dict(environment_num=100, map=7, start_seed=random.randint(0, 1000)))
    fig, axs = plt.subplots(4, 4, figsize=(10, 10), dpi=100)
    for i in range(4):
        for j in range(4):
            env.reset()
            m = env.get_map()
            ax = axs[i][j]
            ax.imshow(m, cmap="bone")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Bird's-eye view of genertaed maps")
    plt.show()
    env.close()
