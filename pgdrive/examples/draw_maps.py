import random

import matplotlib.pyplot as plt

from pgdrive import PGDriveEnvV2
from pgdrive.utils import draw_top_down_map

if __name__ == '__main__':
    env = PGDriveEnvV2(config=dict(environment_num=100, map=7, start_seed=random.randint(0, 1000)))
    fig, axs = plt.subplots(4, 4, figsize=(10, 10), dpi=100)
    count = 0
    for i in range(4):
        for j in range(4):
            count += 1
            env.reset()
            m = draw_top_down_map(env.current_map)
            # m = env.get_map()
            ax = axs[i][j]
            ax.imshow(m, cmap="bone")
            ax.set_xticks([])
            ax.set_yticks([])
            print("Drawing {}-th map!".format(count))
    fig.suptitle("Bird's-eye view of genertaed maps")
    plt.show()
    env.close()
