import matplotlib.pyplot as plt

from metadrive import MetaDriveEnv
from metadrive.component.map.city_map import CityMap
from metadrive.engine.engine_utils import initialize_engine, close_engine, set_global_random_seed
from metadrive.utils.draw_top_down_map import draw_top_down_map


def _t(num_blocks):
    default_config = MetaDriveEnv.default_config()
    initialize_engine(default_config)
    set_global_random_seed(0)
    try:
        map_config = default_config["map_config"]
        map_config.update(dict(type="block_num", config=num_blocks))
        map = CityMap(map_config)
        fig = draw_top_down_map(map, (1024, 1024))
        plt.imshow(fig, cmap="bone")
        plt.xticks([])
        plt.yticks([])
        plt.title("Building a City with {} blocks!".format(num_blocks))
        plt.show()
    finally:
        close_engine()


if __name__ == "__main__":
    _t(20)
