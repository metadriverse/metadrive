import matplotlib.pyplot as plt

from pgdrive.scene_creator.city_map import CityMap
from pgdrive.utils import draw_top_down_map
from pgdrive.world.pg_world import PGWorld

if __name__ == '__main__':
    map_config = dict(type="block_num", config=7)

    world = PGWorld()
    map = CityMap(world, map_config)
    fig = draw_top_down_map(map, (1024, 1024))
    plt.imshow(fig, cmap="bone")
    plt.xticks([])
    plt.yticks([])
    plt.title("Building a City!")
    plt.show()
