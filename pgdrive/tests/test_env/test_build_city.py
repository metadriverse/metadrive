import pygame

from pgdrive import PGDriveEnv
from pgdrive.obs.top_down_renderer import draw_top_down_map
from pgdrive.component.map.city_map import CityMap
from pgdrive.utils.engine_utils import initialize_engine, close_engine


def _t(num_blocks):
    default_config = PGDriveEnv.default_config()
    initialize_engine(default_config)
    try:
        map_config = default_config["map_config"]
        map_config.update(dict(type="block_num", config=num_blocks))
        map = CityMap(map_config, random_seed=map_config["seed"])
        m = draw_top_down_map(map, return_surface=True)
        pygame.image.save(m, "{}.jpg".format(num_blocks))
    finally:
        close_engine()


def test_build_city():
    _t(num_blocks=1)
    _t(num_blocks=3)
    _t(num_blocks=20)
