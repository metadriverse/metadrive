from pgdrive.scene_creator.city_map import CityMap
from pgdrive.world.pg_world import PGWorld


def _t(num_blocks):
    map_config = dict(type="block_num", config=num_blocks)
    world = PGWorld()
    try:
        map = CityMap(world, map_config)
    finally:
        world.close_world()


def test_build_city():
    _t(num_blocks=1)
    _t(num_blocks=3)
    _t(num_blocks=20)
