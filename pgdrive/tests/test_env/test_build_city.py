from pgdrive import PGDriveEnv
from pgdrive.scene_creator.city_map import CityMap
from pgdrive.world.pg_world import PGWorld


def _t(num_blocks):
    default_config = PGDriveEnv.default_config()

    world_config = default_config["pg_world_config"]
    world_config.update({"use_render": False, "use_image": False, "debug": False})
    world = PGWorld(config=world_config)
    try:
        map_config = default_config["map_config"]
        map_config.update(dict(type="block_num", config=num_blocks))
        map = CityMap(world, map_config)
    finally:
        world.close_world()


def test_build_city():
    _t(num_blocks=1)
    _t(num_blocks=3)
    _t(num_blocks=20)
