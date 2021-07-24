from pgdrive import PGDriveEnv
from pgdrive.engine.asset_loader import initialize_asset_loader, AssetLoader
from pgdrive.engine.core.pg_world import PGWorld


def test_asset_loader():
    default_config = PGDriveEnv.default_config()
    world_config = default_config["pg_world_config"]
    world_config.update({"use_render": False, "use_image": False, "debug": False})
    world = PGWorld(config=world_config)
    try:
        world.clear_world()
        initialize_asset_loader(world)
        print(AssetLoader.asset_path)
        print(AssetLoader.file_path("aaa"))
    # print(AssetLoader.get_loader())
    finally:
        world.close_world()


if __name__ == '__main__':
    test_asset_loader()
