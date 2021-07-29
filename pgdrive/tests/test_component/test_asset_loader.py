from pgdrive import PGDriveEnv
from pgdrive.engine.asset_loader import initialize_asset_loader, AssetLoader
from pgdrive.engine.core.engine_core import EngineCore


def test_asset_loader():
    default_config = PGDriveEnv.default_config()
    world = EngineCore(global_config=default_config)
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
