from pg_drive.tests.block_test.test_block_base import TestBlock
from pg_drive.scene_creator.map import MapGenerateMethod, Map

from pg_drive.utils.asset_loader import AssetLoader
from pg_drive.scene_creator.map import Map
import os

if __name__ == "__main__":
    test = TestBlock()
    AssetLoader.init_loader(test.loader, test.asset_path)
    map = Map(
        test.render,
        test.physics_world,
        big_config={
            Map.GENERATE_METHOD: MapGenerateMethod.PG_MAP_FILE,
            Map.GENERATE_PARA: AssetLoader.file_path(os.path.dirname(__file__), "map_test.json")
        }
    )
    test.run()
