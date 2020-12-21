import os

from pgdrive.scene_creator.map import MapGenerateMethod, Map
from pgdrive.tests.block_test.test_block_base import TestBlock
from pgdrive.utils.asset_loader import AssetLoader

if __name__ == "__main__":
    test = TestBlock()
    AssetLoader.init_loader(test, test.asset_path)
    map = Map(test, big_config={Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_NUM, Map.GENERATE_PARA: 12})
    map.save_map("map_test", os.path.dirname(__file__))
