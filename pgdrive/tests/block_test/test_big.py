from pgdrive.scene_creator.algorithm.BIG import BIG
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.tests.block_test.test_block_base import TestBlock
from pgdrive.utils.asset_loader import AssetLoader

if __name__ == "__main__":
    test = TestBlock()
    AssetLoader.init_loader(test, test.asset_path)
    global_network = RoadNetwork()

    big = BIG(2, 5, global_network, test.render, test.world, 888)
    test.vis_big(big)
    test.big.block_num = 40
    # big.generate(BigGenerateMethod.BLOCK_NUM, 10)
    test.run()
