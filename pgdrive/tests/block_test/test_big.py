from pgdrive.scene_creator.algorithm.BIG import BIG
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.tests.block_test.test_block_base import TestBlock
from pgdrive.utils.asset_loader import AssetLoader


def vis_big(debug: bool = False, asset_path=None):
    if asset_path is not None:
        TestBlock.asset_path = asset_path
    test = TestBlock(debug=debug)

    test.cam.setPos(-200, -350, 2000)
    AssetLoader.init_loader(test, test.asset_path)
    global_network = RoadNetwork()

    big = BIG(2, 3.5, global_network, test.render, test.world, 888)
    test.vis_big(big)
    test.big.block_num = 40
    # big.generate(BigGenerateMethod.BLOCK_NUM, 10)
    test.run()


if __name__ == "__main__":
    vis_big()
