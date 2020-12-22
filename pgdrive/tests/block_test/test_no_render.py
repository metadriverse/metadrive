import logging

from pgdrive.scene_creator.algorithm.BIG import BIG
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.tests.block_test.test_block_base import TestBlock

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test = TestBlock(True)
    global_network = RoadNetwork()

    big = BIG(2, 5, global_network, test.render, test.world, 1010)

    # Since some change to generate function, specify the block num to the big
    big.block_num = len("CrTRXOS")
    big._block_sequence = "CrTRXOS"
    test.vis_big(big)

    # big.generate(BigGenerateMethod.BLOCK_NUM, 10)
    test.run()
