import logging

from metadrive.component.algorithm.BIG import BIG
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test = TestBlock(True)
    global_network = NodeRoadNetwork()

    big = BIG(2, 5, global_network, test.render, test.world, 1010)

    # Since some change to generate function, specify the block num to the big
    big.block_num = len("CrTRXOS")
    big._block_sequence = "CrTRXOS"
    test.vis_big(big)

    # big.generate(BigGenerateMethod.BLOCK_NUM, 10)
    test.run()
