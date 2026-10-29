"""This file visualizes a Bottleneck block."""
from metadrive.component.pgblock.bottleneck import Merge, Split
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.engine.asset_loader import initialize_asset_loader
from metadrive.tests.vis_block.vis_block_base import TestBlock


def test_map_visualizer():
    test = TestBlock(window_type="onscreen")
    try:
        initialize_asset_loader(test)

        global_network = NodeRoadNetwork()
        b = FirstPGBlock(global_network, 3.0, 2, test.render, test.world, 1)
        for i in range(1, 13):
            tp = Merge if i % 3 == 0 else Split
            b = tp(i, b.get_socket(0), global_network, i)
            b.construct_block(test.render, test.world)
        test.show_bounding_box(global_network)
        test.taskMgr.step()
        test.taskMgr.step()
        test.taskMgr.step()
        test.taskMgr.step()
        test.taskMgr.step()
    finally:
        test.close()
