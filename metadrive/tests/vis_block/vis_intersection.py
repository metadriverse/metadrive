"""This file visualizes an InterSection block. Use mouse left button to draw down for zooming out."""
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":
    test = TestBlock()
    from metadrive.engine.asset_loader import initialize_asset_loader
    initialize_asset_loader(engine=test)
    global_network = NodeRoadNetwork()
    first = FirstPGBlock(global_network, 3.0, 2, test.render, test.world, 20)
    intersection = InterSection(3, first.get_socket(0), global_network, 1)
    # print(intersection.construct_block(test.render, test.world))

    id = 4
    for socket_idx in range(intersection.SOCKET_NUM):
        block = Curve(id, intersection.get_socket(socket_idx), global_network, id)
        block.construct_block(test.render, test.world)
        id += 1
    intersection = InterSection(id, block.get_socket(0), global_network, 1)
    intersection.construct_block(test.render, test.world)
    test.show_bounding_box(global_network)
    test.run()
