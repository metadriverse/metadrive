"""This file visualizes a ParkingLot block. Use mouse left button to draw down for zooming out."""
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.parking_lot import ParkingLot
from metadrive.component.pgblock.std_intersection import StdInterSection
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":
    StdInterSection.EXIT_PART_LENGTH = 4
    test = TestBlock()
    from metadrive.engine.asset_loader import initialize_asset_loader

    initialize_asset_loader(test)

    global_network = NodeRoadNetwork()
    last = FirstPGBlock(global_network, 3, 1, test.render, test.world, 1)

    last = StdInterSection(1, last.get_socket(0), global_network, 1)
    last.construct_block(test.render, test.world, dict(radius=4))
    inter_1 = last

    last = ParkingLot(2, last.get_socket(1), global_network, 1)
    last.construct_block(test.render, test.world)

    last = StdInterSection(3, last.get_socket(0), global_network, 1)
    last.construct_block(test.render, test.world, dict(radius=4))
    inter_2 = last

    last = StdInterSection(4, last.get_socket(2), global_network, 1)
    last.construct_block(test.render, test.world, dict(radius=4))
    inter_3 = last

    last = ParkingLot(5, last.get_socket(2), global_network, 1)
    last.construct_block(test.render, test.world)

    inter_4 = StdInterSection(6, inter_1.get_socket(2), global_network, 1)
    inter_4.construct_block(test.render, test.world, dict(radius=4))

    # test.show_bounding_box(global_network)
    test.run()
