from pgdrive.scene_creator.blocks.first_block import FirstPGBlock
from pgdrive.scene_creator.blocks.parking_lot import ParkingLot
from pgdrive.scene_creator.blocks.std_intersection import StdInterSection
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":
    StdInterSection.EXIT_PART_LENGTH = 4
    test = TestBlock()
    from pgdrive.engine.asset_loader import initialize_asset_loader

    initialize_asset_loader(test)

    global_network = RoadNetwork()
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
