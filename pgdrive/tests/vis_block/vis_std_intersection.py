from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.blocks.std_intersection import StdInterSection
from pgdrive.scene_creator.blocks.std_t_intersection import StdTInterSection
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":
    test = TestBlock()
    from pgdrive.utils.asset_loader import initialize_asset_loader

    initialize_asset_loader(test)

    global_network = RoadNetwork()
    first = FirstBlock(global_network, 3.0, 1, test.render, test.world, 1)

    intersection = StdInterSection(3, first.get_socket(0), global_network, 1)
    print(intersection.construct_block(test.render, test.world))

    id = 4
    for socket_idx in range(intersection.SOCKET_NUM):
        block = StdTInterSection(id, intersection.get_socket(socket_idx), global_network, id)
        block.construct_block(test.render, test.world)
        id += 1
    test.show_bounding_box(global_network)
    test.run()
