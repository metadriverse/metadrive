from metadrive.component.blocks.first_block import FirstPGBlock
from metadrive.component.blocks.std_intersection import StdInterSection
from metadrive.component.blocks.std_t_intersection import StdTInterSection
from metadrive.component.road.road_network import RoadNetwork
from metadrive.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":
    test = TestBlock()
    from metadrive.engine.asset_loader import initialize_asset_loader

    initialize_asset_loader(test)

    global_network = RoadNetwork()
    first = FirstPGBlock(global_network, 3.0, 1, test.render, test.world, 1)

    intersection = StdInterSection(3, first.get_socket(0), global_network, 1)
    print(intersection.construct_block(test.render, test.world))

    id = 4
    for socket_idx in range(intersection.SOCKET_NUM):
        block = StdTInterSection(id, intersection.get_socket(socket_idx), global_network, id)
        block.construct_block(test.render, test.world)
        id += 1
    test.show_bounding_box(global_network)
    test.run()
