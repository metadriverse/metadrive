from metadrive.component.blocks.curve import Curve
from metadrive.component.blocks.first_block import FirstPGBlock
from metadrive.component.blocks.straight import Straight
from metadrive.component.blocks.t_intersection import TInterSection
from metadrive.component.road.road_network import RoadNetwork
from metadrive.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":
    test = TestBlock(True)
    from metadrive.engine.asset_loader import initialize_asset_loader

    initialize_asset_loader(test)

    global_network = RoadNetwork()
    first = FirstPGBlock(global_network, 3.0, 2, test.render, test.world, 1)

    curve = Curve(1, first.get_socket(0), global_network, 1)
    curve.construct_block(test.render, test.world)

    straight = Straight(2, curve.get_socket(0), global_network, 1)
    straight.construct_block(test.render, test.world)

    intersection = TInterSection(3, straight.get_socket(0), global_network, 20)
    print(intersection.construct_block(test.render, test.world))
    id = 4
    for socket_idx in range(intersection.SOCKET_NUM):
        block = Curve(id, intersection.get_socket(socket_idx), global_network, id + 1)
        block.construct_block(test.render, test.world)
        id += 1
    test.show_bounding_box(global_network)
    test.run()
