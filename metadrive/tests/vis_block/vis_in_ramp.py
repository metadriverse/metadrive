from metadrive.component.blocks.first_block import FirstPGBlock
from metadrive.component.blocks.ramp import InRampOnStraight
from metadrive.component.blocks.straight import Straight
from metadrive.component.road.road_network import RoadNetwork
from metadrive.engine.asset_loader import initialize_asset_loader
from metadrive.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":

    test = TestBlock()

    initialize_asset_loader(test)

    global_network = RoadNetwork()
    straight = FirstPGBlock(global_network, 3.0, 1, test.render, test.world, 1)
    straight = Straight(4, straight.get_socket(0), global_network, 1)
    print(straight.construct_block(test.render, test.world))
    print(len(straight.dynamic_nodes))
    for i in range(1, 3):
        straight = InRampOnStraight(i, straight.get_socket(0), global_network, i)
        print(straight.construct_block(test.render, test.world))
        print(len(straight.dynamic_nodes))
    test.show_bounding_box(global_network)
    test.run()
