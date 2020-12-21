from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.blocks.ramp import InRampOnStraight
from pgdrive.scene_creator.blocks.straight import Straight
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.tests.block_test.test_block_base import TestBlock

if __name__ == "__main__":

    test = TestBlock()
    from pgdrive.utils.asset_loader import AssetLoader

    AssetLoader.init_loader(test, test.asset_path)

    global_network = RoadNetwork()
    straight = FirstBlock(global_network, 3.0, 1, test.render, test.world, 1)
    straight = Straight(4, straight.get_socket(0), global_network, 1)
    print(straight.construct_block(test.render, test.world))
    print(len(straight.bullet_nodes))
    for i in range(1, 3):
        straight = InRampOnStraight(i, straight.get_socket(0), global_network, i)
        print(straight.construct_block(test.render, test.world))
        print(len(straight.bullet_nodes))
    # test.run()
