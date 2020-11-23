from pg_drive.scene_creator.blocks.first_block import FirstBlock
from pg_drive.scene_creator.blocks.straight import Straight
from pg_drive.scene_creator.road.road_network import RoadNetwork
from pg_drive.tests.block_test.test_block_base import TestBlock

if __name__ == "__main__":
    test = TestBlock()
    from pg_drive.utils.visualization_loader import VisLoader
    VisLoader.init_loader(test.loader, test.asset_path)

    global_network = RoadNetwork()
    straight = FirstBlock(global_network, 3.0, 1, test.render, test.world, 1)
    for i in range(1, 3):
        straight = Straight(i, straight.get_socket(0), global_network, i)
        straight.construct_block_in_world(test.render, test.world)
        print(len(straight.bullet_nodes))
    # test.run()
