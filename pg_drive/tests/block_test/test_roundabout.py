from pg_drive.scene_creator.blocks.first_block import FirstBlock
from pg_drive.scene_creator.blocks.roundabout import Roundabout
from pg_drive.scene_creator.road.road_network import RoadNetwork
from pg_drive.tests.block_test.test_block_base import TestBlock
from pg_drive.scene_creator.blocks.curve import Curve

if __name__ == "__main__":
    test = TestBlock(False)
    from pg_drive.utils.visualization_loader import VisLoader
    VisLoader.init_loader(test.loader, test.asset_path)

    global_network = RoadNetwork()
    straight = FirstBlock(global_network, 3.0, 1, test.render, test.world, 1)

    rd = Roundabout(1, straight.get_socket(0), global_network, 1)
    print(rd.construct_block_in_world(test.render, test.world))

    id = 4
    for socket_idx in range(rd.SOCKET_NUM):
        block = Curve(id, rd.get_socket(socket_idx), global_network, id + 1)
        block.construct_block_in_world(test.render, test.world)
        id += 1
    # test.run()
