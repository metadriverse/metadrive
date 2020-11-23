from pg_drive.scene_creator.blocks.curve import Curve
from pg_drive.scene_creator.blocks.first_block import FirstBlock
from pg_drive.scene_creator.road.road_network import RoadNetwork
from pg_drive.tests.block_test.test_block_base import TestBlock

if __name__ == "__main__":
    test = TestBlock()
    from pg_drive.utils.visualization_loader import VisLoader
    VisLoader.init_loader(test.loader, test.asset_path)

    global_network = RoadNetwork()
    curve = FirstBlock(global_network, 3.0, 1, test.render, test.world, 1)
    for i in range(1, 13):
        curve = Curve(i, curve.get_socket(0), global_network, i)
        # print(i)
        while True:
            success = curve.construct_block_in_world(test.render, test.world)
            # print(success)
            if success:
                break
            curve.destruct_block_in_world(test.world)
    # test.run()
