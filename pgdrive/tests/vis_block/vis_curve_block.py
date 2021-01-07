from pgdrive.scene_creator.blocks.curve import Curve
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.tests.vis_block.vis_block_base import TestBlock
from pgdrive.utils.asset_loader import initialize_asset_loader

if __name__ == "__main__":
    test = TestBlock()

    initialize_asset_loader(test)

    global_network = RoadNetwork()
    curve = FirstBlock(global_network, 3.0, 1, test.render, test.world, 1)
    for i in range(1, 13):
        curve = Curve(i, curve.get_socket(0), global_network, i)
        print(i)
        while True:
            success = curve.construct_block(test.render, test.world)
            print(success)
            if success:
                break
            curve.destruct_block(test.world)
    test.show_bounding_box(global_network)
    test.run()
