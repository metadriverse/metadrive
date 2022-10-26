from metadrive.component.pgblock.curve import Curve
from metadrive.component.opendrive_block.opendrive_block import OpenDriveBlock
from metadrive.component.road_network.edge_road_network import OpenDriveRoadNetwork
from metadrive.engine.asset_loader import initialize_asset_loader
from metadrive.tests.vis_block.vis_block_base import TestBlock
from metadrive.utils.opendrive_map_utils.map_load import load_opendrive_map

if __name__ == "__main__":
    test = TestBlock()

    initialize_asset_loader(test)
    map = load_opendrive_map("C:\\Users\\x1\\Desktop\\CARLA_town01.xodr")
    global_network = OpenDriveRoadNetwork()
    i = 0
    for road in map.roads:
        for section in road.lanes.lane_sections:
            block = OpenDriveBlock(i, global_network, 0, section)
            block.construct_block(test.render, test.world)
            i += 1

    test.show_bounding_box(global_network)
    test.run()
