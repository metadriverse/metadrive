from metadrive.component.opendrive_block.opendrive_block import OpenDriveBlock
from metadrive.component.road_network.edge_road_network import OpenDriveRoadNetwork
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.asset_loader import initialize_asset_loader
from metadrive.tests.vis_block.vis_block_base import TestBlock
from metadrive.utils.opendrive.map_load import load_opendrive_map
"""
AS we now add opendrive support through SUMO API, this test script is deprecated

"""


def _test_load_carla_town():
    """
    Test opendrive related feature
    Returns: None

    """
    engine = TestBlock(window_type="none")
    try:
        # load map
        initialize_asset_loader(engine)
        map = load_opendrive_map(AssetLoader.file_path("carla", "CARLA_town01.xodr", unix_style=False))
        global_network = OpenDriveRoadNetwork()
        i = 0
        blocks = []
        for road in map.roads:
            for section in road.lanes.lane_sections:
                block = OpenDriveBlock(i, global_network, 0, section)
                block.construct_block(engine.render, engine.physics_world)
                blocks.append(block)
                i += 1

        # engine.enableMouse()
        engine.show_bounding_box(global_network)
        engine.taskMgr.step()
        engine.taskMgr.step()
        engine.taskMgr.step()
    finally:
        engine.close()
