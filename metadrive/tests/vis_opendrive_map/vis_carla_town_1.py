from metadrive.component.opendrive_block.opendrive_block import OpenDriveBlock
from metadrive.component.road_network.edge_road_network import OpenDriveRoadNetwork
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.asset_loader import initialize_asset_loader
from metadrive.tests.vis_block.vis_block_base import TestBlock
from metadrive.utils.opendrive.map_load import load_opendrive_map

if __name__ == "__main__":
    from metadrive.engine.engine_utils import initialize_engine, set_global_random_seed
    from metadrive.envs.base_env import BASE_DEFAULT_CONFIG

    default_config = BASE_DEFAULT_CONFIG
    default_config["use_render"] = True
    default_config["debug"] = True
    default_config["show_coordinates"] = True
    default_config["debug_static_world"] = True
    engine = initialize_engine(default_config)
    set_global_random_seed(0)

    # load map
    initialize_asset_loader(engine)
    map = load_opendrive_map(AssetLoader.file_path("carla", "CARLA_town01.xodr", return_raw_style=False))
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
    global_network.show_bounding_box(engine, (1, 0, 0, 1))
    lanes = [lane_info.lane for lane_info in global_network.graph.values()]
    engine.show_lane_coordinates(lanes)

    res_x_min, res_x_max, res_y_min, res_y_max = global_network.bounding_box
    engine.camera.setPos((res_x_min + res_x_max) / 2, -(res_y_min + res_y_max) / 2, 700)

    while True:
        engine.step()
