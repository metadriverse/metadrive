import logging

from metadrive.component.map.base_map import BaseMap
from metadrive.component.waymo_block.waymo_block import WaymoBlock
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.waymo_utils.waymo_utils import read_waymo_data
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork


class WaymoMap(BaseMap):
    def __init__(self, waymo_data):
        self.map_id = waymo_data["id"]
        self.waymo_data = waymo_data
        super(WaymoMap, self).__init__(dict(id=waymo_data["id"]))

    def _generate(self):
        block = WaymoBlock(0, self.road_network, 0, self.waymo_data["map"])
        block.construct_block(self.engine.worldNP, self.engine.physics_world)
        self.blocks.append(block)

    @staticmethod
    def waymo_position(pos):
        return pos[0], -pos[1]

    @staticmethod
    def metadrive_position(pos):
        return pos[0], -pos[1]

    @property
    def road_network_type(self):
        return EdgeRoadNetwork

    def destroy(self):
        self.waymo_data = None
        super(WaymoMap, self).destroy()

    def __del__(self):
        logging.debug("Map is Released")


if __name__ == "__main__":
    from metadrive.engine.engine_utils import initialize_engine
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.utils.waymo_utils.waymo_utils import AgentType
    from metadrive.utils.waymo_utils.waymo_utils import RoadEdgeType
    from metadrive.utils.waymo_utils.waymo_utils import RoadLineType

    # touch these items so that pickle can work
    _ = AgentType
    _ = RoadLineType
    _ = RoadEdgeType

    file_path = AssetLoader.file_path("waymo", "test.pkl", return_raw_style=False)
    data = read_waymo_data(file_path)

    default_config = MetaDriveEnv.default_config()
    default_config["use_render"] = True
    default_config["debug"] = True
    default_config["debug_static_world"] = True
    engine = initialize_engine(default_config)
    map = WaymoMap(data)
    map.attach_to_world()
    engine.enableMouse()
    map.road_network.show_bounding_box(engine)

    # argoverse data set is as the same coordinates as panda3d
    pos = map.get_center_point()
    engine.main_camera.set_bird_view_pos(pos)
    while True:
        map.engine.step()
