import logging

from metadrive.component.map.base_map import BaseMap
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.component.waymo_block.waymo_block import WaymoBlock
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.waymo_utils.waymo_utils import read_waymo_data


class WaymoMap(BaseMap):
    def __init__(self, map_index, random_seed=None):
        self.map_index = map_index
        super(WaymoMap, self).__init__(dict(id=self.map_index), random_seed=random_seed)

    def _generate(self):
        block = WaymoBlock(0, self.road_network, 0, self.map_index)
        block.construct_block(self.engine.worldNP, self.engine.physics_world, attach_to_world=True)
        self.blocks.append(block)

    def play(self):
        # For debug
        for b in self.blocks:
            b._create_in_world(skip=True)
            b.attach_to_world(self.engine.worldNP, self.engine.physics_world)
            b.detach_from_world(self.engine.physics_world)

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
        self.map_index = None
        super(WaymoMap, self).destroy()

    def __del__(self):
        # self.destroy()
        logging.debug("Map is Released")
        print("[WaymoMap] Map is Released")


if __name__ == "__main__":
    from metadrive.engine.engine_utils import initialize_engine
    from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
    from metadrive.utils.waymo_utils.waymo_utils import AgentType
    from metadrive.utils.waymo_utils.waymo_utils import RoadEdgeType
    from metadrive.utils.waymo_utils.waymo_utils import RoadLineType
    from metadrive.manager.waymo_data_manager import WaymoDataManager

    # touch these items so that pickle can work
    _ = AgentType
    _ = RoadLineType
    _ = RoadEdgeType

    file_path = AssetLoader.file_path("waymo", "0.pkl", return_raw_style=False)
    data = read_waymo_data(file_path)

    default_config = WaymoEnv.default_config()
    default_config["use_render"] = True
    default_config["debug"] = True
    default_config["debug_static_world"] = True
    default_config["waymo_data_directory"] = AssetLoader.file_path("waymo", return_raw_style=False)
    default_config["case_num"] = 1
    engine = initialize_engine(default_config)

    engine.data_manager = WaymoDataManager()
    map = WaymoMap(map_index=0)
    map.attach_to_world()
    engine.enableMouse()
    map.road_network.show_bounding_box(engine)

    # argoverse data set is as the same coordinates as panda3d
    pos = map.get_center_point()
    engine.main_camera.set_bird_view_pos(pos)
    while True:
        map.engine.step()
