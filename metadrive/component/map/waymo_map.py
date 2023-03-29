import logging

from metadrive.component.map.base_map import BaseMap
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.component.waymo_block.waymo_block import WaymoBlock
from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario.metadrive_type import MetaDriveType
from metadrive.utils.waymo_utils.utils import convert_polyline_to_metadrive
from metadrive.utils.waymo_utils.utils import read_waymo_data
from metadrive.utils.waymo_utils.waymo_type import WaymoRoadLineType, WaymoRoadEdgeType
from metadrive.utils.waymo_utils.waymo_type import WaymoLaneProperty


class WaymoMap(BaseMap):
    def __init__(self, map_index, random_seed=None, need_lane_localization=True):
        self.map_index = map_index
        self.need_lane_localization = need_lane_localization
        super(WaymoMap, self).__init__(dict(id=self.map_index), random_seed=random_seed)

    def _generate(self):
        block = WaymoBlock(
            block_index=0, global_network=self.road_network, random_seed=0, map_index=self.map_index,
            need_lane_localization=self.need_lane_localization
        )
        block.construct_block(self.engine.worldNP, self.engine.physics_world, attach_to_world=True)
        self.blocks.append(block)

    def play(self):
        # For debug
        for b in self.blocks:
            b._create_in_world(skip=True)
            b.attach_to_world(self.engine.worldNP, self.engine.physics_world)
            b.detach_from_world(self.engine.physics_world)

    @property
    def road_network_type(self):
        return EdgeRoadNetwork

    def destroy(self):
        self.map_index = None
        super(WaymoMap, self).destroy()

    def __del__(self):
        # self.destroy()
        logging.debug("Map is Released")
        # print("[WaymoMap] Map is Released")

    def get_boundary_line_vector(self, interval):
        ret = {}
        for lane_id, data in self.blocks[-1].waymo_map_data.items():
            type = data.get("type", None)
            map_feat_id = str(lane_id)
            if WaymoRoadLineType.is_road_line(type):
                if len(data[WaymoLaneProperty.POLYLINE]) <= 1:
                    continue
                if WaymoRoadLineType.is_broken(type):
                    ret[map_feat_id] = {
                        "type": MetaDriveType.BROKEN_YELLOW_LINE
                        if WaymoRoadLineType.is_yellow(type) else MetaDriveType.BROKEN_GREY_LINE,
                        "polyline": convert_polyline_to_metadrive(data[WaymoLaneProperty.POLYLINE], coordinate_transform=self.coordinate_transform)
                    }
                else:
                    ret[map_feat_id] = {
                        "polyline": convert_polyline_to_metadrive(data[WaymoLaneProperty.POLYLINE], coordinate_transform=self.coordinate_transform),
                        "type": MetaDriveType.CONTINUOUS_YELLOW_LINE
                        if WaymoRoadLineType.is_yellow(type) else MetaDriveType.CONTINUOUS_GREY_LINE
                    }
            elif WaymoRoadEdgeType.is_road_edge(type):
                ret[map_feat_id] = {
                    "polyline": convert_polyline_to_metadrive(data[WaymoLaneProperty.POLYLINE], coordinate_transform=self.coordinate_transform),
                    "type": MetaDriveType.CONTINUOUS_GREY_LINE
                }
        return ret

    @property
    def coordinate_transform(self):
        return self.engine.global_config["coordinate_transform"]


if __name__ == "__main__":
    from metadrive.engine.engine_utils import initialize_engine
    from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
    from metadrive.manager.waymo_data_manager import WaymoDataManager

    # # touch these items so that pickle can work

    file_path = AssetLoader.file_path("waymo", "0.pkl", return_raw_style=False)
    # file_path = "/home/shady/Downloads/test_processed/60.pkl"
    data = read_waymo_data(file_path)

    default_config = WaymoEnv.default_config()
    default_config["use_render"] = True
    default_config["debug"] = True
    default_config["debug_static_world"] = True
    default_config["waymo_data_directory"] = AssetLoader.file_path("waymo", return_raw_style=False)
    # default_config["waymo_data_directory"] = "/home/shady/Downloads/test_processed"
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
