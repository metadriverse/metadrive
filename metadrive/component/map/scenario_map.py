import logging

import numpy as np

from metadrive.component.map.base_map import BaseMap
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.component.scenario_block.scenario_block import ScenarioBlock
from metadrive.engine.asset_loader import AssetLoader
from metadrive.type import MetaDriveType
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.math import resample_polyline, get_polyline_length


class ScenarioMap(BaseMap):
    def __init__(self, map_index, map_data, random_seed=None, need_lane_localization=False):
        self.map_index = map_index
        self.map_data = map_data
        self.need_lane_localization = need_lane_localization or self.engine.global_config.get(
            "need_lane_localization", False
        )
        super(ScenarioMap, self).__init__(dict(id=self.map_index), random_seed=random_seed)

    def show_coordinates(self):
        lanes = [lane_info.lane for lane_info in self.road_network.graph.values()]
        self._show_coordinates(lanes)

    def _generate(self):
        block = ScenarioBlock(
            block_index=0,
            global_network=self.road_network,
            random_seed=0,
            map_index=self.map_index,
            map_data=self.map_data,
            need_lane_localization=self.need_lane_localization
        )
        self.crosswalks = block.crosswalks
        self.sidewalks = block.sidewalks
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
        super(ScenarioMap, self).destroy()

    def __del__(self):
        # self.destroy()
        logging.debug("Map is Released")
        # print("[ScenarioMap] Map is Released")

    def get_boundary_line_vector(self, interval):
        """
        Get the polylines of the map, represented by a set of points
        """
        ret = {}
        for lane_id, data in self.blocks[-1].map_data.items():
            type = data.get("type", None)
            map_feat_id = str(lane_id)
            if MetaDriveType.is_road_line(type):
                if len(data[ScenarioDescription.POLYLINE]) <= 1:
                    continue
                line = np.asarray(data[ScenarioDescription.POLYLINE])[..., :2]
                length = get_polyline_length(line)
                resampled = resample_polyline(line, interval) if length > interval * 2 else line
                if MetaDriveType.is_broken_line(type):
                    ret[map_feat_id] = {
                        "type": MetaDriveType.LINE_BROKEN_SINGLE_YELLOW
                        if MetaDriveType.is_yellow_line(type) else MetaDriveType.LINE_BROKEN_SINGLE_WHITE,
                        "polyline": resampled
                    }
                else:
                    ret[map_feat_id] = {
                        "polyline": resampled,
                        "type": MetaDriveType.LINE_SOLID_SINGLE_YELLOW
                        if MetaDriveType.is_yellow_line(type) else MetaDriveType.LINE_SOLID_SINGLE_WHITE
                    }
            elif MetaDriveType.is_road_boundary_line(type):
                line = np.asarray(data[ScenarioDescription.POLYLINE])[..., :2]
                length = get_polyline_length(line)
                resampled = resample_polyline(line, interval) if length > interval * 2 else line
                ret[map_feat_id] = {"polyline": resampled, "type": MetaDriveType.BOUNDARY_LINE}
            elif MetaDriveType.is_lane(type):
                continue
            # else:
            # # for debug
            #     raise ValueError
        return ret

    # def get_map_features(self, interval=2):
    #     """
    #
    #     """
    #     map_features = super(ScenarioMap, self).get_map_features(interval=interval)
    #
    #     # Adding the information stored in original data to here
    #     original_map_features = self.engine.data_manager.get_scenario(self.map_index)["map_features"]
    #
    #     for map_feat_key, old_map_feat in original_map_features.items():
    #
    #         if map_feat_key not in map_features:
    #             # Discard the data for those map features that are not reconstructed by MetaDrive.
    #             pass
    #
    #             # old_map_feat = copy.deepcopy(old_map_feat)
    #             # old_map_feat["valid"] = False
    #             #
    #             # map_features[map_feat_key] = old_map_feat
    #             #
    #             # if "polyline" in old_map_feat:
    #             #     map_features[map_feat_key]["polyline"] = convert_polyline_to_metadrive(
    #             #         old_map_feat["polyline"], coordinate_transform=self.coordinate_transform
    #             #     )
    #             #
    #             # if "position" in old_map_feat:
    #             #     if self.coordinate_transform:
    #             #         map_features[map_feat_key]["position"] = waymo_to_metadrive_vector(old_map_feat["position"])
    #             #     else:
    #             #         map_features[map_feat_key]["position"] = old_map_feat["position"]
    #
    #         else:
    #             # This map features are in both original data and current data.
    #             # We will check if in original data this map features contains some useful metadata.
    #             # If so, copied it to the new data.
    #             if "speed_limit_kmh" in old_map_feat:
    #                 map_features[map_feat_key]["speed_limit_kmh"] = old_map_feat["speed_limit_kmh"]
    #             if "speed_limit_mph" in old_map_feat:
    #                 map_features[map_feat_key]["speed_limit_mph"] = old_map_feat["speed_limit_mph"]
    #
    #     return map_features


if __name__ == "__main__":
    from metadrive.engine.engine_utils import initialize_engine
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.manager.scenario_data_manager import ScenarioDataManager

    # # touch these items so that pickle can work

    # file_path = AssetLoader.file_path("waymo", "0.pkl", unix_style=False)
    # # file_path = "/home/shady/Downloads/test_processed/60.pkl"
    # data = read_scenario_data(file_path)

    default_config = ScenarioEnv.default_config()
    default_config["_render_mode"] = "onscreen"
    default_config["use_render"] = True
    default_config["debug"] = True
    default_config["debug_static_world"] = True
    default_config["data_directory"] = AssetLoader.file_path("waymo", unix_style=False)
    # default_config["data_directory"] = AssetLoader.file_path("nuscenes", unix_style=False)
    # default_config["data_directory"] = "/home/shady/Downloads/test_processed"
    default_config["num_scenarios"] = 1
    engine = initialize_engine(default_config)

    engine.data_manager = ScenarioDataManager()
    m_data = engine.data_manager.get_scenario(0, should_copy=False)["map_features"]
    map = ScenarioMap(map_index=0, map_data=m_data)
    map.attach_to_world()
    engine.enableMouse()
    map.road_network.show_bounding_box(engine)

    pos = map.get_center_point()
    engine.main_camera.set_bird_view_pos_hpr(pos)
    while True:
        map.engine.step()
