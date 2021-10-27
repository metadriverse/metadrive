try:
    import logging
    import os
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from typing import Any, Dict, List, Mapping, Tuple, Union, cast
    import argoverse
    from argoverse.data_loading.vector_map_loader import Node, append_additional_key_value_pair, \
        append_unique_key_value_pair, convert_node_id_list_to_xy, extract_node_waypt, get_lane_identifier, str_to_bool, \
        extract_node_from_ET_element
    from argoverse.map_representation.map_api import ArgoverseMap as AGMap
    from argoverse.map_representation.map_api import PITTSBURGH_ID, MIAMI_ID, ROOT
except ImportError:
    pass

from metadrive.component.argoverse_block.argoverse_block import ArgoverseBlock
from metadrive.component.lane.argoverse_lane import ArgoverseLane
from metadrive.component.map.base_map import BaseMap
from metadrive.constants import LineColor

logger = logging.getLogger(__name__)

_PathLike = Union[str, "os.PathLike[str]"]


class ArgoverseMap(BaseMap):
    """
    This class converting the Argoverse to MetaDrive and allow the interactive behaviors
    """

    # supported city mas
    SUPPORTED_MAPS = ["MIA", "PIT"]

    # block size
    BLOCK_LANE_NUM = 40

    def __init__(self, map_config, random_seed=0):
        # origin ag map
        self.AGMap = AGMap()
        map_config[self.SEED] = random_seed
        assert "city" in map_config, "City name is required when generating argoverse map"
        assert map_config["city"] in self.SUPPORTED_MAPS, "City generation of {} is not supported (We support {} now)". \
            format(map_config["city"], self.SUPPORTED_MAPS)
        self.city = map_config["city"]
        super(ArgoverseMap, self).__init__(map_config=map_config, random_seed=None)
        self.lane_id_lane = None

    def _generate(self):
        """
        Modified from argoverse-map api
        """
        city_name = self.config["city"]
        city_id = MIAMI_ID if city_name == "MIA" else PITTSBURGH_ID
        xml_fpath = Path(ROOT).resolve() / f"pruned_argoverse_{city_name}_{city_id}_vector_map.xml"
        tree = ET.parse(os.fspath(xml_fpath))
        root = tree.getroot()
        logger.info(f"Loaded root: {root.tag}")

        all_graph_nodes = {}
        lane_objs = {}
        # all children are either Nodes or Ways
        for child in root:
            if child.tag == "node":
                node_obj = extract_node_from_ET_element(child)
                all_graph_nodes[node_obj.id] = node_obj
            elif child.tag == "way":
                lane_obj, lane_id = self.extract_lane_segment_from_ET_element(child, all_graph_nodes)
                lane_objs[lane_id] = lane_obj
            else:
                logger.error("Unknown XML item encountered.")
                raise ValueError("Unknown XML item encountered.")
        lane_ids = self.AGMap.get_lane_ids_in_xy_bbox(
            *self.argoverse_position(self.config["center"]), self.config["city"], self.config["radius"]
        )
        self.lane_id_lane = lane_objs
        self._construct_road_network([lane_objs[k] for k in lane_ids])

    def _construct_road_network(self, lanes: list):
        # TODO split the blocks in the future, if we need the whole map
        chosen_lanes = lanes
        for lane in lanes:
            self._post_process_lane(lane)

        block = ArgoverseBlock(0, self.road_network, {lane.id: lane for lane in chosen_lanes})
        block.construct_block(self.engine.worldNP, self.engine.physics_world)
        self.blocks.append(block)

    def _post_process_lane(self, lane: ArgoverseLane):
        if lane.l_neighbor_id is not None and self.lane_id_lane[lane.l_neighbor_id].l_neighbor_id == lane.id:
            lane.line_color = (LineColor.YELLOW, LineColor.GREY)

    @staticmethod
    def extract_lane_segment_from_ET_element(child: ET.Element,
                                             all_graph_nodes: Mapping[int, Node]) -> Tuple[ArgoverseLane, int]:
        """
        Modified from Argoverse map-api
        """
        lane_dictionary: Dict[str, Any] = {}
        lane_id = get_lane_identifier(child)
        node_id_list: List[int] = []
        for element in child:
            # The cast on the next line is the result of a typeshed bug.  This really is a List and not a ItemsView.
            way_field = cast(List[Tuple[str, str]], list(element.items()))
            field_name = way_field[0][0]
            if field_name == "k":
                key = way_field[0][1]
                if key in {"predecessor", "successor"}:
                    append_additional_key_value_pair(lane_dictionary, way_field)
                else:
                    append_unique_key_value_pair(lane_dictionary, way_field)
            else:
                node_id_list.append(extract_node_waypt(way_field))

        lane_dictionary["centerline"] = convert_node_id_list_to_xy(node_id_list, all_graph_nodes)
        predecessors = lane_dictionary.get("predecessor", None)
        successors = lane_dictionary.get("successor", None)
        has_traffic_control = str_to_bool(lane_dictionary["has_traffic_control"])
        is_intersection = str_to_bool(lane_dictionary["is_intersection"])
        lnid = lane_dictionary["l_neighbor_id"]
        rnid = lane_dictionary["r_neighbor_id"]
        l_neighbor_id = None if lnid == "None" else int(lnid)
        r_neighbor_id = None if rnid == "None" else int(rnid)
        lane_segment = ArgoverseLane(
            str(node_id_list[0]),
            str(node_id_list[-1]),
            lane_id,
            has_traffic_control,
            lane_dictionary["turn_direction"],
            is_intersection,
            l_neighbor_id,
            r_neighbor_id,
            predecessors,
            successors,
            lane_dictionary["centerline"],
        )
        return lane_segment, lane_id

    @staticmethod
    def argoverse_position(pos):
        pos[1] *= -1
        return pos

    @staticmethod
    def metadrive_position(pos):
        pos[1] *= -1
        return pos


if __name__ == "__main__":
    from metadrive.engine.engine_utils import initialize_engine
    from metadrive.envs.metadrive_env import MetaDriveEnv

    default_config = MetaDriveEnv.default_config()
    default_config["use_render"] = True
    default_config["debug"] = True
    engine = initialize_engine(default_config)

    # in argoverse coordinates
    xcenter, ycenter = 2599.5505965123866, 1200.0214763629717
    xmin = xcenter - 80  # 150
    xmax = xcenter + 80  # 150
    ymin = ycenter - 80  # 150
    ymax = ycenter + 80  # 150
    map = ArgoverseMap(
        {
            "city": "PIT",
            # "draw_map_resolution": 1024,
            "center": ArgoverseMap.metadrive_position([xcenter, ycenter]),
            "radius": 100
        }
    )
    engine.map_manager.load_map(map)
    engine.enableMouse()

    # argoverse data set is as the same coordinates as panda3d
    engine.main_camera.set_bird_view_pos(ArgoverseMap.metadrive_position([xcenter, ycenter]))
    while True:
        map.engine.step()
