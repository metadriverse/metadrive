import logging

import numpy as np
import tqdm

from metadrive.type import MetaDriveType

try:
    import geopandas as gpd
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
    from shapely.ops import unary_union
except ImportError:
    pass

from metadrive.component.lane.nuplan_lane import NuPlanLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.nuplan_block.nuplan_block import LaneLineProperty
from metadrive.component.nuplan_block.nuplan_block import NuPlanBlock
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.constants import PGLineColor, PGLineType
from metadrive.engine.engine_utils import get_global_config
from metadrive.engine.scene_cull import SceneCull
from metadrive.utils.coordinates_shift import nuplan_to_metadrive_vector, metadrive_to_nuplan_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NuPlanMap(BaseMap):
    def __init__(self, map_name, nuplan_center, radius, random_seed=None, need_lane_localization=True):
        self.need_lane_localization = need_lane_localization
        self.map_name = map_name
        self._center = np.array(nuplan_center)
        self._nuplan_map_api = self.engine.data_manager.current_scenario.map_api
        self.attached_blocks = []
        self.boundary_block = None  # it won't be detached
        self._radius = radius
        self.cull_dist = get_global_config()["scenario_radius"]
        super(NuPlanMap, self).__init__(dict(id=map_name), random_seed=random_seed)

    @property
    def nuplan_center(self):
        return self._center

    def attach_to_world(self, center_point, parent_np=None, physics_world=None):
        parent_node_path, physics_world = self.engine.worldNP or parent_np, self.engine.physics_world or physics_world
        self.road_network = self.road_network_type()
        for block in self.blocks:
            # block.block_network.show_bounding_box(self.engine)
            if not SceneCull.out_of_bounding_box(block.bounding_box, np.array(center_point) - self.nuplan_center,
                                                 self.cull_dist):
                self.road_network.add(block.block_network)
                block.attach_to_world(parent_node_path, physics_world)
                self.attached_blocks.append(block)
        if not self.engine.global_config["load_city_map"]:
            self.boundary_block.attach_to_world(parent_node_path, physics_world)

    def detach_from_world(self, physics_world=None):
        if not self.engine.global_config["load_city_map"]:
            self.boundary_block.detach_from_world(self.engine.physics_world or physics_world)
        for block in self.attached_blocks:
            block.detach_from_world(self.engine.physics_world or physics_world)

    def _generate(self):
        logger.info("\n \n ############### Start Building Map: {} ############### \n".format(self.map_name))
        np.seterr(all='ignore')
        map_api = self._nuplan_map_api
        # Center is Important !
        center = self.nuplan_center
        layer_names = [
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.LANE,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.ROADBLOCK_CONNECTOR,

            # unsupported yet
            # SemanticMapLayer.STOP_SIGN,
            # SemanticMapLayer.DRIVABLE_AREA,
        ]
        center_for_query = Point2D(*metadrive_to_nuplan_vector(center))
        nearest_vector_map = map_api.get_proximal_map_objects(center_for_query, self._radius, layer_names)
        # Filter out stop polygons in turn stop
        if SemanticMapLayer.STOP_LINE in nearest_vector_map:
            stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
            nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
                stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
            ]

        block_polygons = []
        # Lane and lane line
        block_index = 0
        name = {SemanticMapLayer.ROADBLOCK: "Road Block", SemanticMapLayer.ROADBLOCK_CONNECTOR: "Road Connector"}

        for layer in tqdm.tqdm([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]):
            for block in tqdm.tqdm(nearest_vector_map[layer], leave=False, desc="Building {}".format(name[layer])):
                road_block = NuPlanBlock(block_index, self.road_network, 0, self.map_name, self.nuplan_center)

                # We implement the sample() function outside the Block instance, block._sample() will do nothing
                def _sample_topology():
                    for lane_meta_data in block.interior_edges:
                        if hasattr(lane_meta_data, "baseline_path"):
                            road_block.block_network.add_lane(
                                NuPlanLane(
                                    nuplan_center=center,
                                    lane_meta_data=lane_meta_data,
                                    need_lane_localization=self.need_lane_localization
                                )
                            )
                            is_connector = True if layer == SemanticMapLayer.ROADBLOCK_CONNECTOR else False
                            road_block.set_lane_line(
                                lane_meta_data, is_road_connector=is_connector, nuplan_center=center
                            )

                if layer == SemanticMapLayer.ROADBLOCK:
                    block_polygons.append(block.polygon)
                block_index += 1
                setattr(road_block, "_sample_topology", _sample_topology)
                road_block.construct_block(self.engine.worldNP, self.engine.physics_world, attach_to_world=False)
                self.blocks.append(road_block)
                # intersection road connector

        self.boundary_block = NuPlanBlock(block_index, self.road_network, 0, self.map_name, self.nuplan_center)
        interpolygons = [block.polygon for block in nearest_vector_map[SemanticMapLayer.INTERSECTION]]
        # logger.warning("Stop using boundaries! Use exterior instead!")
        boundaries = gpd.GeoSeries(unary_union(interpolygons + block_polygons)).boundary.explode(index_parts=True)
        # boundaries.plot()
        # plt.show()
        for idx, boundary in enumerate(boundaries[0]):
            block_points = np.array(list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1])))
            block_points = nuplan_to_metadrive_vector(block_points, self.nuplan_center)
            id = "boundary_{}".format(idx)
            self.boundary_block.lines[id] = LaneLineProperty(
                id, block_points, PGLineColor.GREY, PGLineType.CONTINUOUS, in_road_connector=False
            )
        self.boundary_block.construct_block(self.engine.worldNP, self.engine.physics_world, attach_to_world=True)
        np.seterr(all='warn')

        logger.info("\n \n ############### Finish Building Map: {} ############### \n".format(self.map_name))

    def play(self):
        # For debug
        for b in self.blocks:
            b._create_in_world(skip=True)
            b.attach_to_world(self.engine.worldNP, self.engine.physics_world)
            b.detach_from_world(self.engine.physics_world)

    def nuplan_to_metadrive_position(self, pos):
        return nuplan_to_metadrive_vector(pos, self.nuplan_center)

    @property
    def road_network_type(self):
        return EdgeRoadNetwork

    def destroy(self):
        self.map_name = None
        super(NuPlanMap, self).destroy()

    def __del__(self):
        # self.destroy()
        logging.debug("Map is Released")
        print("[NuPlanMap] Map is Released")

    def show_coordinates(self):
        lanes = [lane_info.lane for lane_info in self.road_network.graph.values()]
        self.engine.show_lane_coordinates(lanes)

    def get_boundary_line_vector(self, interval):
        ret = {}
        for block in self.attached_blocks + [self.boundary_block]:
            for boundary in block.lines.values():
                type = boundary.type
                map_feat_id = str(boundary.id)

                if type == PGLineType.BROKEN:
                    ret[map_feat_id] = {
                        "type": MetaDriveType.LINE_BROKEN_SINGLE_YELLOW
                        if boundary.color == PGLineColor.YELLOW else MetaDriveType.LINE_BROKEN_SINGLE_WHITE,
                        "polyline": boundary.points
                    }
                else:
                    ret[map_feat_id] = {
                        "polyline": boundary.points,
                        "type": MetaDriveType.LINE_SOLID_SINGLE_YELLOW
                        if boundary.color == PGLineColor.YELLOW else MetaDriveType.LINE_SOLID_SINGLE_WHITE
                    }
        return ret

    # def get_center_point(self):
    #     "Map is set to 0,0 in nuplan map"
    #     return [0, 0]


if __name__ == "__main__":
    from metadrive.envs.real_data_envs.nuplan_env import NuPlanEnv
    from metadrive.manager.nuplan_data_manager import NuPlanDataManager
    from metadrive.engine.engine_utils import initialize_engine, set_global_random_seed

    default_config = NuPlanEnv.default_config()
    default_config["use_render"] = True
    default_config["debug"] = True
    default_config["show_coordinates"] = True
    default_config["debug_static_world"] = True
    engine = initialize_engine(default_config)
    set_global_random_seed(0)

    engine.data_manager = NuPlanDataManager()
    engine.data_manager.seed(0)

    center = nuplan_to_metadrive_vector([664396, 3997613])

    map = NuPlanMap(map_name=0, nuplan_center=center, radius=500)
    map.attach_to_world(center)
    # engine.enableMouse()
    map.road_network.show_bounding_box(engine, (1, 0, 0, 1))
    lanes = [lane_info.lane for lane_info in map.road_network.graph.values()]
    engine.show_lane_coordinates(lanes)

    def detach_map():
        map.road_network.remove_bounding_box()
        map.detach_from_world()

    def attach_map():
        position = np.array(center)
        map.attach_to_world(position)
        map.road_network.show_bounding_box(engine, (1, 0, 0, 1))
        engine.main_camera.set_bird_view_pos(pos)

    engine.accept("d", detach_map)
    engine.accept("a", attach_map)

    pos = map.get_center_point()
    engine.main_camera.set_bird_view_pos(pos)

    while True:
        map.engine.step()
