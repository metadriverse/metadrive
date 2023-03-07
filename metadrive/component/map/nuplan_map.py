import logging
import tqdm
import logging

import tqdm

from metadrive.engine.engine_utils import get_global_config
import geopandas as gpd
import numpy as np
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from shapely.ops import unary_union

from metadrive.component.lane.nuplan_lane import NuPlanLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.nuplan_block.nuplan_block import LaneLineProperty
from metadrive.component.nuplan_block.nuplan_block import NuPlanBlock
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.constants import LineColor, LineType
from metadrive.engine.scene_cull import SceneCull

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NuPlanMap(BaseMap):
    def __init__(self, map_name, nuplan_center, random_seed=None):

        self.map_name = map_name
        self._center = np.array(nuplan_center)
        self._nuplan_map_api = self.engine.data_manager.current_scenario.map_api
        self._attached_block = []
        self.boundary_block = None  # it won't be detached
        self._radius = get_global_config()["city_map_radius"]
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
            if not SceneCull.out_of_bounding_box(block.bounding_box,
                                                 np.array(center_point) - self.nuplan_center,
                                                 self.cull_dist):
                self.road_network.add(block.block_network)
                block.attach_to_world(parent_node_path, physics_world)
                self._attached_block.append(block)

    def detach_from_world(self, physics_world=None):
        for block in self._attached_block:
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

        nearest_vector_map = map_api.get_proximal_map_objects(Point2D(*center), self._radius, layer_names)
        # Filter out stop polygons in turn stop
        if SemanticMapLayer.STOP_LINE in nearest_vector_map:
            stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
            nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
                stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
            ]

        block_polygons = []
        # Lane and lane line
        block_index = 0
        name = {SemanticMapLayer.ROADBLOCK: "Road Block",
                SemanticMapLayer.ROADBLOCK_CONNECTOR: "Road Connector"}

        for layer in tqdm.tqdm([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]):
            for block in tqdm.tqdm(nearest_vector_map[layer], leave=False, desc="Building {}".format(name[layer])):
                road_block = NuPlanBlock(block_index, self.road_network, 0, self.map_name, self.nuplan_center)

                # We implement the sample() function outside the Block instance, block._sample() will do nothing
                def _sample_topology():
                    for lane_meta_data in block.interior_edges:
                        if hasattr(lane_meta_data, "baseline_path"):
                            road_block.block_network.add_lane(
                                NuPlanLane(nuplan_center=center, lane_meta_data=lane_meta_data))
                            is_connector = True if layer == SemanticMapLayer.ROADBLOCK_CONNECTOR else False
                            road_block.set_lane_line(lane_meta_data, is_road_connector=is_connector,
                                                     nuplan_center=center)

                if layer == SemanticMapLayer.ROADBLOCK:
                    block_polygons.append(block.polygon)
                block_index += 1
                setattr(road_block, "_sample_topology", _sample_topology)
                road_block.construct_block(self.engine.worldNP, self.engine.physics_world, attach_to_world=False)
                self.blocks.append(road_block)
                # intersection road connector

        self.boundary_block = NuPlanBlock(block_index, self.road_network, 0, self.map_name, self.nuplan_center)
        interpolygons = [block.polygon for block in nearest_vector_map[SemanticMapLayer.INTERSECTION]]
        boundaries = gpd.GeoSeries(unary_union(interpolygons + block_polygons)).boundary.explode()
        # boundaries.plot()
        # plt.show()
        for idx, boundary in enumerate(boundaries[0]):
            block_points = np.array(list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1])))
            block_points -= center
            self.boundary_block.boundaries["boundary_{}".format(idx)] = LaneLineProperty(
                block_points, LineColor.GREY, LineType.CONTINUOUS, in_road_connector=False
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

    def metadrive_to_nuplan_position(self, pos):
        return pos[0] + self.nuplan_center[0], pos[1] + self.nuplan_center[1]

    def nuplan_to_metadrive_position(self, pos):
        return pos[0] - self.nuplan_center[0], pos[1] - self.nuplan_center[1]

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


if __name__ == "__main__":
    from metadrive.envs.real_data_envs.nuplan_env import NuPlanEnv
    from metadrive.manager.nuplan_data_manager import NuPlanDataManager
    from metadrive.engine.engine_utils import initialize_engine, set_global_random_seed

    default_config = NuPlanEnv.default_config()
    default_config["use_render"] = True
    default_config["city_map_radius"] = 500
    default_config["debug"] = True
    default_config["show_coordinates"] = True
    default_config["debug_static_world"] = True
    engine = initialize_engine(default_config)
    set_global_random_seed(0)

    engine.data_manager = NuPlanDataManager()
    engine.data_manager.seed(0)
    map = NuPlanMap(map_name=0, nuplan_center=[664396.54429387, 3997613.41534655])
    map.attach_to_world([664396.54429387, 3997613.41534655])
    # engine.enableMouse()
    map.road_network.show_bounding_box(engine, (1, 0, 0, 1))


    def detach_map():
        map.road_network.remove_bounding_box()
        map.detach_from_world()


    def attach_map():
        position = np.array([664396, 3997613 + np.random.randint(0, 1000)])
        map.attach_to_world(position)
        map.road_network.show_bounding_box(engine, (1, 0, 0, 1))
        engine.main_camera.set_bird_view_pos(pos)


    engine.accept("d", detach_map)
    engine.accept("a", attach_map)

    # argoverse data set is as the same coordinates as panda3d
    pos = map.get_center_point()
    engine.main_camera.set_bird_view_pos(pos)

    while True:
        map.engine.step()
