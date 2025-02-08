import logging
import math
from abc import ABC

import cv2
import numpy as np
from panda3d.core import NodePath, Vec3

from metadrive.base_class.base_runnable import BaseRunnable
from metadrive.constants import CamMask
from metadrive.constants import MapTerrainSemanticColor, MetaDriveType, PGDrivableAreaProperty
from metadrive.utils.shapely_utils.geom import find_longest_edge

logger = logging.getLogger(__name__)


class BaseMap(BaseRunnable, ABC):
    """
    Base class for Map generation!
    """
    # only used to save and read maps
    FILE_SUFFIX = ".json"

    # define string in json and config
    SEED = "seed"
    LANE_WIDTH = "lane_width"
    LANE_WIDTH_RAND_RANGE = "lane_width_rand_range"
    LANE_NUM = "lane_num"
    BLOCK_ID = "id"
    BLOCK_SEQUENCE = "block_sequence"
    PRE_BLOCK_SOCKET_INDEX = "pre_block_socket_index"

    # generate_method
    GENERATE_CONFIG = "config"
    GENERATE_TYPE = "type"

    # default lane parameter
    MAX_LANE_WIDTH = 4.5
    MIN_LANE_WIDTH = 3.0
    MAX_LANE_NUM = 3
    MIN_LANE_NUM = 2

    def __init__(self, map_config: dict = None, random_seed=None):
        """
        Map can be stored and recover to save time when we access the map encountered before
        """
        assert random_seed is None
        # assert random_seed == map_config[
        #     self.SEED
        # ], "Global seed {} should equal to seed in map config {}".format(random_seed, map_config[self.SEED])
        super(BaseMap, self).__init__(config=map_config)

        # map features
        self.road_network = self.road_network_type()
        self.crosswalks = {}
        self.sidewalks = {}

        # A flatten representation of blocks, might cause chaos in city-level generation.
        self.blocks = []

        # Generate map and insert blocks
        # self.engine = get_engine()
        self._generate()
        assert self.blocks, "The generate methods does not fill blocks!"

        # lanes debug
        self.lane_coordinates_debug_node = None
        if self.engine.global_config["show_coordinates"]:
            self.show_coordinates()

        #  a trick to optimize performance
        self.spawn_roads = None
        self.detach_from_world()

        # save a backup
        # self._semantic_map = None
        # self._height_map = None

    def _generate(self):
        """Key function! Please overwrite it! This func aims at fill the self.road_network adn self.blocks"""
        raise NotImplementedError("Please use child class like PGMap to replace Map!")

    def attach_to_world(self, parent_np=None, physics_world=None):
        parent_node_path, physics_world = self.engine.worldNP or parent_np, self.engine.physics_world or physics_world
        for block in self.blocks:
            block.attach_to_world(parent_node_path, physics_world)
        if self.lane_coordinates_debug_node is not None:
            self.lane_coordinates_debug_node.reparentTo(parent_node_path)

    def detach_from_world(self, physics_world=None):
        for block in self.blocks:
            block.detach_from_world(self.engine.physics_world or physics_world)
        if self.lane_coordinates_debug_node is not None:
            self.lane_coordinates_debug_node.detachNode()

    def get_meta_data(self):
        """
        Save the generated map to map file
        """
        return dict(map_type=self.class_name, map_features=self.get_map_features())

    @property
    def num_blocks(self):
        return len(self.blocks)

    def destroy(self):
        self.detach_from_world()
        # if self._semantic_map is not None:
        #     del self._semantic_map
        #     self._semantic_map = None
        # if self._height_map is not None:
        #     del self._height_map
        #     self._height_map = None

        for block in self.blocks:
            block.destroy()
        self.blocks = []

        if self.road_network is not None:
            self.road_network.destroy()
        self.road_network = None
        self.spawn_roads = None

        if self.lane_coordinates_debug_node is not None:
            self.lane_coordinates_debug_node.removeNode()
        super(BaseMap, self).destroy()

    @property
    def road_network_type(self):
        raise NotImplementedError

    def get_center_point(self):
        x_min, x_max, y_min, y_max = self.road_network.get_bounding_box()
        return (x_max + x_min) / 2, (y_max + y_min) / 2

    def __del__(self):
        # self.destroy()
        logger.debug("{} 2is being deleted.".format(type(self)))

    def show_coordinates(self):
        pass

    def _show_coordinates(self, lanes):
        if self.lane_coordinates_debug_node is not None:
            self.lane_coordinates_debug_node.detachNode()
            self.lane_coordinates_debug_node.removeNode()

        self.lane_coordinates_debug_node = NodePath("Lane Coordinates debug")
        self.lane_coordinates_debug_node.hide(CamMask.AllOn)
        self.lane_coordinates_debug_node.show(CamMask.MainCam)
        for lane in lanes:
            long_start = lateral_start = lane.position(0, 0)
            lateral_end = lane.position(0, 2)

            long_end = long_start + lane.heading_at(0) * 4
            np_y = self.engine._draw_line_3d(Vec3(*long_start, 0), Vec3(*long_end, 0), color=[0, 1, 0, 1], thickness=3)
            np_x = self.engine._draw_line_3d(
                Vec3(*lateral_start, 0), Vec3(*lateral_end, 0), color=[1, 0, 0, 1], thickness=3
            )
            np_x.reparentTo(self.lane_coordinates_debug_node)
            np_y.reparentTo(self.lane_coordinates_debug_node)

    def get_map_features(self, interval=2):
        """
        Get the map features represented by a set of point lists or polygons
        Args:
            interval: Sampling rate

        Returns: None

        """
        map_features = self.road_network.get_map_features(interval)
        boundary_line_vector = self.get_boundary_line_vector(interval)
        map_features.update(boundary_line_vector)
        map_features.update(self.sidewalks)
        map_features.update(self.crosswalks)
        return map_features

    def get_boundary_line_vector(self, interval):
        return {}

    # @time_me
    def get_semantic_map(
        self,
        center_point,
        size=512,
        pixels_per_meter=8,
        color_setting=MapTerrainSemanticColor,
        line_sample_interval=2,
        yellow_line_thickness=1,
        white_line_thickness=1,
        layer=("lane_line", "lane")
    ):
        """
        Get semantics of the map for terrain generation
        :param center_point: 2D point, the center to select the rectangular region
        :param size: [m] length and width
        :param pixels_per_meter: the returned map will be in (size*pixels_per_meter * size*pixels_per_meter) size
        :param color_setting: color palette for different attribute. When generating terrain, make sure using
        :param line_sample_interval: [m] It determines the resolution of sampled points.
        :param polyline_thickness: [m] The width of the road lines
        :param layer: layer to get
        MapTerrainAttribute
        :return: semantic map
        """
        center_p = center_point

        # if self._semantic_map is None:
        all_lanes = self.get_map_features(interval=line_sample_interval)
        polygons = []
        polylines = []

        points_to_skip = math.floor(PGDrivableAreaProperty.STRIPE_LENGTH * 2 / line_sample_interval)
        for obj in all_lanes.values():
            if MetaDriveType.is_lane(obj["type"]) and "lane" in layer:
                polygons.append((obj["polygon"], color_setting.get_color(obj["type"])))
            elif "lane_line" in layer and (MetaDriveType.is_road_line(obj["type"])
                                           or MetaDriveType.is_road_boundary_line(obj["type"])):
                if MetaDriveType.is_broken_line(obj["type"]):
                    for index in range(0, len(obj["polyline"]) - 1, points_to_skip * 2):
                        if index + points_to_skip < len(obj["polyline"]):
                            polylines.append(
                                (
                                    [obj["polyline"][index],
                                     obj["polyline"][index + points_to_skip]], color_setting.get_color(obj["type"])
                                )
                            )
                else:
                    polylines.append((obj["polyline"], color_setting.get_color(obj["type"])))

        size = int(size * pixels_per_meter)
        mask = np.zeros([size, size, 1], dtype=np.uint8)
        mask[..., 0] = color_setting.get_color(MetaDriveType.GROUND)
        # create an example bounding box polygon
        # for idx in range(len(polygons)):
        for polygon, color in polygons:
            points = [
                [
                    int((x - center_p[0]) * pixels_per_meter + size / 2),
                    int((y - center_p[1]) * pixels_per_meter) + size / 2
                ] for x, y in polygon
            ]
            cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=color)
        for line, color in polylines:
            points = [
                [
                    int((p[0] - center_p[0]) * pixels_per_meter + size / 2),
                    int((p[1] - center_p[1]) * pixels_per_meter) + size / 2
                ] for p in line
            ]
            thickness = yellow_line_thickness if color == MapTerrainSemanticColor.YELLOW else white_line_thickness
            # thickness = min(thickness, 2)  # clip
            cv2.polylines(mask, np.array([points]).astype(np.int32), False, color, thickness)

        if "crosswalk" in layer:
            for id, sidewalk in self.crosswalks.items():
                polygon = sidewalk["polygon"]
                points = [
                    [
                        int((x - center_p[0]) * pixels_per_meter + size / 2),
                        int((y - center_p[1]) * pixels_per_meter) + size / 2
                    ] for x, y in polygon
                ]
                # edges = find_longest_parallel_edges(polygon)
                # p_1, p_2 = edges[0]
                p_1, p_2 = find_longest_edge(polygon)[0]
                dir = (
                    p_2[0] - p_1[0],
                    p_2[1] - p_1[1],
                )
                # 0-2pi
                angle = np.arctan2(*dir) / np.pi * 180 + 180
                angle = int(angle / 2) + color_setting.get_color(MetaDriveType.CROSSWALK)
                cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=angle)

        #     self._semantic_map = mask
        # return self._semantic_map
        return mask

    # @time_me
    def get_height_map(
        self,
        center_point,
        size=2048,
        pixels_per_meter=1,
        extension=2,
        height=1,
    ):
        """
        Get height of the map for terrain generation
        :param size: [m] length and width
        :param center_point: 2D point, the center to select the rectangular region
        :param pixels_per_meter: the returned map will be in (size*pixels_per_meter * size*pixels_per_meter) size
        :param extension: If > 1, the returned height map's drivable region will be enlarged.
        :param height: height of drivable area.
        :return: heightfield image in uint 16 nparray
        """
        center_p = center_point

        # if self._height_map is None:
        extension = max(1, extension)
        all_lanes = self.get_map_features()
        polygons = []

        for obj in all_lanes.values():
            if MetaDriveType.is_lane(obj["type"]):
                polygons.append(obj["polygon"])

        size = int(size * pixels_per_meter)
        mask = np.zeros([size, size, 1])

        need_scale = abs(extension - 1) > 1e-1

        for sidewalk in self.sidewalks.values():
            polygons.append(sidewalk["polygon"])

        for polygon in polygons:
            points = [
                [
                    int((x - center_p[0]) * pixels_per_meter + size / 2),
                    int((y - center_p[1]) * pixels_per_meter) + size / 2
                ] for x, y in polygon
            ]
            cv2.fillPoly(mask, np.asarray([points]).astype(np.int32), color=[height])
        if need_scale:
            # Define a kernel. A 3x3 rectangle kernel
            kernel = np.ones(((extension + 1) * pixels_per_meter, (extension + 1) * pixels_per_meter), np.uint8)

            # Apply dilation
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = np.expand_dims(mask, axis=-1)
        #     self._height_map = mask
        # return self._height_map
        return mask

    def show_bounding_box(self):
        """
        Draw the bounding box of map in 3D renderer
        Returns: None

        """
        self.road_network.show_bounding_box(self.engine)
