import logging
import math

import cv2
import numpy as np
from metadrive.base_class.base_runnable import BaseRunnable
from metadrive.constants import MapTerrainSemanticColor, MetaDriveType, DrivableAreaProperty
from metadrive.engine.engine_utils import get_global_config
from shapely.geometry import Polygon, MultiPolygon

logger = logging.getLogger(__name__)


class BaseMap(BaseRunnable):
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
        self.film_size = (get_global_config()["draw_map_resolution"], get_global_config()["draw_map_resolution"])
        self.road_network = self.road_network_type()

        # A flatten representation of blocks, might cause chaos in city-level generation.
        self.blocks = []

        # Generate map and insert blocks
        # self.engine = get_engine()
        self._generate()
        assert self.blocks, "The generate methods does not fill blocks!"

        #  a trick to optimize performance
        self.spawn_roads = None
        self.detach_from_world()

        # save a backup
        self._semantic_map = None
        self._height_map = None

        if self.engine.global_config["show_coordinates"]:
            self.show_coordinates()

    def _generate(self):
        """Key function! Please overwrite it! This func aims at fill the self.road_network adn self.blocks"""
        raise NotImplementedError("Please use child class like PGMap to replace Map!")

    def attach_to_world(self, parent_np=None, physics_world=None):
        parent_node_path, physics_world = self.engine.worldNP or parent_np, self.engine.physics_world or physics_world
        for block in self.blocks:
            block.attach_to_world(parent_node_path, physics_world)

    def detach_from_world(self, physics_world=None):
        for block in self.blocks:
            block.detach_from_world(self.engine.physics_world or physics_world)

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
        if self._semantic_map is not None:
            del self._semantic_map
            self._semantic_map = None
        if self._height_map is not None:
            del self._height_map
            self._height_map = None

        for block in self.blocks:
            block.destroy()
        self.blocks = []

        if self.road_network is not None:
            self.road_network.destroy()
        self.road_network = None

        self.spawn_roads = None

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

    def get_map_features(self, interval=2):
        map_features = self.road_network.get_map_features(interval)

        boundary_line_vector = self.get_boundary_line_vector(interval)

        map_features.update(boundary_line_vector)

        return map_features

    def get_boundary_line_vector(self, interval):
        return {}

    # @time_me
    def get_semantic_map(
        self,
        size=512,
        pixels_per_meter=8,
        color_setting=MapTerrainSemanticColor,
        line_sample_interval=2,
        polyline_thickness=1,
        layer=("lane_line", "lane")
    ):
        """
        Get semantics of the map
        :param size: [m] length and width
        :param pixels_per_meter: the returned map will be in (size*pixels_per_meter * size*pixels_per_meter) size
        :param color_setting: color palette for different attribute. When generating terrain, make sure using
        :param line_sample_interval: [m] It determines the resolution of sampled points.
        :param layer: layer to get
        MapTerrainAttribute
        :return: heightfield image
        """
        if self._semantic_map is None:
            all_lanes = self.get_map_features(interval=line_sample_interval)
            polygons = []
            polylines = []

            points_to_skip = math.floor(DrivableAreaProperty.STRIPE_LENGTH * 2 / line_sample_interval)
            for obj in all_lanes.values():
                if MetaDriveType.is_lane(obj["type"]) and "lane" in layer:
                    polygons.append((obj["polygon"], MapTerrainSemanticColor.get_color(obj["type"])))
                elif "lane_line" in layer and (MetaDriveType.is_road_line(obj["type"])
                                               or MetaDriveType.is_sidewalk(obj["type"])):
                    if MetaDriveType.is_broken_line(obj["type"]):
                        for index in range(0, len(obj["polyline"]) - 1, points_to_skip * 2):
                            if index + points_to_skip < len(obj["polyline"]):
                                polylines.append(
                                    (
                                        [obj["polyline"][index], obj["polyline"][index + points_to_skip]],
                                        MapTerrainSemanticColor.get_color(obj["type"])
                                    )
                                )
                    else:
                        polylines.append((obj["polyline"], MapTerrainSemanticColor.get_color(obj["type"])))

            size = int(size * pixels_per_meter)
            mask = np.zeros([size, size, 4], dtype=np.float32)
            mask[..., 0:] = color_setting.get_color(MetaDriveType.GROUND)
            # create an example bounding box polygon
            # for idx in range(len(polygons)):
            center_p = self.get_center_point()
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
                cv2.polylines(mask, np.array([points]).astype(np.int32), False, color, polyline_thickness)
            self._semantic_map = mask
        return self._semantic_map

    def get_height_map(
        self,
        size=2048,
        pixels_per_meter=1,
        extension=2,
        height=1,
    ):
        """
        Get height of the map
        :param size: [m] length and width
        :param pixels_per_meter: the returned map will be in (size*pixels_per_meter * size*pixels_per_meter) size
        :param extension: If > 1, the returned height map's drivable region will be enlarged.
        :param height: height of drivable area.
        :return: heightfield image in uint 16 nparray
        """
        if self._height_map is None:
            extension = max(1, extension)
            all_lanes = self.get_map_features()
            polygons = []

            for obj in all_lanes.values():
                if MetaDriveType.is_lane(obj["type"]):
                    polygons.append(obj["polygon"])

            size = int(size * pixels_per_meter)
            mask = np.zeros([size, size, 1])

            center_p = self.get_center_point()
            need_scale = abs(extension - 1) > 1e-1
            for polygon in polygons:
                if need_scale:
                    scaled_polygon = Polygon(polygon).buffer(extension, join_style=2)
                    if isinstance(scaled_polygon, MultiPolygon):
                        scaled_polygons = scaled_polygon.geoms
                    else:
                        scaled_polygons = [scaled_polygon]
                    for scaled_polygon in scaled_polygons:
                        points = [
                            [
                                int(
                                    (scaled_polygon.exterior.coords.xy[0][index] - center_p[0]) * pixels_per_meter +
                                    size / 2
                                ),
                                int((scaled_polygon.exterior.coords.xy[1][index] - center_p[1]) * pixels_per_meter) +
                                size / 2
                            ] for index in range(len(scaled_polygon.exterior.coords.xy[0]))
                        ]
                else:
                    points = [
                        [
                            int((x - center_p[0]) * pixels_per_meter + size / 2),
                            int((y - center_p[1]) * pixels_per_meter) + size / 2
                        ] for x, y in polygon
                    ]
                cv2.fillPoly(mask, np.asarray([points]).astype(np.int32), color=[height])
            self._height_map = mask
        return self._height_map
