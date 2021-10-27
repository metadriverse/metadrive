from metadrive.component.road_network.base_road_network import BaseRoadNetwork, LaneIndex
import copy
import logging
from typing import List, Tuple, Dict

import numpy as np
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.road_network.road import Road
from metadrive.component.road_network.base_road_network import BaseRoadNetwork
from metadrive.constants import Decoration
from metadrive.utils.math_utils import get_boxes_bounding_box
from metadrive.utils.scene_utils import get_lanes_bounding_box

from collections import namedtuple

lane_info = namedtuple("neighbor_lanes", "lane entry_lanes exit_lanes left_lanes right_lanes")


class EdgeRoadNetwork(BaseRoadNetwork):
    """
    Compared to NodeRoadNetwork representing the relation of lanes in a node-based graph, EdgeRoadNetwork stores the
    relationship in edge-based graph, which is more common in real map representation
    """

    def __init__(self):
        super(EdgeRoadNetwork, self).__init__()
        self.graph = {}

    def add_lane(self, lane) -> None:
        self.graph[lane.index] = lane_info(lane=lane,
                                           entry_lanes=lane.entry_lanes,
                                           exit_lanes=lane.exit_lanes,
                                           left_lanes=lane.left_lanes,
                                           right_lanes=lane.right_lanes)

    def get_lane(self, index: LaneIndex):
        return self.graph[index]

    def __isub__(self, other):
        for id, lane_info in other.graph.items():
            self.graph.pop(id)
        return self

    def add(self, other, no_intersect=True):
        for id, lane_info in other.graph.items():
            if no_intersect:
                assert id not in self.graph.keys(), "Intersect: {} exists in two network".format(id)
            self.graph[id] = other.graph[id]
        return self

    def _get_bounding_box(self):
        """
       By using this bounding box, the edge length of x, y direction and the center of this road network can be
       easily calculated.
       :return: minimum x value, maximum x value, minimum y value, maximum y value
       """
        lanes = []
        for id, lane_info, in self.graph.items():
            lanes.append(lane_info.lane)
        res_x_max, res_x_min, res_y_max, res_y_min = get_boxes_bounding_box([get_lanes_bounding_box(lanes)])
        return res_x_min, res_x_max, res_y_min, res_y_max
