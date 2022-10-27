from metadrive.component.road_network.base_road_network import BaseRoadNetwork, LaneIndex
import gc
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
        self.graph[lane.index] = lane_info(
            lane=lane,
            entry_lanes=lane.entry_lanes,
            exit_lanes=lane.exit_lanes,
            left_lanes=lane.left_lanes,
            right_lanes=lane.right_lanes
        )

    def get_lane(self, index: LaneIndex):
        return self.graph[index].lane

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

    def shortest_path(self, start: str, goal: str):
        return next(self.bfs_paths(start, goal), [])

    def bfs_paths(self, start: str, goal: str) -> List[List[str]]:
        """
        Breadth-first search of all routes from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: list of paths from start to goal.
        """
        queue = [(start, [start])]
        while queue:
            (node, path) = queue.pop(0)
            if node not in self.graph:
                yield []
            for _next in set(self.graph[node].exit_lanes) - set(path):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    def get_peer_lanes_from_index(self, lane_index):
        info: lane_info = self.graph[lane_index]
        ret = [self.graph[lane_index].lane]
        for left_n in info.left_lanes:
            ret.append(self.graph[left_n["id"]].lane)
        for right_n in info.right_lanes:
            ret.append(self.graph[right_n["id"]].lane)
        return ret

    def destroy(self):
        super(EdgeRoadNetwork, self).destroy()
        for k, v in self.graph.items():
            v.lane.destroy()
            self.graph[k]: lane_info = None
        self.graph = None

    def __del__(self):
        logging.debug("{} is released".format(self.__class__.__name__))


class OpenDriveRoadNetwork(EdgeRoadNetwork):
    def add_lane(self, lane) -> None:
        self.graph[lane.index] = lane_info(
            lane=lane, entry_lanes=None, exit_lanes=None, left_lanes=None, right_lanes=None
        )
