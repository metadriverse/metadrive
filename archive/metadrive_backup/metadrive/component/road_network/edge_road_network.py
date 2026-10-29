import logging
from collections import namedtuple
from typing import List

from metadrive.component.road_network.base_road_network import BaseRoadNetwork
from metadrive.component.road_network.base_road_network import LaneIndex
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.utils.math import get_boxes_bounding_box
from metadrive.utils.pg.utils import get_lanes_bounding_box

lane_info = namedtuple("edge_lane", ["lane", "entry_lanes", "exit_lanes", "left_lanes", "right_lanes"])


class EdgeRoadNetwork(BaseRoadNetwork):
    """
    Compared to NodeRoadNetwork representing the relation of lanes in a node-based graph, EdgeRoadNetwork stores the
    relationship in edge-based graph, which is more common in real map representation
    """
    def __init__(self):
        super(EdgeRoadNetwork, self).__init__()
        self.graph = {}

    def add_lane(self, lane) -> None:
        assert lane.index is not None, "Lane index can not be None"
        self.graph[lane.index] = lane_info(
            lane=lane,
            entry_lanes=lane.entry_lanes or [],
            exit_lanes=lane.exit_lanes or [],
            left_lanes=lane.left_lanes or [],
            right_lanes=lane.right_lanes or []
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

        :param start: starting edges
        :param goal: goal edge
        :return: list of paths from start to goal.
        """
        lanes = self.graph[start].left_lanes + self.graph[start].right_lanes + [start]

        queue = [(lane, [lane]) for lane in lanes]
        while queue:
            (lane, path) = queue.pop(0)
            if lane not in self.graph:
                yield []
            if len(self.graph[lane].exit_lanes) == 0:
                continue
            for _next in set(self.graph[lane].exit_lanes):
                if _next in path:
                    # circle
                    continue
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
        """
        Destroy all lanes in this road network
        Returns: None

        """
        super(EdgeRoadNetwork, self).destroy()
        if self.graph is not None:
            for k, v in self.graph.items():
                v.lane.destroy()
                self.graph[k]: lane_info = None
            self.graph = None

    def __del__(self):
        logging.debug("{} is released".format(self.__class__.__name__))

    def get_map_features(self, interval=2):

        ret = {}
        for id, lane_info in self.graph.items():
            assert id == lane_info.lane.index
            ret[id] = {
                SD.POLYLINE: lane_info.lane.get_polyline(interval),
                SD.POLYGON: lane_info.lane.polygon,
                SD.TYPE: lane_info.lane.metadrive_type,
                SD.ENTRY: lane_info.entry_lanes,
                SD.EXIT: lane_info.exit_lanes,
                SD.LEFT_NEIGHBORS: lane_info.left_lanes,
                SD.RIGHT_NEIGHBORS: lane_info.right_lanes,
                "speed_limit_kmh": lane_info.lane.speed_limit
            }
        return ret

    def get_all_lanes(self):
        """
        This function will return all lanes in the road network
        :return: list of lanes
        """
        ret = []
        for id, lane_info in self.graph.items():
            ret.append(lane_info.lane)
        return ret


class OpenDriveRoadNetwork(EdgeRoadNetwork):
    def add_lane(self, lane) -> None:
        assert lane.index is not None, "Lane index can not be None"
        self.graph[lane.index] = lane_info(
            lane=lane, entry_lanes=None, exit_lanes=None, left_lanes=None, right_lanes=None
        )
