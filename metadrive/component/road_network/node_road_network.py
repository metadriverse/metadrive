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

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]


class NodeRoadNetwork(BaseRoadNetwork):
    """
    This network uses two node to describe the road network graph, and the edges between two nodes represent road, which
    is a list of lanes connecting two lanes
    """
    graph: Dict[str, Dict[str, List[AbstractLane]]]

    def __init__(self, debug=False):
        super(NodeRoadNetwork, self).__init__()
        self.graph = {}
        self.indices = []
        self._graph_helper = None
        self.debug = debug
        self.is_initialized = False

    def after_init(self):
        assert not self.is_initialized
        self._update_indices()
        self._init_graph_helper()
        self.is_initialized = True

    def add(self, other, no_intersect=True):
        assert not self.is_initialized, "Adding new blocks should be done before road network initialization!"
        set_1 = set(self.graph) - {Decoration.start, Decoration.end}
        set_2 = set(other.graph) - {Decoration.start, Decoration.end}
        intersect = set_1.intersection(set_2)
        if len(intersect) != 0 and no_intersect:
            raise ValueError("Same start node {} in two road network".format(intersect))
        # handle decoration_lanes
        dec_lanes = self.get_all_decoration_lanes() + other.get_all_decoration_lanes()
        self.graph.update(copy.copy(other.graph))
        self.update_decoration_lanes(dec_lanes)
        return self

    def __isub__(self, other):
        intersection = self.graph.keys() & other.graph.keys() - {Decoration.start, Decoration.end}
        if len(intersection) != 0:
            for k in intersection:
                self.graph.pop(k, None)
        if Decoration.start in other.graph.keys():
            for lane in other.graph[Decoration.start][Decoration.end]:
                if lane in self.graph[Decoration.start][Decoration.end]:
                    self.graph[Decoration.start][Decoration.end].remove(lane)
        return self

    def get_all_decoration_lanes(self) -> List:
        if Decoration.start in self.graph:
            return self.graph[Decoration.start][Decoration.end]
        else:
            return []

    def update_decoration_lanes(self, lanes):
        if len(lanes) == 0:
            return
        if Decoration.start in self.graph:
            self.graph.pop(Decoration.start, None)
        self.graph[Decoration.start] = {Decoration.end: lanes}

    def clear(self):
        self.graph.clear()

    def get_positive_lanes(self):
        """
        In order to remain the lane index, ret is a 2-dim array structure like [Road_lanes[lane_1, lane_2]]
        """
        ret = []
        for _from, _to_dict in self.graph.items():
            for _to, lanes in _to_dict.items():
                road = Road(_from, _to)
                if not road.is_negative_road() and road.is_valid_road():
                    ret.append(lanes)
        return ret

    def get_negative_lanes(self):
        """
        In order to remain the lane index, ret is a 2-dim array structure like like [Road_lanes[lane_1, lane_2]]
        """
        ret = []
        for _from, _to_dict in self.graph.items():
            for _to, lanes in _to_dict.items():
                road = Road(_from, _to)
                if road.is_negative_road() and road.is_valid_road():
                    ret.append(lanes)
        return ret

    def _get_bounding_box(self):
        """
        By using this bounding box, the edge length of x, y direction and the center of this road network can be
        easily calculated.
        :return: minimum x value, maximum x value, minimum y value, maximum y value
        """
        boxes = []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                if len(lanes) == 0:
                    continue
                boxes.append(get_lanes_bounding_box(lanes))
        res_x_max, res_x_min, res_y_max, res_y_min = get_boxes_bounding_box(boxes)
        return res_x_min, res_x_max, res_y_min, res_y_max

    def remove_all_roads(self, start_node: str, end_node: str):
        """
        Remove all road between two road nodes
        :param start_node: start node name
        :param end_node: end node name
        :return: roads removed
        """
        ret = []
        paths = self.bfs_paths(start_node, end_node)
        for path in paths:
            for next_idx, node in enumerate(path[:-1], 1):
                road_removed = self.remove_road(Road(node, path[next_idx]))
                ret += road_removed
        return ret

    def remove_road(self, road):
        assert isinstance(road, Road), "Only Road Type can be deleted"
        ret = self.graph[road.start_node].pop(road.end_node)
        if len(self.graph[road.start_node]) == 0:
            self.graph.pop(road.start_node)
        return ret

    def add_road(self, road, lanes: List):
        assert isinstance(road, Road), "Only Road Type can be added to road network"
        if road.start_node not in self.graph:
            self.graph[road.start_node] = {}
        if road.end_node not in self.graph[road.start_node]:
            self.graph[road.start_node][road.end_node] = []
        self.graph[road.start_node][road.end_node] += lanes

    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:
        """
        A lane is encoded as an edge in the road network.

        :param _from: the node at which the lane starts.
        :param _to: the node at which the lane ends.
        :param AbstractLane lane: the lane geometry.
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def _init_graph_helper(self):
        self._graph_helper = GraphLookupTable(self.graph, self.debug)

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        Get the lane geometry corresponding to a given index in the road network.

        :param index: a tuple (origin node, destination node, lane id on the road).
        :return: the corresponding lane geometry.
        """
        _from, _to, _id = index
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        return self.graph[_from][_to][_id]

    def _update_indices(self):
        indexes = []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    indexes.append((_from, _to, _id))
        self.indices = indexes

    def get_closest_lane_index(self, position, return_all=False):
        return self._graph_helper.get(position, return_all)

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
            for _next in set(self.graph[node].keys()) - set(path):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    def shortest_path(self, start: str, goal: str) -> List[str]:
        """
        Breadth-first search of shortest checkpoints from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest checkpoints from start to goal.
        """
        start_road_node = start[0]
        assert start != goal
        return next(self.bfs_paths(start_road_node, goal), [])


class GraphLookupTable:
    def __init__(self, graph, debug):
        self.graph = graph
        self.debug = debug

    def get(self, position, return_all):
        log = dict()
        count = 0
        for _, (_from, to_dict) in enumerate(self.graph.items()):
            if _from == "decoration":
                continue
            for lanes_id, lanes in to_dict.items():
                lane = next(iter(lanes))
                log[count] = (lane.distance(position), (_from, lanes_id))
                count += 1

        distance_index_mapping = []
        for rank, candidate_count in enumerate(sorted(log, key=lambda key: log[key][0])):
            first_lane_distance, (section_id, lanes_id) = log[candidate_count]
            lanes = self.graph[section_id][lanes_id]
            for lane_id, lane in enumerate(lanes):
                if lanes_id == Decoration.start:
                    continue
                if lane_id == 0:
                    dist = first_lane_distance
                else:
                    dist = lane.distance(position)
                distance_index_mapping.append((dist, (section_id, lanes_id, lane_id)))
            # if rank > 10:
            #     # Take first rank 5 lanes into consideration. The number may related to the number of
            #     # lanes in intersection. We have 3 lanes in intersection, so computing the first 4 ranks can make
            #     # thing work. We choose take first 5 lanes into consideration.
            #     # In futurem we shall refactor the whole system, so this vulnerable code would be removed.
            #     break
        if self.graph.get(Decoration.start, False):
            for id, lane in enumerate(self.graph[Decoration.start][Decoration.end]):
                dist = lane.distance(position)
                distance_index_mapping.append((dist, (Decoration.start, Decoration.end, id)))

        distance_index_mapping = sorted(distance_index_mapping, key=lambda d: d[0])
        if return_all:
            return distance_index_mapping
        else:
            ret_ind = 0
            index = distance_index_mapping[ret_ind][1]
            distance = distance_index_mapping[ret_ind][0]
            return index, distance
