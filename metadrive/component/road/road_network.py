import copy
import logging
from typing import List, Tuple, Dict

import numpy as np

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.road.road import Road
from metadrive.constants import Decoration
from metadrive.utils.math_utils import get_boxes_bounding_box
from metadrive.utils.scene_utils import get_road_bounding_box

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]


class RoadNetwork:
    graph: Dict[str, Dict[str, List[AbstractLane]]]

    def __init__(self, debug=False):
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

    def __sub__(self, other):
        ret = RoadNetwork()
        ret.graph = self.graph
        ret -= other
        return ret

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
        for _from, _to_dict in self.graph:
            for _to, lanes in _to_dict:
                road = Road(_from, _to)
                if road.is_negative_road() and road.is_valid_road():
                    ret.append(lanes)
        return ret

    def get_bounding_box(self):
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
                boxes.append(get_road_bounding_box(lanes))
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

    def next_lane(
        self,
        current_index: LaneIndex,
        route: Route = None,
        position: np.ndarray = None,
        # Don't change this, since we need to make map identical to old version. get_np_random is used for traffic only.
        np_random: np.random.RandomState = None
    ) -> LaneIndex:
        """
        Get the index of the next lane that should be followed after finishing the current lane.

        - If a plan is available and matches with current lane, follow it.
        - Else, pick next road randomly.
        - If it has the same number of lanes as current road, stay in the same lane.
        - Else, pick next road's closest lane.
        :param current_index: the index of the current lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished.
        """
        assert np_random

        _from, _to, _id = current_index
        next_to = None
        # Pick next road according to planned route
        if route:
            if route[0][:2] == current_index[:2]:  # We just finished the first step of the route, drop it.
                route.pop(0)
            if route and route[0][0] == _to:  # Next road in route is starting at the end of current road.
                _, next_to, _ = route[0]
            elif route:
                logger.warning("Route {} does not start after current road {}.".format(route[0], current_index))
        # Randomly pick next road
        if not next_to:
            try:
                next_to = list(self.graph[_to].keys())[np_random.randint(len(self.graph[_to]))]
            except KeyError:
                # logger.warning("End of lane reached.")
                return current_index

        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(lanes, key=lambda l: self.get_lane((_to, next_to, l)).distance(position))

        return _to, next_to, next_id

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
        assert start != goal
        return next(self.bfs_paths(start, goal), [])

    def all_side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
        :param lane_index: the index of a lane.
        :return: all lanes belonging to the same road.
        """
        return [(lane_index[0], lane_index[1], i) for i in range(len(self.graph[lane_index[0]][lane_index[1]]))]

    def side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
        :param lane_index: the index of a lane.
        :return: indexes of lanes next to a an input lane, to its right or left.
        """
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes

    @staticmethod
    def is_same_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
        """Is lane 1 in the same road as lane 2?"""
        return lane_index_1[:2] == lane_index_2[:2] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    @staticmethod
    def is_leading_to_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
        """Is lane 1 leading to of lane 2?"""
        return lane_index_1[1] == lane_index_2[0] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    def is_connected_road(
        self,
        lane_index_1: LaneIndex,
        lane_index_2: LaneIndex,
        route: Route = None,
        same_lane: bool = False,
        depth: int = 0
    ) -> bool:
        """
        Is the lane 2 leading to a road within lane 1's route?

        Vehicles on these lanes must be considered for collisions.
        :param lane_index_1: origin lane
        :param lane_index_2: target lane
        :param route: route from origin lane, if any
        :param same_lane: compare lane id
        :param depth: search depth from lane 1 along its route
        :return: whether the roads are connected
        """
        if RoadNetwork.is_same_road(lane_index_2, lane_index_1, same_lane) \
                or RoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                # Route is starting at current road, skip it
                return self.is_connected_road(lane_index_1, lane_index_2, route[1:], same_lane, depth)
            elif route and route[0][0] == lane_index_1[1]:
                # Route is continuing from current road, follow it
                return self.is_connected_road(route[0], lane_index_2, route[1:], same_lane, depth - 1)
            else:
                # Recursively search all roads at intersection
                _from, _to, _id = lane_index_1
                return any(
                    [
                        self.is_connected_road((_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1)
                        for l1_to in self.graph.get(_to, {}).keys()
                    ]
                )
        return False

    def lanes_list(self) -> List[AbstractLane]:
        return [lane for to in self.graph.values() for ids in to.values() for lane in ids]

    def position_heading_along_route(self, route: Route, longitudinal: float, lateral: float) \
            -> Tuple[np.ndarray, float]:
        """
        Get the absolute position and heading along a route composed of several lanes at some local coordinates.

        :param route: a planned route, list of lane indexes
        :param longitudinal: longitudinal position
        :param lateral: : lateral position
        :return: position, heading
        """
        while len(route) > 1 and longitudinal > self.get_lane(route[0]).length:
            longitudinal -= self.get_lane(route[0]).length
            route = route[1:]
        return self.get_lane(route[0]).position(longitudinal,
                                                lateral), self.get_lane(route[0]).heading_theta_at(longitudinal)

    def get_roads(self, *, direction="all", lane_num=None) -> List:
        """
        Return all roads in road_network
        :param direction: "positive"/"negative"
        :param lane_num: only roads with lane_num lanes will be returned
        :return: List[Road]
        """
        assert direction in ["positive", "negative", "all"], "incorrect road direction"
        ret = []
        for _from, _to_dict in self.graph.items():
            if direction == "all" or (direction == "positive" and _from[0] != "-") or (direction == "negative"
                                                                                       and _from[0] == "-"):
                for _to, lanes in _to_dict.items():
                    if lane_num is None or len(lanes) == lane_num:
                        ret.append(Road(_from, _to))
        return ret


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
