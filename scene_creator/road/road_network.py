import copy
import logging
from typing import List, Tuple, Dict, Union

import numpy as np

from scene_creator.lanes.lane import LineType, AbstractLane
from scene_creator.lanes.straight_lane import StraightLane

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

    def __add__(self, other):
        graph_1 = copy.copy(self.graph)
        graph_2 = copy.copy(other)
        set_1 = set(graph_1)
        set_2 = set(graph_2)
        if len(set_1.intersection(set_2)) == 0:
            graph_1.update(graph_2)
            new_road_network = RoadNetwork()
            new_road_network.graph = graph_1
            return new_road_network
        else:
            raise ValueError("Same start node in two road network")

    def __iadd__(self, other):
        from scene_creator.basic_utils import Decoration
        set_1 = set(self.graph) - {Decoration.start, Decoration.end}
        set_2 = set(other.graph) - {Decoration.start, Decoration.end}
        if len(set_1.intersection(set_2)) == 0:
            self.graph.update(copy.copy(other.graph))
            return self
        else:
            raise ValueError("Same start node in two road network")

    def __isub__(self, other):
        intersection = self.graph.keys() & other.graph.keys()
        if len(intersection) != 0:
            for k in intersection:
                self.graph.pop(k, None)
        return self

    def __sub__(self, other):
        ret = RoadNetwork()
        ret.graph = self.graph
        ret -= other
        return ret

    def clear(self):
        self.graph.clear()

    def get_positive_lanes(self):
        from .road import Road
        ret = []
        for _from, _to_dict in self.graph.items():
            for _to, lanes in _to_dict.items():
                road = Road(_from, _to)
                if not road.is_negative_road() and road.is_valid_road():
                    ret += lanes
        return ret

    def get_negative_lanes(self):
        from .road import Road
        ret = []
        for _from, _to_dict in self.graph:
            for _to, lanes in _to_dict:
                road = Road(_from, _to)
                if road.is_negative_road() and road.is_valid_road():
                    ret += lanes
        return ret

    def remove_road(self, road):
        from scene_creator.road.road import Road
        assert isinstance(road, Road), "Only Road Type can be deleted"
        ret = self.graph[road.start_node].pop(road.end_node)
        if len(self.graph[road.start_node]) == 0:
            self.graph.pop(road.start_node)
        return ret

    def add_road(self, road, lanes: List):
        from scene_creator.road.road import Road
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

    def build_helper(self):
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

    def update_indices(self):
        indexes = []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    indexes.append((_from, _to, _id))
        self.indices = indexes

    def get_closest_lane_index(self, position: np.ndarray) -> LaneIndex:
        """
        Get the index of the lane closest to a physx_world position.

        :param position: a physx_world position [m].
        :return: the index of the closest lane.
        """
        ret, dist = self._graph_helper.get(position)

        # if self.debug:
        #     # Old code
        #     distances = []
        #     for _from, to_dict in self.graph.items():
        #         for _to, lanes in to_dict.items():
        #             for _id, l in enumerate(lanes):
        #                 distances.append(l.distance(position))
        #     key = int(np.argmin(distances))
        #     key = self.indices[key]
        #     if ret[0] != key[0] or ret[1] != key[1] or ret[2] != key[2]:
        #         if abs(dist - min(distances)) > 1e-4:
        #             raise ValueError("ERROR! Different! ", ret, key, dist, min(distances))

        return ret

    def get_closet_lane_index_v2(self, position, current_lane):
        pass

    def next_lane(
        self,
        current_index: LaneIndex,
        route: Route = None,
        position: np.ndarray = None,
        np_random: np.random.RandomState = np.random
    ) -> LaneIndex:
        """
        Get the index of the next lane that should be followed after finishing the current lane.

        - If a plan is available and matches with current lane, follow it.
        - Else, pick next road randomly.
        - If it has the same number of lanes as current road, stay in the same lane.
        - Else, pick next road's closest lane.
        :param current_index: the index of the current lane.
        :param route: the planned route, if any.
        :param position: the chrono_vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished.
        """
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

    @staticmethod
    def straight_road_network(lanes: int = 4, length: float = 10000, angle: float = 0) -> 'RoadNetwork':
        net = RoadNetwork()
        for lane in range(lanes):
            origin = np.array([0, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            origin = rotation @ origin
            end = rotation @ end
            line_types = (
                LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE
            )
            net.add_lane("0", "1", StraightLane(origin, end, line_types=line_types))
        return net

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
        return self.get_lane(route[0]).position(longitudinal, lateral), self.get_lane(route[0]).heading_at(longitudinal)

    def get_map(self):
        pass


class GraphLookupTable:
    def __init__(self, graph, debug):
        self.graph = graph
        self.debug = debug

    def get(self, position):
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
                if lanes_id == "decoration_":
                    continue
                if lane_id == 0:
                    dist = first_lane_distance
                else:
                    dist = lane.distance(position)
                distance_index_mapping.append((dist, (section_id, lanes_id, lane_id)))
            if rank > 4:
                # Take first rank 5 lanes into consideration. The number may related to the number of
                # lanes in intersection. We have 3 lanes in intersection, so computing the first 4 ranks can make
                # thing work. We choose take first 5 lanes into consideration.
                # In futurem we shall refactor the whole system, so this vulnerable code would be removed.
                break
        if self.graph.get("decoration", False):
            for id, lane in enumerate(self.graph["decoration"]["decoration_"]):
                dist = lane.distance(position)
                distance_index_mapping.append((dist, ("decoration", "decoration_", id)))

        ret_ind = np.argmin([d for d, _ in distance_index_mapping])
        index = distance_index_mapping[ret_ind][1]
        distance = distance_index_mapping[ret_ind][0]
        return index, distance
