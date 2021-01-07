import math
from typing import List, TYPE_CHECKING, Tuple

import numpy as np

from pgdrive.scene_creator.lanes.circular_lane import CircularLane
from pgdrive.scene_creator.lanes.lane import AbstractLane
from pgdrive.utils.math_utils import get_points_bounding_box

if TYPE_CHECKING:
    from pgdrive.scene_creator.blocks.block import BlockSocket
    from pgdrive.scene_creator.road.road import Road
    from pgdrive.scene_creator.road.road_network import RoadNetwork


class Decoration:
    start = "decoration"
    end = "decoration_"


class Goal:
    """
    Goal at intersection
    The keywords 0, 1, 2 should be reserved, and only be used in roundabout and intersection
    """

    RIGHT = 0
    STRAIGHT = 1
    LEFT = 2
    ADVERSE = 3  # Useless now


def get_lanes_on_road(road: "Road", roadnet: "RoadNetwork") -> List["AbstractLane"]:
    return roadnet.graph[road.start_node][road.end_node]


def block_socket_merge(
    socket_1: "BlockSocket", socket_2: "BlockSocket", global_network: "RoadNetwork", positive_merge: False
):
    global_network.graph[socket_1.positive_road.start_node][socket_2.negative_road.start_node] = \
        global_network.graph[socket_1.positive_road.start_node].pop(socket_1.positive_road.end_node)

    global_network.graph[socket_2.positive_road.start_node][socket_1.negative_road.start_node] = \
        global_network.graph[socket_2.positive_road.start_node].pop(socket_2.positive_road.end_node)


def check_lane_on_road(road_network: "RoadNetwork", lane, positive: float = 0, ignored=None) -> bool:
    """
    Calculate if the new lane intersects with other lanes in current road network
    The return Value is True when cross !!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    graph = road_network.graph
    for _from, to_dict in graph.items():
        for _to, lanes in to_dict.items():
            if ignored and (_from, _to) == ignored:
                continue
            if len(lanes) == 0:
                continue
            x_max_1, x_min_1, y_max_1, y_min_1 = get_road_bounding_box(lanes)
            x_max_2, x_min_2, y_max_2, y_min_2 = get_road_bounding_box([lane])
            if x_min_1 > x_max_2 or x_min_2 > x_max_1 or y_min_1 > y_max_2 or y_min_2 > y_max_1:
                continue
            for _id, l in enumerate(lanes):
                for i in range(1, int(lane.length), 1):
                    sample_point = lane.position(i, positive * lane.width_at(i) / 2.0)
                    longitudinal, lateral = l.local_coordinates(sample_point)
                    is_on = math.fabs(lateral) <= l.width_at(longitudinal) / 2.0 and 0 <= longitudinal <= l.length
                    if is_on:
                        return True
    return False


def get_road_bounding_box(lanes, extra_lateral=3) -> Tuple:
    """
    Return (x_max, x_min, y_max, y_min) as bounding box of this road
    :param lanes: Lanes in this road
    :param extra_lateral: extra width in lateral direction, usually side_walk width
    :return: x_max, x_min, y_max, y_min
    """
    line_points = get_curve_contour(lanes, extra_lateral) if isinstance(lanes[0], CircularLane) \
        else get_straight_contour(lanes, extra_lateral)
    return get_points_bounding_box(line_points)


def get_straight_contour(lanes, extra_lateral) -> List:
    """
    Get several points as bounding box of this road
    :param lanes: lanes contained in road
    :param extra_lateral: extra length in lateral direction, usually side_walk
    :return: points
    :param lanes:
    :return:
    """
    ret = []
    for lane, dir in [(lanes[0], -1), (lanes[-1], 1)]:
        ret.append(lane.position(0.1, dir * (lane.width / 2.0 + extra_lateral)))
        ret.append(lane.position(lane.length - 0.1, dir * (lane.width / 2.0 + extra_lateral)))
    return ret


def get_curve_contour(lanes, extra_lateral) -> List:
    """
    Get several points as bounding box of this road
    :param lanes: lanes contained in road
    :param extra_lateral: extra length in lateral direction, usually side_walk
    :return: points
    """
    points = []
    for lane, lateral_dir in [(lanes[0], -1), (lanes[-1], 1)]:
        pi_2 = (np.pi / 2.0)
        points += [
            lane.position(0.1, lateral_dir * (lane.width / 2.0 + extra_lateral)),
            lane.position(lane.length - 0.1, lateral_dir * (lane.width / 2.0 + extra_lateral))
        ]
        start_phase = (lane.start_phase // pi_2) * pi_2
        start_phase += pi_2 if lane.direction == 1 else 0
        for phi_index in range(4):
            phi = start_phase + phi_index * pi_2 * lane.direction
            if lane.direction * phi > lane.direction * lane.end_phase:
                break
            point = lane.center + (lane.radius - lateral_dir * (lane.width / 2.0 + extra_lateral) *
                                   lane.direction) * np.array([math.cos(phi), math.sin(phi)])
            points.append(point)
    return points


def get_all_lanes(roadnet: "RoadNetwork"):
    graph = roadnet.graph
    res = []
    for from_, to_dict in graph.items():
        for _to, lanes in to_dict.items():
            for l in lanes:
                res.append(l)
    return res
