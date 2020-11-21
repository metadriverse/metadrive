import math
import time
import numpy as np
from scene_creator.road.road_network import RoadNetwork
from typing import Tuple


def wrap_to_pi(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def get_vertical_vector(vector: np.array):
    # len = np.linalg.norm(vector)
    # return np.array([vector[1], -vector[0]]) / len, np.array([-vector[1], vector[0]]) / len

    length = norm(vector[0], vector[1])
    return (vector[1] / length, -vector[0] / length), (-vector[1] / length, vector[0] / length)


def check_lane_on_road(road_network: RoadNetwork, lane, positive: float = 0, ignored=None) -> bool:
    """
    Calculate if the new lane intersects with other lanes in current road network
    The return Value is True when cross !!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    graph = road_network.graph
    for _from, to_dict in graph.items():
        for _to, lanes in to_dict.items():
            if ignored and (_from, _to) == ignored:
                continue
            x_max_1, x_min_1, y_max_1, y_min_1 = get_road_bound_box(lanes)
            x_max_2, x_min_2, y_max_2, y_min_2 = get_road_bound_box([lane])
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


def get_road_bound_box(lanes):
    from scene_creator.lanes.circular_lane import CircularLane
    from scene_creator.lanes.straight_lane import StraightLane
    if isinstance(lanes[0], StraightLane):
        lane_1 = lanes[0]
        line_1_start = lane_1.position(0.1, -lane_1.width / 2.0)
        line_1_end = lane_1.position(lane_1.length - 0.1, -lane_1.width / 2.0)
        lane_last = lanes[-1]
        line_2_start = lane_last.position(0.1, lane_1.width / 2.0)
        line_2_end = lane_last.position(lane_last.length - 0.1, lane_1.width / 2.0)
        x_max = max(line_1_start[0], line_1_end[0], line_2_start[0], line_2_end[0])
        x_min = min(line_1_start[0], line_1_end[0], line_2_start[0], line_2_end[0])
        y_max = max(line_1_start[1], line_1_end[1], line_2_start[1], line_2_end[1])
        y_min = min(line_1_start[1], line_1_end[1], line_2_start[1], line_2_end[1])
        return x_max, x_min, y_max, y_min
    elif isinstance(lanes[0], CircularLane):
        line_points = get_arc_bound_box_points(lanes[0], -1)
        line_points += get_arc_bound_box_points(lanes[-1], 1)
        x_max = -np.inf
        x_min = np.inf
        y_max = -np.inf
        y_min = np.inf
        for p in line_points:
            x_max = max(x_max, p[0])
            x_min = min(x_min, p[0])
            y_max = max(y_max, p[1])
            y_min = min(y_min, p[1])
        return x_max, x_min, y_max, y_min

    else:
        raise ValueError("not lane type, can not calculate rectangular")


def get_arc_bound_box_points(lane, lateral_dir):
    pi_2 = (np.pi / 2.0)
    points = [lane.position(0.1, lateral_dir * lane.width), lane.position(lane.length - 0.1, lateral_dir * lane.width)]
    start_phase = (lane.start_phase // pi_2) * pi_2
    start_phase += pi_2 if lane.direction == 1 else 0
    for phi_index in range(4):
        phi = start_phase + phi_index * pi_2 * lane.direction
        if lane.direction * phi > lane.direction * lane.end_phase:
            break
        point = lane.center + (lane.radius - lateral_dir * lane.width *
                               lane.direction) * np.array([math.cos(phi), math.sin(phi)])
        points.append(point)
    return points


def get_all_lanes(roadnet: RoadNetwork):
    graph = roadnet.graph
    res = []
    for from_, to_dict in graph.items():
        for _to, lanes in to_dict.items():
            for l in lanes:
                res.append(l)
    return res


def time_me(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        fn(*args, **kwargs)
        print("%s cost %s second" % (fn.__name__, time.clock() - start))

    return _wrapper


def norm(x, y):
    return math.sqrt(x**2 + y**2)


def clip(a, low, high):
    return min(max(a, low), high)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def do_every(duration: float, timer: float) -> bool:
    return duration < timer


def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x > 0:
        return eps
    else:
        return -eps


def rotated_rectangles_intersect(rect1: Tuple, rect2: Tuple) -> bool:
    """
    Do two rotated rectangles intersect?

    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    :return: do they?
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def point_in_rectangle(point, rect_min, rect_max) -> bool:
    """
    Check if a point is inside a rectangle

    :param point: a point (x, y)
    :param rect_min: x_min, y_min
    :param rect_max: x_max, y_max
    """
    return rect_min[0] <= point[0] <= rect_max[0] and rect_min[1] <= point[1] <= rect_max[1]


def point_in_rotated_rectangle(point: np.ndarray, center: np.ndarray, length: float, width: float, angle: float) \
        -> bool:
    """
    Check if a point is inside a rotated rectangle

    :param point: a point
    :param center: rectangle center
    :param length: rectangle length
    :param width: rectangle width
    :param angle: rectangle angle [rad]
    :return: is the point inside the rectangle
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_in_rectangle(ru, (-length / 2, -width / 2), (length / 2, width / 2))


def has_corner_inside(rect1: Tuple, rect2: Tuple) -> bool:
    """
    Check if rect1 has a corner inside rect2

    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    (c1, l1, w1, a1) = rect1
    (c2, l2, w2, a2) = rect2
    c1 = np.array(c1)
    l1v = np.array([l1 / 2, 0])
    w1v = np.array([0, w1 / 2])
    r1_points = np.array([[0, 0], -l1v, l1v, -w1v, w1v, -l1v - w1v, -l1v + w1v, +l1v - w1v, +l1v + w1v])
    c, s = np.cos(a1), np.sin(a1)
    r = np.array([[c, -s], [s, c]])
    rotated_r1_points = r.dot(r1_points.transpose()).transpose()
    return any([point_in_rotated_rectangle(c1 + np.squeeze(p), c2, l2, w2, a2) for p in rotated_r1_points])
