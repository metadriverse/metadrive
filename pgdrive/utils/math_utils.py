import math
import time
from typing import Tuple
import numpy as np


def safe_clip(array, min_val, max_val):
    array = np.nan_to_num(array)
    return np.clip(array, min_val, max_val)


def wrap_to_pi(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def get_vertical_vector(vector: np.array):
    length = norm(vector[0], vector[1])
    return (vector[1] / length, -vector[0] / length), (-vector[1] / length, vector[0] / length)


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


def get_points_bounding_box(line_points):
    """
    Get bounding box from several points
    :param line_points: Key points on lines
    :return: bounding box
    """
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


def get_boxes_bounding_box(boxes):
    """
    Get a max bounding box from sveral small bound boxes
    :param boxes: List of other bounding box
    :return: Max bounding box
    """
    res_x_max = -np.inf
    res_x_min = np.inf
    res_y_min = np.inf
    res_y_max = -np.inf
    for x_max, x_min, y_max, y_min in boxes:
        res_x_max = max(res_x_max, x_max)
        res_x_min = min(res_x_min, x_min)
        res_y_max = max(res_y_max, y_max)
        res_y_min = min(res_y_min, y_min)
    return res_x_max, res_x_min, res_y_max, res_y_min
