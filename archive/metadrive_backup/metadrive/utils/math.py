import math
from typing import Tuple

import numpy as np

number_pos_inf = float("inf")
number_neg_inf = float("-inf")


def safe_clip(array, min_val, max_val):
    array = np.nan_to_num(array.astype(np.float64), copy=False, nan=0.0, posinf=max_val, neginf=min_val)
    return np.clip(array, min_val, max_val).astype(np.float64)


def safe_clip_for_small_array(array, min_val, max_val):
    array = list(array)
    for i in range(len(array)):
        if math.isnan(array[i]):
            array[i] = 0.0
        elif array[i] == number_pos_inf:
            array[i] = max_val
        elif array[i] == number_neg_inf:
            array[i] = min_val
        array[i] = clip(array[i], min_val, max_val)
    return array


def wrap_to_pi(x: float) -> float:
    """Wrap the input radian to (-pi, pi]. Note that -pi is exclusive and +pi is inclusive.

    Args:
        x (float): radian.

    Returns:
        The radian in range (-pi, pi].
    """
    angles = x
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def get_vertical_vector(vector: np.array):
    length = norm(vector[0], vector[1])
    # return (vector[1] / length, -vector[0] / length), (-vector[1] / length, vector[0] / length)
    return (-vector[1] / length, vector[0] / length), (vector[1] / length, -vector[0] / length)


def norm(x, y):
    return math.sqrt(x**2 + y**2)


def clip(a, low, high):
    return min(max(a, low), high)


def point_distance(x, y):
    return norm(x[0] - y[0], x[1] - y[1])


def panda_vector(position_x, position_y, z=0.0):
    return position_x, position_y, z


def distance_greater(vec1, vec2, length):
    """Return whether the distance between two vectors is greater than the given length."""
    return ((vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2) > length**2


def mph_to_kmh(speed_in_mph: float):
    speed_in_kmh = speed_in_mph * 1.609344
    return speed_in_kmh


def get_laser_end(lidar_range, perceive_distance, laser_index, heading_theta, vehicle_position_x, vehicle_position_y):
    angle = lidar_range[laser_index] + heading_theta
    return (
        perceive_distance * math.cos(angle) + vehicle_position_x,
        perceive_distance * math.sin(angle) + vehicle_position_y
    )


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


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
    c, s = math.cos(angle), math.sin(angle)
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
    c, s = math.cos(a1), math.sin(a1)
    r = np.array([[c, -s], [s, c]])
    rotated_r1_points = r.dot(r1_points.transpose()).transpose()
    return any([point_in_rotated_rectangle(c1 + np.squeeze(p), c2, l2, w2, a2) for p in rotated_r1_points])


def get_points_bounding_box(line_points):
    """
    Get bounding box from several points
    :param line_points: Key points on lines
    :return: bounding box
    """
    new_line_points = np.array(line_points)
    new_x_max = new_line_points[:, 0].max()
    new_x_min = new_line_points[:, 0].min()
    new_y_max = new_line_points[:, 1].max()
    new_y_min = new_line_points[:, 1].min()
    return new_x_max, new_x_min, new_y_max, new_y_min


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


class Vector(tuple):
    """
    Avoid using this data structure!
    """
    def __sub__(self, other):
        return Vector((self[0] - other[0], self[1] - other[1]))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, float) or np.isscalar(other):
            return Vector((self[0] * other, self[1] * other))
        else:
            return Vector((self[0] * other[0], self[1] * other[1]))

    def __add__(self, other):
        if isinstance(other, float) or np.isscalar(other):
            return Vector((self[0] + other, self[1] + other))
        else:
            return Vector((self[0] + other[0], self[1] + other[1]))

    def __truediv__(self, other):
        if isinstance(other, float) or np.isscalar(other):
            ret = Vector((self[0] / other, self[1] / other))
            return ret
        raise ValueError()

    def tolist(self):
        return list(self)

    def __rsub__(self, other):
        return Vector(other) - self

    def __neg__(self):
        return Vector((-self[0], -self[1]))

    def dot(self, other):
        return self[0] * other[0] + self[1] * other[1]


def compute_angular_velocity(initial_heading, final_heading, dt):
    """
    Calculate the angular velocity between two headings given in radians.

    Parameters:
    initial_heading (float): The initial heading in radians.
    final_heading (float): The final heading in radians.
    dt (float): The time interval between the two headings in seconds.

    Returns:
    float: The angular velocity in radians per second.
    """

    # Calculate the difference in headings
    delta_heading = final_heading - initial_heading

    # Adjust the delta_heading to be in the range (-π, π]
    delta_heading = (delta_heading + math.pi) % (2 * math.pi) - math.pi

    # Compute the angular velocity
    angular_vel = delta_heading / dt

    return angular_vel


def get_polyline_length(points_array):
    diff = np.diff(points_array, axis=0)
    squared_diff = diff**2
    squared_diff_sum = np.sum(squared_diff, axis=1)
    distances = np.sqrt(squared_diff_sum)
    return np.sum(distances)


def resample_polyline(points, target_distance):
    # Calculate the cumulative distance along the original polyline
    distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0., axis=0)

    # Create a linearly spaced array of distances for the resampled polyline
    resampled_distances = np.arange(0, distances[-1], target_distance)

    # Interpolate the points along the resampled distances
    resampled_points = np.empty((len(resampled_distances), points.shape[1]))
    for i in range(points.shape[1]):
        resampled_points[:, i] = np.interp(resampled_distances, distances, points[:, i])

    return resampled_points
