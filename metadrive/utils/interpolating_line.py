import math

import numpy as np

from metadrive.utils.math import norm, get_vertical_vector


class InterpolatingLine:
    """
    This class provides point set with interpolating function
    """

    def __init__(self, points):
        points = np.asarray(points)[..., :2]
        self.segment_property, self._start_points, self._end_points = self._get_properties(points)
        self._distance_b_a = self._end_points - self._start_points
        self.length = sum([seg["length"] for seg in self.segment_property])

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return self.get_point(longitudinal, lateral)

    def local_coordinates(self, position, only_in_lane_point=False):
        """
        Finding the local coordinate of a given point when projected to this interpolating line.
        We will iterate over all segments, from start to end, and compute the relative position of the point
        w.r.t. the start point of the segment.
        Since each line segment is straight, the relative position of the position to the start point is exactly
        the local coordination.
        If the point is not fall into this line segment, then we will increment the "longitudinal position" by the
        length of this line segment, and then computing next line segment.

        Here we provide two implementations.

        Option 1: Iterate over all segments, and when finding the longitudinal position of the point w.r.t. the
        start point of the segment is negative, stop iteration and return current "accumulated longitudinal position"
        and lateral position.
        This option assumes the longitudinal position is accumulating increasingly when iterating over segments.
        Option 1 might be problematic in the case of extremely curved road.
        (PZH: But I can't image such case and why it fails option 1.)

        Option 2: Iterate over all segments and never stop the iteration until all segments are visited. Compute the
        "accumulated longitudinal position" and lateral position of the point in each segment when assuming the point
        is exactly falling into the segment. Then, search all reported coordinates in all segments, find the one
        with minimal lateral position, and report the value.
        This option is problematic as raised by PZH that the return value might be wrong.
        Let's say there is a correct segment A where the position is falling into, and there exists another segment
        B where the lateral position of the point in segment B is 0, but segment B is far away from A in longitudinal.
        In this case, the option 2 algorithm will return segment B because it has minimal lateral, but it is wrong.

        We will use Option 1.
        """
        target_segment_idx = self.min_lineseg_index(position, self._start_points, self._end_points, self._distance_b_a)

        long = 0
        for idx, seg in enumerate(self.segment_property):
            if idx != target_segment_idx:
                long += seg["length"]
            else:
                delta_x = position[0] - seg["start_point"][0]
                delta_y = position[1] - seg["start_point"][1]
                long += delta_x * seg["direction"][0] + delta_y * seg["direction"][1]
                lateral = delta_x * seg["lateral_direction"][0] + delta_y * seg["lateral_direction"][1]
                return long, lateral

        # deprecated content
        # Four elements:
        #   accumulated longitude,
        #   segment-related longitude,
        #   segment-related lateral,
        # ret = []
        # exclude_ret = []
        # accumulate_len = 0
        #
        # # _debug = []
        #
        # for seg in self.segment_property:
        #     delta_x = position[0] - seg["start_point"][0]
        #     delta_y = position[1] - seg["start_point"][1]
        #     longitudinal = delta_x * seg["direction"][0] + delta_y * seg["direction"][1]
        #     lateral = delta_x * seg["lateral_direction"][0] + delta_y * seg["lateral_direction"][1]
        #     # _debug.append(longitudinal)
        #
        #     if longitudinal < 0.0:
        #         dist_square = norm(delta_x, delta_y)
        #         if dist_square < seg["length"] * 2:
        #             current_long = accumulate_len + longitudinal
        #             current_lat = lateral
        #             return current_long, current_lat
        #
        #     if not only_in_lane_point:
        #         ret.append([accumulate_len + longitudinal, longitudinal, lateral])
        #     else:
        #         if abs(lateral) <= self.width / 2 and -1. <= accumulate_len + longitudinal <= self.length + 1:
        #             ret.append([accumulate_len + longitudinal, longitudinal, lateral])
        #         else:
        #             exclude_ret.append([accumulate_len + longitudinal, longitudinal, lateral])
        #     accumulate_len += seg["length"]
        # if len(ret) == 0:
        #     # for corner case
        #     ret = exclude_ret
        # ret.sort(key=lambda seg: abs(seg[-1]))
        # return ret[0][0], ret[0][-1]

    def _get_properties(self, points):
        points = np.asarray(points)[..., :2]
        ret = []
        start_points = []
        end_points = []
        p_start_idx = 0
        while p_start_idx < len(points) - 1:
            for p_end_idx in range(p_start_idx + 1, len(points)):
                if np.linalg.norm(points[p_start_idx] - points[p_end_idx]) > 1:
                    break
            p_start = points[p_start_idx]
            p_end = points[p_end_idx]

            if np.linalg.norm(p_start - p_end) < 1e-6:
                p_start_idx = p_end_idx  # next
                continue

            seg_property = {
                "length": self.points_distance(p_start, p_end),
                "direction": np.asarray(self.points_direction(p_start, p_end)),
                "lateral_direction": np.asarray(self.points_lateral_direction(p_start, p_end)),
                "heading": self.points_heading(p_start, p_end),
                "start_point": p_start,
                "end_point": p_end
            }
            ret.append(seg_property)
            start_points.append(seg_property["start_point"])
            end_points.append(seg_property["end_point"])
            p_start_idx = p_end_idx  # next
        if len(ret) == 0:
            # static, length=zero
            seg_property = {
                "length": 0.1,
                "direction": np.asarray((1, 0)),
                "lateral_direction": np.asarray((0, 1)),
                "heading": 0,
                "start_point": points[0],
                "end_point": np.asarray([points[0][0] + 0.1, points[0][1]])
            }
            ret.append(seg_property)
            start_points.append(seg_property["start_point"])
            end_points.append(seg_property["end_point"])
        return ret, np.asarray(start_points), np.asarray(end_points)

    @staticmethod
    def points_distance(start_p, end_p):
        return norm((end_p - start_p)[0], (end_p - start_p)[1])

    @staticmethod
    def points_direction(start_p, end_p):
        return (end_p - start_p) / norm((end_p - start_p)[0], (end_p - start_p)[1])

    @staticmethod
    def points_lateral_direction(start_p, end_p):
        # direction = (end_p - start_p) / norm((end_p - start_p)[0], (end_p - start_p)[1])
        return np.asarray(get_vertical_vector(end_p - start_p)[1])

    @staticmethod
    def points_heading(start_p, end_p):
        return math.atan2(end_p[1] - start_p[1], end_p[0] - start_p[0])

    def get_point(self, longitudinal, lateral=None):
        """
        Get point on this line by interpolating
        """
        accumulate_len = 0
        for seg in self.segment_property:
            accumulate_len += seg["length"]
            if accumulate_len + 0.1 >= longitudinal:
                break
        if lateral is not None:
            return (seg["start_point"] + (longitudinal - accumulate_len + seg["length"]) *
                    seg["direction"]) + lateral * seg["lateral_direction"]
        else:
            return seg["start_point"] + (longitudinal - accumulate_len + seg["length"]) * seg["direction"]

    def get_heading_theta(self, longitudinal: float) -> float:
        """
        In rad
        """
        accumulate_len = 0

        assert len(self.segment_property) > 0

        for seg in self.segment_property:
            accumulate_len += seg["length"]
            if accumulate_len > longitudinal:
                return seg["heading"]

        return seg["heading"]

    def segment(self, longitudinal: float):
        """
        Return the segment piece on this lane of current position
        """
        accumulate_len = 0
        for index, seg in enumerate(self.segment_property):
            accumulate_len += seg["length"]
            if accumulate_len + 0.1 >= longitudinal:
                return self.segment_property[index]
        return self.segment_property[index]

    def lateral_direction(self, longitude):
        lane_segment = self.segment(longitude)
        lateral = lane_segment["lateral_direction"]
        return lateral

    def destroy(self):
        del self.segment_property
        self.segment_property = []
        self.length = None

    @staticmethod
    def min_lineseg_index(p, a, b, d_ba=None):
        """Cartesian distance from point to line segment
        Edited to support arguments as series, from:
        https://stackoverflow.com/a/54442561/11208892

        Args:
            - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
            - a: np.array of shape (x, 2)
            - b: np.array of shape (x, 2)
        """
        # normalized tangent vectors
        p = np.asarray(p)
        if d_ba is None:
            d_ba = b - a
        d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                             .reshape(-1, 1)))

        # signed parallel distance components
        # rowwise dot products of 2D vectors
        s = np.multiply(a - p, d).sum(axis=1)
        t = np.multiply(p - b, d).sum(axis=1)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros(len(s))])

        # perpendicular distance component
        # rowwise cross products of 2D vectors
        d_pa = p - a
        c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
        min_dists = np.hypot(h, c)

        # get index
        return np.argmin(min_dists)
