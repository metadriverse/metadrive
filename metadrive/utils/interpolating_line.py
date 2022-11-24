import math

import numpy as np

from metadrive.utils.math_utils import norm


class InterpolatingLine:
    """
    This class provides point set with interpolating function
    """
    def __init__(self, points):
        self.segment_property = self._get_properties(points)
        self.length = sum([seg["length"] for seg in self.segment_property])

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return self.get_point(longitudinal, lateral)

    def local_coordinates(self, position, only_in_lane_point=False):
        ret = []  # ret_longitude, ret_lateral, sort_key
        exclude_ret = []
        accumulate_len = 0
        for seg in self.segment_property:
            delta_x = position[0] - seg["start_point"][0]
            delta_y = position[1] - seg["start_point"][1]
            longitudinal = delta_x * seg["direction"][0] + delta_y * seg["direction"][1]
            lateral = delta_x * seg["lateral_direction"][0] + delta_y * seg["lateral_direction"][1]
            if not only_in_lane_point:
                ret.append([accumulate_len + longitudinal, lateral])
            else:
                if abs(lateral) <= self.width / 2 and -1. <= accumulate_len + longitudinal <= self.length + 1:
                    ret.append([accumulate_len + longitudinal, lateral])
                else:
                    exclude_ret.append([accumulate_len + longitudinal, lateral])
            accumulate_len += seg["length"]
        if len(ret) == 0:
            # for corner case
            ret = exclude_ret
        ret.sort(key=lambda seg: abs(seg[-1]))
        return ret[0][0], ret[0][1]

    def _get_properties(self, points):
        ret = []
        for idx, p_start in enumerate(points[:-1]):
            p_end = points[idx + 1]
            seg_property = {
                "length": self.points_distance(p_start, p_end),
                "direction": self.points_direction(p_start, p_end),
                "lateral_direction": self.points_lateral_direction(p_start, p_end),
                "heading": self.points_heading(p_start, p_end),
                "start_point": p_start,
                "end_point": p_end
            }
            ret.append(seg_property)
        return ret

    @staticmethod
    def points_distance(start_p, end_p):
        return norm((end_p - start_p)[0], (end_p - start_p)[1])

    @staticmethod
    def points_direction(start_p, end_p):
        return (end_p - start_p) / norm((end_p - start_p)[0], (end_p - start_p)[1])

    @staticmethod
    def points_lateral_direction(start_p, end_p):
        direction = (end_p - start_p) / norm((end_p - start_p)[0], (end_p - start_p)[1])
        return np.array([-direction[1], direction[0]])

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
        self.segment_property = None
        self.length = None
