import math
from typing import Tuple, Union

import numpy as np

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.constants import LineType
from metadrive.utils import norm


class WayPointLane(AbstractLane):
    """
    CenterLineLane is created by giving the center line points array or way points array.
    By using this lane type, map can be constructed from Waymo/Argoverse/OpenstreetMap dataset
    """
    def __init__(
        self,
        center_line_points: Union[list, np.ndarray],
        width: float,
        forbidden: bool = False,
        speed_limit: float = 1000,
        priority: int = 0
    ):
        super(WayPointLane, self).__init__()
        self.set_speed_limit(speed_limit)
        self.width = width
        self.forbidden = forbidden
        self.priority = priority
        self.line_types = (LineType.BROKEN, LineType.BROKEN)
        self.center_line_points = center_line_points

        # Segment is the part between two adjacent way points
        self.segment_property = self._get_properties()
        self.length = sum([seg["length"] for seg in self.segment_property])

    def _get_properties(self):
        ret = []
        for idx, p_start in enumerate(self.center_line_points[:-1]):
            p_end = self.center_line_points[idx + 1]
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

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def heading_theta_at(self, longitudinal: float) -> float:
        accumulate_len = 0
        for seg in self.segment_property:
            accumulate_len += seg["length"]
            if accumulate_len > longitudinal:
                return seg["heading"]

        return seg["heading"]

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        accumulate_len = 0
        for seg in self.segment_property:
            if accumulate_len + 0.1 >= longitudinal:
                return seg["start_point"] + (accumulate_len -
                                             longitudinal) * seg["direction"] + lateral * seg["lateral_direction"]
            accumulate_len += seg["length"]

        return seg["start_point"] + (longitudinal - accumulate_len +
                                     seg["length"]) * seg["direction"] + lateral * seg["lateral_direction"]

    def local_coordinates(self, position: Tuple[float, float]):
        ret = []  # ret_longitude, ret_lateral, sort_key
        accumulate_len = 0
        for seg in self.segment_property:
            delta_x = position[0] - seg["start_point"][0]
            delta_y = position[1] - seg["start_point"][1]
            longitudinal = delta_x * seg["direction"][0] + delta_y * seg["direction"][1]
            lateral = delta_x * seg["lateral_direction"][0] + delta_y * seg["lateral_direction"][1]
            ret.append([accumulate_len + longitudinal, lateral, longitudinal + lateral])
            accumulate_len += seg["length"]
        ret.sort(key=lambda seg: seg[-1])
        return ret[0][0], ret[0][1]

    def segment(self, longitudinal: float):
        """
        Return the segment piece on this lane of current position
        """
        accumulate_len = 0
        for index, seg in enumerate(self.segment_property):
            if accumulate_len + 0.1 >= longitudinal:
                return self.segment_property[index]
        return self.segment_property[index]
