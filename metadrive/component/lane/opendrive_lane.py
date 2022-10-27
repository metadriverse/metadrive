import math
from typing import Tuple

import numpy as np

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.utils.interpolating_line import InterpolatingLine
from metadrive.utils.opendrive_map_utils.elements.geometry import Line, Arc
from metadrive.utils.opendrive_map_utils.map_load import get_lane_id


class OpenDriveLane(AbstractLane, InterpolatingLine):
    ARC_SEGMENT_LENGTH = 1  # m
    """An OpenDrive Lane"""
    def __init__(self, width, lane_data) -> None:
        AbstractLane.__init__(self)
        self.lane_data = lane_data
        self.index = get_lane_id(lane_data)
        self._section_index = lane_data.lane_section.idx
        self.single_side = lane_data.lane_section.singleSide

        geos = self.lane_data.parentRoad.planView._geometries
        self._initialize_geometry(geos)
        self.width = width

        self.roadMark_color = lane_data.roadMark.get("color", None)
        self.roadMark_type = lane_data.roadMark.get("type", None)
        self.roadMark_material = lane_data.roadMark.get("material", None)

    def _initialize_geometry(self, geos):
        points = []
        for geo in geos:
            if isinstance(geo, Line):
                heading = geo.heading
                length = geo.length
                start = geo.start_position
                points.append(start)
                # if geo is geos[-1]:
                #     # last geo
                end = start + np.array([np.cos(heading) * length, np.sin(heading) * length])
                points.append(end)
            elif isinstance(geo, Arc):
                continue
                arc_points = self._arc_interpolate(1 / geo.curvature, geo.length, geo.start_position, geo.heading)
                points += arc_points
                if geo is not geos[-1] and len(arc_points) > 1:
                    # not last geo
                    points.pop()
            else:
                raise ValueError("Only support Line and Arc, currently")
        InterpolatingLine.__init__(self, points)

    def _arc_interpolate(self, radius, length, start_position, start_phase):
        ret = []
        arc_degree = ((radius / length) + np.pi) % np.pi * 2 - np.pi
        start_degree = start_phase
        origin = start_position - np.array([np.cos(start_degree) * radius, np.sin(start_degree) * radius])
        num_to_seg = math.floor(length / self.ARC_SEGMENT_LENGTH)
        if num_to_seg == 0:
            degree = start_degree - arc_degree
            degree = (degree + np.pi) % np.pi * 2 - np.pi
            ret.append(np.array([np.cos(degree) * radius, np.sin(degree) * radius]) + origin)
            ret.append(degree)
        # else:
        #     each_seg_degree = arc_degree / num_to_seg
        #     for i in range(num_to_seg + 1):
        #         degree = i * each_seg_degree + start_degree
        #         degree = (degree + np.pi) % np.pi * 2- np.pi
        #         ret.append(np.array([np.cos(degree) * radius, np.sin(degree) * radius]) + origin)
        return ret

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def heading_theta_at(self, longitudinal: float) -> float:
        """
        In rad
        """
        return self.get_heading_theta(longitudinal)

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return self.get_point(longitudinal, lateral)

    def local_coordinates(self, position: Tuple[float, float]):
        ret = []  # ret_longitude, ret_lateral, sort_key
        accumulate_len = 0
        for seg in self.segment_property:
            delta_x = position[0] - seg["start_point"][0]
            delta_y = position[1] - seg["start_point"][1]
            longitudinal = delta_x * seg["direction"][0] + delta_y * seg["direction"][1]
            lateral = delta_x * seg["lateral_direction"][0] + delta_y * seg["lateral_direction"][1]
            ret.append([accumulate_len + longitudinal, lateral])
            accumulate_len += seg["length"]
        ret.sort(key=lambda seg: abs(seg[-1]))
        return ret[0][0], ret[0][1]

    def construct_lane_in_block(self, block, lane_index=None):
        """
        Straight lane can be represented by one segment
        """
        middle = self.position(self.length / 2, 0)
        end = self.position(self.length, 0)
        direction_v = end - middle
        theta = -math.atan2(direction_v[1], direction_v[0])
        width = self.width_at(0) + block.SIDEWALK_LINE_DIST * 2
        self.construct_lane_segment(block, middle, width, self.length, theta, lane_index)

    def is_lane_line(self):
        return True if self.roadMark_type in [
            "solid", "broken", "broken broken", "solid solid", "solid broken", "broken solid"
        ] else False

    def destroy(self):
        self.width = None
        # waymo lane line will be processed separately
        self.start = None
        self.end = None
        InterpolatingLine.destroy(self)
