import logging
from typing import Tuple

import math
import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.utils.opendrive.elements.geometry import Line, Arc
from metadrive.utils.opendrive.map_load import get_lane_id


class OpenDriveLane(PointLane):
    ARC_SEGMENT_LENGTH = 1  # m
    """An OpenDrive Lane"""
    def __init__(self, width, lane_data) -> None:
        self.lane_data = lane_data
        self.width = width
        geos = self.lane_data.parentRoad.planView._geometries
        self._initialize_geometry(geos)
        self.single_side = lane_data.lane_section.singleSide

        self._section_index = lane_data.lane_section.idx
        self.index = get_lane_id(lane_data)
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
        super(OpenDriveLane, self).__init__(points, self.width)

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

    def is_lane_line(self):
        return True if self.roadMark_type in [
            "solid", "broken", "broken broken", "solid solid", "solid broken", "broken solid"
        ] else False

    def destroy(self):
        self.width = None
        # lane line will be processed separately
        self.start = None
        self.end = None
        super(OpenDriveLane, self).destroy()
