import math
from metadrive.type import MetaDriveType
from shapely import geometry

from typing import Tuple, Union

import numpy as np

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.constants import PGDrivableAreaProperty
from metadrive.constants import PGLineType
from metadrive.utils.interpolating_line import InterpolatingLine
from metadrive.utils.math import get_points_bounding_box
from metadrive.utils.math import wrap_to_pi


class PointLane(AbstractLane, InterpolatingLine):
    """
    CenterLineLane is created by giving the center line points array or way points array.
    By using this lane type, map can be constructed from Waymo/nuPlan/OpenstreetMap dataset
    """
    VIS_LANE_WIDTH = 6.5
    POLYGON_SAMPLE_RATE = 1

    def __init__(
        self,
        center_line_points: Union[list, np.ndarray],
        width: float,
        polygon=None,
        forbidden: bool = False,
        speed_limit: float = 1000,
        priority: int = 0,
        need_lane_localization=True,
        auto_generate_polygon=True,
        metadrive_type=MetaDriveType.LANE_SURFACE_STREET
    ):
        center_line_points = np.array(center_line_points)[..., :2]
        AbstractLane.__init__(self, metadrive_type)
        InterpolatingLine.__init__(self, center_line_points)
        self._bounding_box = get_points_bounding_box(center_line_points)
        self._polygon = polygon
        self.width = width if width else self.VIS_LANE_WIDTH
        if self._polygon is None and auto_generate_polygon:
            self._polygon = self.auto_generate_polygon()
        self.need_lane_localization = need_lane_localization
        self.set_speed_limit(speed_limit)
        self.forbidden = forbidden
        self.priority = priority
        # waymo lane line will be processed separately
        self.line_types = (PGLineType.NONE, PGLineType.NONE)
        self.is_straight = True if abs(self.heading_theta_at(0.1) -
                                       self.heading_theta_at(self.length - 0.1)) < np.deg2rad(10) else False
        self.start = self.position(0, 0)
        assert np.linalg.norm(self.start - center_line_points[0]) < 0.1, "Start point error!"
        self.end = self.position(self.length, 0)
        assert np.linalg.norm(self.end - center_line_points[-1]) < 1, "End point error!"

    def auto_generate_polygon(self):
        start_heading = self.heading_theta_at(0)
        start_dir = [math.cos(start_heading), math.sin(start_heading)]

        end_heading = self.heading_theta_at(self.length)
        end_dir = [math.cos(end_heading), math.sin(end_heading)]
        polygon = []
        longs = np.arange(0, self.length + self.POLYGON_SAMPLE_RATE, self.POLYGON_SAMPLE_RATE)
        for k in range(2):
            if k == 1:
                longs = longs[::-1]
            for t, longitude, in enumerate(longs):
                lateral = self.width_at(longitude) / 2
                lateral *= -1 if k == 0 else 1
                point = self.position(longitude, lateral)
                if (t == 0 and k == 0) or (t == len(longs) - 1 and k == 1):
                    # control the adding sequence
                    if k == 1:
                        # last point
                        polygon.append([point[0], point[1]])

                    # extend
                    polygon.append(
                        [
                            point[0] - start_dir[0] * self.POLYGON_SAMPLE_RATE,
                            point[1] - start_dir[1] * self.POLYGON_SAMPLE_RATE
                        ]
                    )

                    if k == 0:
                        # first point
                        polygon.append([point[0], point[1]])
                elif (t == 0 and k == 1) or (t == len(longs) - 1 and k == 0):

                    if k == 0:
                        # second point
                        polygon.append([point[0], point[1]])

                    polygon.append(
                        [
                            point[0] + end_dir[0] * self.POLYGON_SAMPLE_RATE,
                            point[1] + end_dir[1] * self.POLYGON_SAMPLE_RATE
                        ]
                    )

                    if k == 1:
                        # third point
                        polygon.append([point[0], point[1]])
                else:
                    polygon.append([point[0], point[1]])
        return np.asarray(polygon)

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def heading_theta_at(self, longitudinal: float) -> float:
        """
        In rad
        """
        return self.get_heading_theta(longitudinal)

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return InterpolatingLine.position(self, longitudinal, lateral)

    def local_coordinates(self, position: Tuple[float, float], only_in_lane_point=False):
        return InterpolatingLine.local_coordinates(self, position, only_in_lane_point)

    def is_in_same_direction(self, another_lane):
        """
        Return True if two lane is in same direction
        """
        my_start_heading = self.heading_theta_at(0.1)
        another_start_heading = another_lane.heading_theta_at(0.1)

        my_end_heading = self.heading_theta_at(self.length - 0.1)
        another_end_heading = another_lane.heading_theta_at(self.length - 0.1)

        return True if abs(wrap_to_pi(my_end_heading) - wrap_to_pi(another_end_heading)) < 0.2 and abs(
            wrap_to_pi(my_start_heading) - wrap_to_pi(another_start_heading)
        ) < 0.2 else False

    def get_bounding_box(self):
        return self._bounding_box

    def destroy(self):
        self._bounding_box = None
        self.width = None
        self.forbidden = None
        self.priority = None

        self.line_types = None
        self.is_straight = None
        self.start = None
        self.end = None
        self._polygon = None
        InterpolatingLine.destroy(self)
        AbstractLane.destroy(self)

    @property
    def polygon(self):
        return self._polygon
