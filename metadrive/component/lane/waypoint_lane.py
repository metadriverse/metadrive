from typing import Tuple, Union

import numpy as np
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.constants import LineType
from metadrive.utils.interpolating_line import InterpolatingLine
from metadrive.utils.math_utils import get_points_bounding_box
from metadrive.utils.math_utils import wrap_to_pi


class WayPointLane(AbstractLane, InterpolatingLine):
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
        AbstractLane.__init__(self)
        InterpolatingLine.__init__(self, center_line_points)
        self._bounding_box = get_points_bounding_box(center_line_points)
        self.set_speed_limit(speed_limit)
        self.width = width
        self.forbidden = forbidden
        self.priority = priority
        # waymo lane line will be processed separately
        self.line_types = (LineType.NONE, LineType.NONE)
        self.is_straight = True if abs(self.heading_theta_at(0.1) -
                                       self.heading_theta_at(self.length - 0.1)) < np.deg2rad(10) else False
        self.start = self.position(0, 0)
        assert np.linalg.norm(self.start - center_line_points[0]) < 0.1, "Start point error!"
        self.end = self.position(self.length, 0)
        assert np.linalg.norm(self.end - center_line_points[-1]) < 1, "End point error!"

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
        # waymo lane line will be processed separately
        self.line_types = None
        self.is_straight = None
        self.start = None
        self.end = None
        InterpolatingLine.destroy(self)
