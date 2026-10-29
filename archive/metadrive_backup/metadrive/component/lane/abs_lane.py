import math
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, AnyStr

import numpy as np
from shapely import geometry

from metadrive.constants import MetaDriveType
from metadrive.utils import norm


class AbstractLane(MetaDriveType):
    """A lane on the road, described by its central curve."""

    metaclass__ = ABCMeta
    length: float  # lane length
    start: Tuple[float, float]  # lane start position
    end: Tuple[float, float]  # lane end position

    def __init__(self, type=MetaDriveType.LANE_SURFACE_STREET):
        super(AbstractLane, self).__init__(type)
        self.speed_limit = 1000  # should be set manually
        self.index: Union[Tuple, AnyStr, None] = None
        self._polygon = None
        self._shapely_polygon = None
        self.need_lane_localization = True

    def set_speed_limit(self, speed_limit):
        self.speed_limit = speed_limit

    @abstractmethod
    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        """
        Convert local lane coordinates to a physx_world position.

        :param longitudinal: longitudinal lane coordinate [m]
        :param lateral: lateral lane coordinate [m]
        :return: the corresponding physx_world position [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        """
        Convert a physx_world position to local lane coordinates.

        :param position: a physx_world position [m]
        :return: the (longitudinal, lateral) lane coordinates [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def heading_theta_at(self, longitudinal: float) -> float:
        """
        Get the lane heading at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane heading [rad]
        """
        raise NotImplementedError()

    def heading_at(self, longitudinal) -> np.array:
        heaidng_theta = self.heading_theta_at(longitudinal)
        return np.array([math.cos(heaidng_theta), math.sin(heaidng_theta)])

    @abstractmethod
    def width_at(self, longitudinal: float) -> float:
        """
        Get the lane width at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane width [m]
        """
        raise NotImplementedError()

    def distance(self, position):
        """Compute the L1 distance [m] from a position to the lane."""
        s, r = self.local_coordinates(position)
        a = s - self.length
        b = 0 - s
        # return abs(r) + max(s - self.length, 0) + max(0 - s, 0)
        return abs(r) + (a if a > 0 else 0) + (b if b > 0 else 0)

    def is_previous_lane_of(self, target_lane, error_region=1e-1):
        x_1, y_1 = self.end
        x_2, y_2 = target_lane.start
        if norm(x_1 - x_2, y_1 - y_2) < error_region:
            return True
        return False

    def destroy(self):
        self._polygon = None
        self._shapely_polygon = None

    def get_polyline(self, interval=2, lateral=0):
        """
        This method will return the center line of this Lane in a discrete vector representation
        """
        ret = []
        for i in np.arange(0, self.length, interval):
            ret.append(self.position(i, lateral))
        ret.append(self.position(self.length, lateral))
        return np.array(ret)

    @property
    def id(self):
        return self.index

    def point_on_lane(self, point):
        """
        Return True if the point is in the lane polygon
        """
        s_point = geometry.Point(point[0], point[1])
        return self.shapely_polygon.contains(s_point)

    @property
    def polygon(self):
        """
        Return the polygon of this lane
        Returns: a list of 2D points representing Polygon

        """
        raise NotImplementedError("Overwrite this function to allow getting polygon for this lane")

    @property
    def shapely_polygon(self):
        """Return the polygon in shapely.geometry.Polygon"""
        if self._shapely_polygon is None:
            assert self.polygon is not None
            self._shapely_polygon = geometry.Polygon(geometry.LineString(self.polygon))
        return self._shapely_polygon
