import math
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, Sequence

import numpy as np

Vector = Union[np.ndarray, Sequence[float]]


class LineType:
    """A lane side line type."""

    NONE = "none"
    STRIPED = "stripped"
    CONTINUOUS = "continuous"
    SIDE = "side"
    CONTINUOUS_LINE = 4  # deprecated

    @staticmethod
    def prohibit(line_type) -> bool:
        if line_type in [LineType.CONTINUOUS, LineType.CONTINUOUS_LINE, LineType.SIDE]:
            return True
        else:
            return False


class LineColor:
    GREY = (1, 1, 1, 1)
    YELLOW = (245 / 255, 192 / 255, 67 / 255, 1)


class AbstractLane(object):
    """A lane on the road, described by its central curve."""

    metaclass__ = ABCMeta
    DEFAULT_WIDTH: float = 4
    VEHICLE_LENGTH: float = 5
    length: float = 0
    line_types: Tuple[LineType, LineType]
    forbidden = None
    line_color = [LineColor.GREY, LineColor.GREY]

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
    def heading_at(self, longitudinal: float) -> float:
        """
        Get the lane heading at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane heading [rad]
        """
        raise NotImplementedError()

    @abstractmethod
    def width_at(self, longitudinal: float) -> float:
        """
        Get the lane width at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane width [m]
        """
        raise NotImplementedError()

    def on_lane(self, position: np.ndarray, longitudinal: float = None, lateral: float = None, margin: float = 0) \
            -> bool:
        """
        Whether a given physx_world position is on the lane.

        :param position: a physx_world position [m]
        :param longitudinal: (optional) the corresponding longitudinal lane coordinate, if known [m]
        :param lateral: (optional) the corresponding lateral lane coordinate, if known [m]
        :param margin: (optional) a supplementary margin around the lane width
        :return: is the position on the lane?
        """
        if not longitudinal or not lateral:
            longitudinal, lateral = self.local_coordinates(position)
        is_on = math.fabs(lateral) <= self.width_at(longitudinal) / 2 + margin and \
                -self.VEHICLE_LENGTH <= longitudinal < self.length + self.VEHICLE_LENGTH
        return is_on

    def is_reachable_from(self, position: np.ndarray) -> bool:
        """
        Whether the lane is reachable from a given physx_world position

        :param position: the physx_world position [m]
        :return: is the lane reachable?
        """
        if self.forbidden:
            return False
        longitudinal, lateral = self.local_coordinates(position)
        is_close = math.fabs(lateral) <= 2 * self.width_at(longitudinal) and \
                   0 <= longitudinal < self.length + self.VEHICLE_LENGTH
        return is_close

    def after_end(self, position: np.ndarray, longitudinal: float = None, lateral: float = None) -> bool:
        if not longitudinal:
            longitudinal, _ = self.local_coordinates(position)
        return longitudinal > self.length - self.VEHICLE_LENGTH / 2

    def distance(self, position):
        """Compute the L1 distance [m] from a position to the lane."""
        s, r = self.local_coordinates(position)
        a = s - self.length
        b = 0 - s
        # return abs(r) + max(s - self.length, 0) + max(0 - s, 0)
        return abs(r) + (a if a > 0 else 0) + (b if b > 0 else 0)
