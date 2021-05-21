from typing import Sequence, Tuple

import numpy as np

from pgdrive.utils.element import Element

LaneIndex = Tuple[str, str, int]


class BaseObject(Element):
    def __init__(self, lane, lane_index: LaneIndex, position: Sequence[float], heading: float = 0.):
        """
        :param lane: the lane to spawn object
        :param lane_index: the lane_index of the spawn point
        :param position: cartesian position of object in the surface
        :param heading: the angle from positive direction of horizontal axis
        """
        super(BaseObject, self).__init__()
        self.position = position
        self.speed = 0
        self.heading = heading / np.pi * 180
        self.lane_index = lane_index
        self.lane = lane
        self.body_node = None
