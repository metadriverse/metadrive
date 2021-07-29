from typing import Sequence, Tuple

import numpy as np

from pgdrive.component.base_object import BaseObject

LaneIndex = Tuple[str, str, int]


class BaseStaticObject(BaseObject):
    def __init__(self, lane, lane_index: LaneIndex, position: Sequence[float], heading: float = 0., random_seed=None):
        """
        :param lane: the lane to spawn object
        :param lane_index: the lane_index of the spawn point
        :param position: cartesian position of object in the surface
        :param heading: the angle from positive direction of horizontal axis
        """
        super(BaseStaticObject, self).__init__(random_seed=random_seed)
        self.position = position
        self.speed = 0
        self.heading = heading / np.pi * 180
        self.lane_index = lane_index
        self.lane = lane
        self.body_node = None
        self.heading_theta = 0
