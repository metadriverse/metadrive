from typing import Sequence, Tuple

import numpy as np

from metadrive.base_class.base_object import BaseObject

LaneIndex = Tuple[str, str, int]


class BaseStaticObject(BaseObject):
    MASS = 1

    def __init__(self, lane, position: Sequence[float], heading_theta: float = 0., random_seed=None):
        """
        :param lane: the lane to spawn object
        :param position: cartesian position of object in the surface
        :param heading_theta: the angle from positive direction of horizontal axis
        """
        super(BaseStaticObject, self).__init__(random_seed=random_seed)
        self.set_position(position, 0)
        self.set_heading_theta(heading_theta)
        self.lane_index = lane.index
        self.lane = lane

    def set_static(self, static: bool = False):
        mass = 0 if static else self.MASS
        self._body.setMass(mass)
