from typing import Sequence, Tuple

import numpy as np

from pgdrive.base_class.base_object import BaseObject

LaneIndex = Tuple[str, str, int]


class BaseStaticObject(BaseObject):
    MASS = 1

    def __init__(self, lane, position: Sequence[float], heading: float = 0., random_seed=None):
        """
        :param lane: the lane to spawn object
        :param position: cartesian position of object in the surface
        :param heading: the angle from positive direction of horizontal axis
        """
        super(BaseStaticObject, self).__init__(random_seed=random_seed)
        self.position = position
        self.speed = 0
        self.velocity = np.array([0.0, 0.0])
        self.heading = heading / np.pi * 180
        self.lane_index = lane.index
        self.lane = lane

    @property
    def heading_theta(self):
        return self.heading

    def set_static(self, static: bool = False):
        mass = 0 if static else self.MASS
        self._body.setMass(mass)
