import numpy as np
from enum import Enum

from metadrive.component.lane.straight_lane import StraightLane


class ExtensionDirection(Enum):
    EXTEND = 0
    SHRINK = 1


class ExtendingLane(StraightLane):
    def __init__(self, extension_direction: ExtensionDirection, *args, **kwargs):
        super(ExtendingLane, self).__init__(*args, **kwargs)
        self.extension_direction = extension_direction

    # def width_at(self, longitudinal: float) -> float:
    #     if self.extension_direction == ExtensionDirection.EXTEND:
    #         return (longitudinal / self.length) * self.width
    #     else:
    #         return self.width - (longitudinal / self.length) * self.width
    #
    # def get_polyline(self, interval=2, lateral=0):
    #     ret = []
    #     for i in np.arange(0, self.length, interval):
    #         ret.append(self.position(i, lateral))
    #     last_lateral = self.width_at(self.length) - self.width / 2
    #     ret.append(self.position(self.length, min(lateral, last_lateral)))
    #     return np.array(ret)

    @property
    def polygon(self):
        if self._polygon is not None:
            return self._polygon

        polygon = []
        longs = np.arange(0, self.length + self.POLYGON_SAMPLE_RATE, self.POLYGON_SAMPLE_RATE)
        for longitude in longs:
            point = self.position(longitude, -self.width / 2)
            polygon.append([point[0], point[1]])
        for longitude in longs[::-1]:
            latitude = self.width_at(longitude) - self.width / 2
            point = self.position(longitude, latitude)
            polygon.append([point[0], point[1]])
        self._polygon = np.asarray(polygon)

        return self._polygon
