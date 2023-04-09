import numpy as np

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.constants import DrivableAreaProperty


class PGLane(AbstractLane):
    POLYGON_SAMPLE_RATE = 2
    radius = 0.0

    def __init__(self):
        super(PGLane, self).__init__()
        self._polygon = None

    def construct_sidewalk(self, block, lateral):
        radius = self.radius
        segment_num = int(self.length / DrivableAreaProperty.SIDEWALK_LENGTH)
        for segment in range(segment_num):
            lane_start = self.position(segment * DrivableAreaProperty.SIDEWALK_LENGTH, lateral)
            lane_end = self.position((segment + 1) * DrivableAreaProperty.SIDEWALK_LENGTH, lateral)
            if segment == segment_num - 1:
                lane_end = self.position(self.length, lateral)
            if radius == 0:
                factor = 1
            else:
                if self.direction == 1:
                    factor = (1 - block.SIDEWALK_LINE_DIST / radius)
                else:
                    factor = (1 + block.SIDEWALK_WIDTH / radius) * (1 + block.SIDEWALK_LINE_DIST / radius)
            node_path_list = self.construct_sidewalk_segment(
                block,
                lane_start,
                lane_end,
                length_multiply=factor,
                extra_thrust=DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST
            )
            self._node_path_list.extend(node_path_list)

    @property
    def polygon(self):
        if self._polygon is None:
            polygon = []
            longs = np.arange(0, self.length + 1., self.POLYGON_SAMPLE_RATE)
            for lateral in [+self.width_at(0) / 2, -self.width_at(0) / 2]:
                for longitude in longs:
                    point = self.position(longitude, lateral)
                    polygon.append([point[0], point[1], 0.1])
                    polygon.append([point[0], point[1], 0.])
            self._polygon = np.asarray(polygon)
        return self._polygon

    def get_polygon(self):
        return self.polygon
