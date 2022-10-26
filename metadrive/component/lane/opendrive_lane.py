import math
from typing import Tuple, Sequence, Union
from metadrive.utils.opendrive_map_utils.elements.geometry import Line
import numpy as np
from metadrive.component.lane.metadrive_lane import MetaDriveLane
from metadrive.constants import LineType
from metadrive.utils.math_utils import norm
from metadrive.utils.opendrive_map_utils.map_load import get_lane_id


class OpenDriveLane(MetaDriveLane):
    """An OpenDrive Lane"""

    def __init__(
            self,
            width,
            lane_data
    ) -> None:
        super(OpenDriveLane, self).__init__()
        self.lane_data = lane_data
        self.index = get_lane_id(lane_data)
        self._section_index = lane_data.lane_section.idx

        self.single_side = lane_data.lane_section.singleSide
        geo = self.lane_data.parentRoad.planView._geometries[self._section_index]
        assert isinstance(geo, Line), "Only support OpenDrive straight line"

        self.width = width
        self.heading = geo.heading
        self.length = geo.length
        self.start = geo.start_position
        self.end = self.start + np.array([np.cos(self.heading) * self.length, np.sin(self.heading) * self.length])

        self.roadMark_color = lane_data.roadMark.get("color", None)
        self.roadMark_type = lane_data.roadMark.get("type", None)
        self.roadMark_material = lane_data.roadMark.get("material", None)

        self.direction = (self.end - self.start) / self.length
        self.direction_lateral = np.array([-self.direction[1], self.direction[0]])

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return self.start + longitudinal * self.direction + lateral * self.direction_lateral

    def heading_theta_at(self, longitudinal: float) -> float:
        return self.heading

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: Tuple[float, float]) -> Tuple[float, float]:
        delta_x = position[0] - self.start[0]
        delta_y = position[1] - self.start[1]
        longitudinal = delta_x * self.direction[0] + delta_y * self.direction[1]
        lateral = delta_x * self.direction_lateral[0] + delta_y * self.direction_lateral[1]
        return float(longitudinal), float(lateral)

    def construct_lane_in_block(self, block, lane_index=None):
        """
        Straight lane can be represented by one segment
        """
        middle = self.position(self.length / 2, 0)
        end = self.position(self.length, 0)
        direction_v = end - middle
        theta = -math.atan2(direction_v[1], direction_v[0])
        width = self.width_at(0) + block.SIDEWALK_LINE_DIST * 2
        self.construct_lane_segment(block, middle, width, self.length, theta, lane_index)

    def is_lane_line(self):
        return True if self.roadMark_type in ["solid", "broken", "broken broken", "solid solid", "solid broken",
                                              "broken solid"] else False
