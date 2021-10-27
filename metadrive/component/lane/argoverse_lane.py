from typing import Optional, List

import numpy as np
from argoverse.map_representation.lane_segment import LaneSegment
from metadrive.component.lane.waypoint_lane import WayPointLane
from metadrive.constants import LineType, LineColor


class ArgoverseLane(WayPointLane, LaneSegment):
    # according to api of get_vector_map_lane_polygons(), the lane width in argoverse dataset is 3.8m
    LANE_WIDTH = 3.8

    def __init__(
        self,
        start_node: str,
        end_node: str,
        id: int,
        has_traffic_control: bool,
        turn_direction: str,
        is_intersection: bool,
        l_neighbor_id: Optional[int],
        r_neighbor_id: Optional[int],
        predecessors: List[int],
        successors: Optional[List[int]],
        centerline: np.ndarray,
        lane_width=None
    ):
        # convert_to_MetaDrive_coordinates
        centerline[:, 1] *= -1
        LaneSegment.__init__(
            self, id, has_traffic_control, turn_direction, is_intersection, l_neighbor_id, r_neighbor_id, predecessors,
            successors, centerline
        )
        WayPointLane.__init__(self, centerline, self.LANE_WIDTH if lane_width is None else lane_width)
        self.start_node = start_node
        self.end_node = end_node
        # if is_intersection:
        #     # if turn_direction == "RIGHT" and r_neighbor_id is None:
        #     #     self.line_types = (LineType.NONE, LineType.CONTINUOUS)
        #     # elif turn_direction == "LEFT" and l_neighbor_id is None:
        #     #     self.line_types = (LineType.CONTINUOUS,LineType.NONE)
        #     # else:
        #     #     self.line_types = (LineType.CONTINUOUS if l_neighbor_id is None else LineType.NONE, LineType.CONTINUOUS if r_neighbor_id is None else LineType.NONE)
        #     self.line_types = (LineType.NONE, LineType.NONE)
        # else:
        self.line_types = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        self.line_color = (LineColor.GREY, LineColor.GREY)

    def is_in_same_direction(self, another_lane):
        if self.id == another_lane.l_neighbor_id and self.l_neighbor_id == another_lane.id:
            return False
        return super(ArgoverseLane, self).is_in_same_direction(another_lane)
