from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.type import MetaDriveType
from metadrive.constants import PGLineType, PGLineColor
from typing import Tuple


class PGLane(AbstractLane):
    POLYGON_SAMPLE_RATE = 1
    radius = 0.0
    line_types: Tuple[PGLineType, PGLineType]
    line_colors = [PGLineColor.GREY, PGLineColor.GREY]
    DEFAULT_WIDTH: float = 3.5

    def __init__(self, type=MetaDriveType.LANE_SURFACE_STREET):
        super(PGLane, self).__init__(type)

    # def construct_sidewalk(self, block, sidewalk_height=None, lateral_direction=1):
    #     """
    #     Construct the sidewalk for this lane
    #
    #     Args:
    #         block: Current block.
    #         sidewalk_height: The height of sidewalk. Default to None and PGDrivableAreaProperty.SIDEWALK_THICKNESS will
    #             be used.
    #         lateral_direction: Whether to extend the sidewalk outward (to the right of the road) or inward (to the left
    #             of the road). It's useful if we want to put sidewalk to the center of the road, e.g. in the scenario
    #             where we only have "positive road" and there is no roads in opposite direction. Should be either -1 or
    #             +1.
    #
    #     Returns: None
    #     """
    #     # engine = get_engine()
    #     # cloud_points_vis = ColorLineNodePath(
    #     #     engine.render, thickness=3.0
    #     # )
    #     # draw_lists = [[], []]
    #     if str(self.index) in block.sidewalks:
    #         return
    #     polygon = []
    #     longs = np.arange(
    #         0, self.length + PGDrivableAreaProperty.SIDEWALK_LENGTH, PGDrivableAreaProperty.SIDEWALK_LENGTH
    #     )
    #     start_lat = +self.width_at(0) / 2 + 0.2
    #     side_lat = start_lat + PGDrivableAreaProperty.SIDEWALK_WIDTH
    #     assert lateral_direction == -1 or lateral_direction == 1
    #     start_lat *= lateral_direction
    #     side_lat *= lateral_direction
    #     if self.radius != 0 and side_lat > self.radius:
    #         raise ValueError(
    #             "The sidewalk width ({}) is too large."
    #             " It should be < radius ({})".format(side_lat, self.radius)
    #         )
    #     for k, lateral in enumerate([start_lat, side_lat]):
    #         if k == 1:
    #             longs = longs[::-1]
    #         for longitude in longs:
    #             longitude = min(self.length + 0.1, longitude)
    #             point = self.position(longitude, lateral)
    #             polygon.append([point[0], point[1]])
    #             # draw_lists[k].append([point[0], point[1], 1])
    #     # cloud_points_vis.drawLines(draw_lists)
    #     # cloud_points_vis.create()
    #     block.sidewalks[str(self.index)] = {"polygon": polygon, "height": sidewalk_height}
    #
    # def construct_lane_line_in_block(self, block, construct_left_right=(True, True)):
    #     """
    #     Construct lane line in the Panda3d world for getting contact information
    #     """
    #     for idx, line_type, line_color, need, in zip([-1, 1], self.line_types, self.line_colors, construct_left_right):
    #         if not need:
    #             continue
    #         lateral = idx * self.width_at(0) / 2
    #         if line_type == PGLineType.CONTINUOUS:
    #             self.construct_continuous_line(block, lateral, line_color, line_type)
    #         elif line_type == PGLineType.BROKEN:
    #             self.construct_broken_line(block, lateral, line_color, line_type)
    #         elif line_type == PGLineType.SIDE:
    #             self.construct_continuous_line(block, lateral, line_color, line_type)
    #             self.construct_sidewalk(block)
    #         elif line_type == PGLineType.GUARDRAIL:
    #             self.construct_continuous_line(block, lateral, line_color, line_type)
    #             self.construct_sidewalk(
    #                 block, sidewalk_height=PGDrivableAreaProperty.GUARDRAIL_HEIGHT, lateral_direction=idx
    #             )
    #         elif line_type == PGLineType.NONE:
    #             continue
    #         else:
    #             raise ValueError(
    #                 "You have to modify this function and implement a constructing method for line type: {}".
    #                 format(line_type)
    #             )
