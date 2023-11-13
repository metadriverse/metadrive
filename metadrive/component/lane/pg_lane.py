import numpy as np

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.constants import PGDrivableAreaProperty
from metadrive.constants import PGLineType
from metadrive.engine.core.draw_line import ColorLineNodePath
from metadrive.engine.engine_utils import get_engine
from metadrive.type import MetaDriveType


class PGLane(AbstractLane):
    POLYGON_SAMPLE_RATE = 1
    radius = 0.0

    def __init__(self, type=MetaDriveType.LANE_SURFACE_STREET):
        super(PGLane, self).__init__(type)

    def construct_broken_line(self, block, lateral, line_color, line_type):
        """
        Lateral: left[-1/2 * width] or right[1/2 * width]
        """
        segment_num = int(self.length / (2 * PGDrivableAreaProperty.STRIPE_LENGTH))
        for segment in range(segment_num):
            start = self.position(segment * PGDrivableAreaProperty.STRIPE_LENGTH * 2, lateral)
            end = self.position(
                segment * PGDrivableAreaProperty.STRIPE_LENGTH * 2 + PGDrivableAreaProperty.STRIPE_LENGTH, lateral
            )
            if segment == segment_num - 1:
                end = self.position(self.length - PGDrivableAreaProperty.STRIPE_LENGTH, lateral)
            node_path_list = self.construct_lane_line_segment(block, start, end, line_color, line_type)
            self._node_path_list.extend(node_path_list)

    def construct_continuous_line(self, block, lateral, line_color, line_type):
        """
        We process straight line to several pieces by default, which can be optimized through overriding this function
        Lateral: left[-1/2 * width] or right[1/2 * width]
        """
        segment_num = int(self.length / PGDrivableAreaProperty.LANE_SEGMENT_LENGTH)
        if segment_num == 0:
            start = self.position(0, lateral)
            end = self.position(self.length, lateral)
            node_path_list = self.construct_lane_line_segment(block, start, end, line_color, line_type)
            self._node_path_list.extend(node_path_list)
        for segment in range(segment_num):
            start = self.position(PGDrivableAreaProperty.LANE_SEGMENT_LENGTH * segment, lateral)
            if segment == segment_num - 1:
                end = self.position(self.length, lateral)
            else:
                end = self.position((segment + 1) * PGDrivableAreaProperty.LANE_SEGMENT_LENGTH, lateral)
            node_path_list = self.construct_lane_line_segment(block, start, end, line_color, line_type)
            self._node_path_list.extend(node_path_list)

    def construct_sidewalk(self, block):
        """
        Construct the sidewalk for this lane
        Args:
            block:

        Returns:

        """
        # engine = get_engine()
        # cloud_points_vis = ColorLineNodePath(
        #     engine.render, thickness=3.0
        # )
        # draw_lists = [[], []]
        if str(self.index) in block.sidewalks:
            return
        polygon = []
        longs = np.arange(
            0, self.length + PGDrivableAreaProperty.SIDEWALK_LENGTH, PGDrivableAreaProperty.SIDEWALK_LENGTH
        )
        start_lat = +self.width_at(0) / 2 + 0.2
        side_lat = start_lat + PGDrivableAreaProperty.SIDEWALK_WIDTH
        if self.radius != 0 and side_lat > self.radius:
            raise ValueError("The sidewalk width ({}) is too large."
                             " It should be < radius ({})".format(side_lat, self.radius))
        for k, lateral in enumerate([start_lat, side_lat]):
            if k == 1:
                longs = longs[::-1]
            for longitude in longs:
                longitude = min(self.length + 0.1, longitude)
                point = self.position(longitude, lateral)
                polygon.append([point[0], point[1]])
                # draw_lists[k].append([point[0], point[1], 1])
        # cloud_points_vis.drawLines(draw_lists)
        # cloud_points_vis.create()
        block.sidewalks[str(self.index)] = {"polygon": polygon}

    def construct_lane_line_in_block(self, block, construct_left_right=(True, True)):
        """
        Construct lane line in the Panda3d world for getting contact information
        """
        for idx, line_type, line_color, need, in zip([-1, 1], self.line_types, self.line_colors, construct_left_right):
            if not need:
                continue
            lateral = idx * self.width_at(0) / 2
            if line_type == PGLineType.CONTINUOUS:
                self.construct_continuous_line(block, lateral, line_color, line_type)
            elif line_type == PGLineType.BROKEN:
                self.construct_broken_line(block, lateral, line_color, line_type)
            elif line_type == PGLineType.SIDE:
                self.construct_continuous_line(block, lateral, line_color, line_type)
                self.construct_sidewalk(block)
            elif line_type == PGLineType.NONE:
                continue
            else:
                raise ValueError(
                    "You have to modify this cuntion and implement a constructing method for line type: {}".
                    format(line_type)
                )
