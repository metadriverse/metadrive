from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.constants import DrivableAreaProperty


class MetaDriveLane(AbstractLane):
    radius = 0.0

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
            self.construct_sidewalk_segment(
                block,
                lane_start,
                lane_end,
                length_multiply=factor,
                extra_thrust=DrivableAreaProperty.SIDEWALK_WIDTH / 2 + DrivableAreaProperty.SIDEWALK_LINE_DIST
            )
