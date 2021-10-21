from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.constants import DrivableAreaProperty


class MetaDriveLane(AbstractLane):
    radius = 0.0

    def construct_sidewalk(self, block, lateral):
        radius = self.radius
        segment_num = int(self.length / DrivableAreaProperty.SIDEWALK_LENGTH)
        for segment in range(segment_num):
            lane_start = self.position(segment * DrivableAreaProperty.SIDEWALK_LENGTH, lateral)
            if segment != segment_num - 1:
                lane_end = self.position((segment + 1) * DrivableAreaProperty.SIDEWALK_LENGTH, lateral)
            else:
                lane_end = self.position(self.length, lateral)
            self.construct_sidewalk_segment(block, lane_start, lane_end, radius, self.direction)