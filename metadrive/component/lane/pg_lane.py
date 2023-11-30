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
