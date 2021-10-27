import math
from typing import Tuple

from metadrive.component.lane.metadrive_lane import MetaDriveLane
from metadrive.constants import LineType
from metadrive.utils.math_utils import wrap_to_pi, norm, Vector


class CircularLane(MetaDriveLane):
    """A lane going in circle arc."""

    CIRCULAR_SEGMENT_LENGTH = 4

    def __init__(
        self,
        center: Vector,
        radius: float,
        start_phase: float,
        end_phase: float,
        clockwise: bool = True,
        width: float = MetaDriveLane.DEFAULT_WIDTH,
        line_types: Tuple[LineType, LineType] = (LineType.BROKEN, LineType.BROKEN),
        forbidden: bool = False,
        speed_limit: float = 1000,
        priority: int = 0
    ) -> None:
        super().__init__()
        self.set_speed_limit(speed_limit)
        self.center = Vector(center)
        self.radius = radius
        self.start_phase = start_phase
        self.end_phase = end_phase
        self.direction = 1 if clockwise else -1
        self.width = width
        self.line_types = line_types
        self.forbidden = forbidden
        self.priority = priority

        self.length = self.radius * (self.end_phase - self.start_phase) * self.direction
        self.start = self.position(0, 0)
        self.end = self.position(self.length, 0)

    def update_properties(self):
        self.length = self.radius * (self.end_phase - self.start_phase) * self.direction
        self.start = self.position(0, 0)
        self.end = self.position(self.length, 0)

    # def position(self, longitudinal: float, lateral: float) -> np.ndarray:
    def position(self, longitudinal: float, lateral: float) -> Vector:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        # return self.center + (self.radius - lateral * self.direction) * np.array([math.cos(phi), math.sin(phi)])
        return self.center + (self.radius - lateral * self.direction) * Vector((math.cos(phi), math.sin(phi)))

    def heading_theta_at(self, longitudinal: float) -> float:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        psi = phi + math.pi / 2 * self.direction
        return psi

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: Tuple[float, float]) -> Tuple[float, float]:
        delta_x = position[0] - self.center[0]
        delta_y = position[1] - self.center[1]
        phi = math.atan2(delta_y, delta_x)
        phi = self.start_phase + wrap_to_pi(phi - self.start_phase)
        r = norm(delta_x, delta_y)
        longitudinal = self.direction * (phi - self.start_phase) * self.radius
        lateral = self.direction * (self.radius - r)
        return longitudinal, lateral
