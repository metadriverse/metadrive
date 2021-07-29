import math
from typing import Tuple

import numpy as np
from pgdrive.scene_creator.lane.abs_lane import AbstractLane, Vector
from pgdrive.constants import LineType
from pgdrive.utils.math_utils import wrap_to_pi, norm, PGVector


class CircularLane(AbstractLane):
    """A lane going in circle arc."""
    def __init__(
        self,
        center: Vector,
        radius: float,
        start_phase: float,
        end_phase: float,
        clockwise: bool = True,
        width: float = AbstractLane.DEFAULT_WIDTH,
        line_types: Tuple[LineType, LineType] = (LineType.BROKEN, LineType.BROKEN),
        forbidden: bool = False,
        speed_limit: float = 1000,
        priority: int = 0
    ) -> None:
        super().__init__()
        self.set_speed_limit(speed_limit)
        self.center = PGVector(center)
        self.radius = radius
        self.start_phase = start_phase
        self.end_phase = end_phase
        self.direction = 1 if clockwise else -1
        self.width = width
        self.line_types = line_types
        self.forbidden = forbidden
        self.priority = priority
        self.length = self.radius * (self.end_phase - self.start_phase) * self.direction

    def update_properties(self):
        self.length = self.radius * (self.end_phase - self.start_phase) * self.direction

    # def position(self, longitudinal: float, lateral: float) -> np.ndarray:
    def position(self, longitudinal: float, lateral: float) -> PGVector:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        # return self.center + (self.radius - lateral * self.direction) * np.array([math.cos(phi), math.sin(phi)])
        return self.center + (self.radius - lateral * self.direction) * PGVector((math.cos(phi), math.sin(phi)))

    def heading_at(self, longitudinal: float) -> float:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        psi = phi + np.pi / 2 * self.direction
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
