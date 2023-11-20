import math
from typing import Tuple

import numpy as np

from metadrive.component.lane.pg_lane import PGLane
from metadrive.constants import MetaDriveType
from metadrive.constants import PGLineType
from metadrive.utils.math import wrap_to_pi, norm, Vector


class CircularLane(PGLane):
    """A lane going in circle arc."""
    def __init__(
        self,
        center: Vector,
        radius: float,
        start_phase: float,
        angle: float,
        clockwise: bool = True,
        width: float = PGLane.DEFAULT_WIDTH,
        line_types: Tuple[PGLineType, PGLineType] = (PGLineType.BROKEN, PGLineType.BROKEN),
        forbidden: bool = False,
        speed_limit: float = 1000,
        priority: int = 0,
        metadrive_type=MetaDriveType.LANE_SURFACE_STREET,
    ) -> None:
        assert angle > 0, "Angle should be greater than 0"
        super().__init__(metadrive_type)
        self.set_speed_limit(speed_limit)
        self.center = Vector(center)
        self.radius = radius
        self._clock_wise = clockwise
        self.start_phase = wrap_to_pi(start_phase)
        self.angle = angle
        # TODO(pzh): Don't do the wrap_to_pi for self.end_phase because sometime we might want to use self.end_phase
        #  - self.start_phase to get self.angle (or we probably should eliminate those use cases?)
        self.end_phase = self.start_phase + (-self.angle if self.is_clockwise() else self.angle)
        self.direction = -1 if clockwise else 1
        self.width = width
        self.line_types = line_types
        self.forbidden = forbidden
        self.priority = priority

        self.length = abs(self.radius * (self.end_phase - self.start_phase))
        assert self.length > 0, "end_phase should > (<) start_phase if anti-clockwise (clockwise)"
        self.start = self.position(0, 0)
        self.end = self.position(self.length, 0)

    def update_properties(self):
        self.length = abs(self.radius * (self.end_phase - self.start_phase))
        assert self.length > 0, "end_phase should > (<) start_phase if anti-clockwise (clockwise)"
        self.start = self.position(0, 0)
        self.end = self.position(self.length, 0)

    # def position(self, longitudinal: float, lateral: float) -> np.ndarray:
    def position(self, longitudinal: float, lateral: float) -> Vector:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        # return self.center + (self.radius - lateral * self.direction) * np.array([math.cos(phi), math.sin(phi)])
        # return self.center + (self.radius - lateral * self.direction) * Vector((math.cos(phi), math.sin(phi)))
        return self.center + (self.radius + lateral * self.direction) * Vector((math.cos(phi), math.sin(phi)))

    def heading_theta_at(self, longitudinal: float) -> float:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        psi = phi + math.pi / 2 * self.direction
        return psi

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Compute the local coordinates (longitude, lateral) of the given position in this circular lane.

        Args:
            position: floats in 2D

        Returns:
            longtidue in float
            lateral in float
        """
        delta_x = position[0] - self.center[0]
        delta_y = position[1] - self.center[1]
        abs_phase = math.atan2(delta_y, delta_x)
        abs_phase = wrap_to_pi(abs_phase)

        # Processing the relative phase (angle) is extremely complex.
        start_phase = wrap_to_pi(self.start_phase)
        end_phase = wrap_to_pi(self.end_phase)

        diff_to_start_phase = abs(wrap_to_pi(abs_phase - start_phase))
        diff_to_end_phase = abs(wrap_to_pi(abs_phase - end_phase))

        if diff_to_start_phase > np.pi and diff_to_end_phase > np.pi:
            raise ValueError(
                f"Undetermined position. Relative phase of the given point to the start phase is"
                f" {wrap_to_pi(abs_phase - start_phase)} while the phase to the end phase is "
                f"{wrap_to_pi(abs_phase - end_phase)}. Both of them are > 180deg. "
                f"We don't know how to compute the longitudinal in this case."
            )

        # If the point is closer to the end point
        if diff_to_start_phase > diff_to_end_phase:
            if self.is_clockwise():
                diff = self.end_phase - abs_phase
            else:
                diff = abs_phase - self.end_phase
            relative_phase = wrap_to_pi(diff)
            longitudinal = relative_phase * self.radius + self.length
        else:
            if self.is_clockwise():
                diff = self.start_phase - abs_phase
            else:
                diff = abs_phase - self.start_phase
            relative_phase = wrap_to_pi(diff)
            longitudinal = relative_phase * self.radius

        # Compute the lateral
        distance_to_center = norm(delta_x, delta_y)
        lateral = self.direction * (distance_to_center - self.radius)

        return longitudinal, lateral

    @property
    def polygon(self):
        if self._polygon is None:
            start_heading = self.heading_theta_at(0)
            start_dir = [math.cos(start_heading), math.sin(start_heading)]

            end_heading = self.heading_theta_at(self.length)
            end_dir = [math.cos(end_heading), math.sin(end_heading)]
            polygon = []
            longs = np.arange(0, self.length + self.POLYGON_SAMPLE_RATE, self.POLYGON_SAMPLE_RATE)
            for k, lateral in enumerate([+self.width / 2, -self.width / 2]):
                if k == 1:
                    longs = longs[::-1]
                for t, longitude in enumerate(longs):
                    point = self.position(longitude, lateral)
                    if (t == 0 and k == 0) or (t == len(longs) - 1 and k == 1):
                        # control the adding sequence
                        if k == 1:
                            # last point
                            polygon.append([point[0], point[1]])

                        # extend
                        polygon.append(
                            [
                                point[0] - start_dir[0] * self.POLYGON_SAMPLE_RATE,
                                point[1] - start_dir[1] * self.POLYGON_SAMPLE_RATE
                            ]
                        )

                        if k == 0:
                            # first point
                            polygon.append([point[0], point[1]])
                    elif (t == 0 and k == 1) or (t == len(longs) - 1 and k == 0):

                        if k == 0:
                            # second point
                            polygon.append([point[0], point[1]])

                        polygon.append(
                            [
                                point[0] + end_dir[0] * self.POLYGON_SAMPLE_RATE,
                                point[1] + end_dir[1] * self.POLYGON_SAMPLE_RATE
                            ]
                        )

                        if k == 1:
                            # third point
                            polygon.append([point[0], point[1]])
                    else:
                        polygon.append([point[0], point[1]])
            self._polygon = polygon
        return self._polygon

    def is_clockwise(self):
        return self._clock_wise
