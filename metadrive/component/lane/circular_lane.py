import math
from typing import Tuple

import numpy as np

from metadrive.component.lane.pg_lane import PGLane
from metadrive.constants import DrivableAreaProperty
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
        priority: int = 0
    ) -> None:
        assert angle > 0, "Angle should be greater than 0"
        super().__init__()
        self.set_speed_limit(speed_limit)
        self.center = Vector(center)
        self.radius = radius
        self._clock_wise = clockwise
        self.start_phase = start_phase
        self.end_phase = self.start_phase + (-angle if self.is_clockwise() else angle)
        self.angle = angle
        self.direction = -1 if clockwise else 1
        self.width = width
        self.line_types = line_types
        self.forbidden = forbidden
        self.priority = priority

        self.length = self.radius * (self.end_phase - self.start_phase) * self.direction
        assert self.length > 0, "end_phase should > (<) start_phase if anti-clockwise (clockwise)"
        self.start = self.position(0, 0)
        self.end = self.position(self.length, 0)

    def update_properties(self):
        self.length = self.radius * (self.end_phase - self.start_phase) * self.direction
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
        delta_x = position[0] - self.center[0]
        delta_y = position[1] - self.center[1]
        phi = math.atan2(delta_y, delta_x)
        phi = self.start_phase + wrap_to_pi(phi - self.start_phase)
        r = norm(delta_x, delta_y)
        longitudinal = self.direction * (phi - self.start_phase) * self.radius
        # lateral = self.direction * (self.radius - r)
        lateral = -self.direction * (self.radius - r)
        return longitudinal, lateral

    def construct_lane_in_block(self, block, lane_index):
        if self.index is None:
            self.index = lane_index
        segment_num = int(self.length / DrivableAreaProperty.LANE_SEGMENT_LENGTH)
        if segment_num == 0:
            middle = self.position(self.length / 2, 0)
            end = self.position(self.length, 0)
            theta = self.heading_theta_at(self.length / 2)
            width = self.width_at(0) + DrivableAreaProperty.SIDEWALK_LINE_DIST * 2
            self.construct_lane_segment(block, middle, width, self.length, theta, lane_index)
            return

        for i in range(segment_num):
            middle = self.position(self.length * (i + .5) / segment_num, 0)
            theta = self.heading_theta_at(self.length * (i + .5) / segment_num)
            width = self.width_at(0) + DrivableAreaProperty.SIDEWALK_LINE_DIST * 2
            length = self.length
            self._construct_lane_only_vis_segment(block, middle, width, length * 1.3 / segment_num, theta)

        self._construct_lane_only_physics_polygon(block, self.polygon)

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
