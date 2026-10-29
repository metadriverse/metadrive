from typing import List, Tuple, Union
import math
import numpy as np

from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.constants import PGLineType, PGLineColor
from metadrive.utils.utils import import_pygame
from metadrive.type import MetaDriveType
from metadrive.constants import PGDrivableAreaProperty
from collections import namedtuple

PositionType = Union[Tuple[float, float], np.ndarray]
pygame = import_pygame()
COLOR_BLACK = pygame.Color("black")
history_object = namedtuple("history_object", "name position heading_theta WIDTH LENGTH color done type")


class ObservationWindow:
    def __init__(self, max_range, resolution):
        self.max_range = max_range
        self.resolution = resolution
        self.receptive_field = None
        self.receptive_field_double = None

        self.canvas_rotate = None
        self.canvas_uncropped = pygame.Surface(
            (int(self.resolution[0] * np.sqrt(2)) + 1, int(self.resolution[1] * np.sqrt(2)) + 1)
        )

        self.canvas_display = pygame.Surface(self.resolution)
        self.canvas_display.fill(COLOR_BLACK)

    def reset(self, canvas_runtime):
        canvas_runtime.fill(COLOR_BLACK)

        # Assume max_range is only the radius!
        self.receptive_field_double = (
            int(canvas_runtime.pix(self.max_range[0] * np.sqrt(2))) * 2,
            int(canvas_runtime.pix(self.max_range[1] * np.sqrt(2))) * 2
        )
        self.receptive_field = (
            int(canvas_runtime.pix(self.max_range[0])) * 2, int(canvas_runtime.pix(self.max_range[1])) * 2
        )
        self.canvas_rotate = pygame.Surface(self.receptive_field_double)
        self.canvas_rotate.fill(COLOR_BLACK)
        self.canvas_display.fill(COLOR_BLACK)
        self.canvas_uncropped.fill(COLOR_BLACK)

    def _blit(self, canvas, position):
        self.canvas_rotate.blit(
            canvas, (0, 0), (
                position[0] - self.receptive_field_double[0] / 2, position[1] - self.receptive_field_double[1] / 2,
                self.receptive_field_double[0], self.receptive_field_double[1]
            )
        )

    def _rotate(self, heading):
        rotation = np.rad2deg(heading) + 90
        scale = self.canvas_uncropped.get_size()[0] / self.canvas_rotate.get_size()[0]
        return pygame.transform.rotozoom(self.canvas_rotate, rotation, scale)

    def _crop(self, new_canvas):
        size = self.canvas_display.get_size()
        self.canvas_display.blit(
            new_canvas,
            (0, 0),
            (
                new_canvas.get_size()[0] / 2 - size[0] / 2,  # Left
                new_canvas.get_size()[1] / 2 - size[1] / 2,  # Top
                size[0],  # Width
                size[1]  # Height
            )
        )

    def render(self, canvas, position, heading):
        # Prepare a runtime canvas for rotation. Assume max_range is only the radius, not diameter!
        self._blit(canvas, position)

        # Rotate the image so that ego is always heading top
        new_canvas = self._rotate(heading)

        # Crop the rotated image and then resize to the desired resolution
        self._crop(new_canvas)

        return self.canvas_display

    def get_observation_window(self):
        return self.canvas_display

    def get_size(self):
        assert self.canvas_rotate is not None
        return self.canvas_rotate.get_size()

    def get_screen_window(self):
        return self.get_observation_window()


class WorldSurface(pygame.Surface):
    """
    A pygame Surface implementing a local coordinate system so that we can move and zoom in the displayed area.
    From highway-env, See more information on its Github page: https://github.com/eleurent/highway-env.
    """

    BLACK = (0, 0, 0)
    GREY = (100, 100, 100)
    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    WHITE = (255, 255, 255)
    INITIAL_SCALING = 5.5
    INITIAL_CENTERING = [0.5, 0.5]
    SCALING_FACTOR = 1.3
    MOVING_FACTOR = 0.1
    LANE_LINE_COLOR = (35, 35, 35)

    def __init__(self, size: Tuple[int, int], flags: object, surf: pygame.SurfaceType) -> None:
        surf.fill(pygame.Color("White"))
        super().__init__(size, flags, surf)
        self.raw_size = size
        self.raw_flags = flags
        self.raw_surface = surf
        self.origin = np.array([0, 0])
        self.scaling = self.INITIAL_SCALING
        self.centering_position = self.INITIAL_CENTERING
        self.fill(self.WHITE)

    def pix(self, length: float) -> int:
        """
        Convert a distance [m] to pixels [px].

        :param length: the input distance [m]
        :return: the corresponding size [px]
        """
        return math.ceil(length * self.scaling)

    def pos2pix(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert two world coordinates [m] into a position in the surface [px]
        The coordinates is changed to right-handed as well

        :param x: x world coordinate [m]
        :param y: y world coordinate [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pix(x - self.origin[0]), self.raw_size[-1] - self.pix(y - self.origin[1])

    def vec2pix(self, vec: PositionType) -> Tuple[int, int]:
        """
        Convert a world position [m] into a position in the surface [px].

        :param vec: a world position [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pos2pix(vec[0], vec[1])

    def is_visible(self, vec: PositionType, margin: int = 50) -> bool:
        """
        Is a position visible in the surface?
        :param vec: a position
        :param margin: margins around the frame to test for visibility
        :return: whether the position is visible
        """
        x, y = self.vec2pix(vec)
        return -margin < x < self.get_width() + margin and -margin < y < self.get_height() + margin

    def move_display_window_to(self, position: PositionType) -> None:
        """
        Set the origin of the displayed area to center on a given world position.

        :param position: a world position [m]
        """
        self.origin = position - np.array(
            [
                self.centering_position[0] * self.get_width() / self.scaling,
                self.centering_position[1] * self.get_height() / self.scaling
            ]
        )

    def copy(self):
        ret = WorldSurface(size=self.raw_size, flags=self.raw_flags, surf=self.raw_surface)
        ret.origin = self.origin
        ret.scaling = self.scaling
        ret.centering_position = self.centering_position
        ret.blit(self, (0, 0))
        return ret

    @staticmethod
    def to_cv2_image(surface):
        """
        convert the pygame.surface to image
        Args:
            surface: pygame.surface

        Returns: image in numpy array

        """
        img_array = np.array(pygame.surfarray.pixels3d(surface))
        image_object = np.transpose(img_array, (1, 0, 2))
        # image_object[:, :, [0, 2]] = image_object[:, :, [2, 0]]
        return image_object


class ObjectGraphics:
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN
    font = None

    @classmethod
    def display(
        cls,
        object: history_object,
        surface,
        color,
        heading,
        label: bool = False,
        draw_contour=False,
        contour_width=1
    ) -> None:
        """
        Display a vehicle on a pygame surface.

        The vehicle is represented as a colored rotated rectangle.

        :param object: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param label: whether a text label should be rendered
        """
        if not surface.is_visible(object.position):
            return
        w = surface.pix(object.WIDTH)
        h = surface.pix(object.LENGTH)
        position = [*surface.pos2pix(object.position[0], object.position[1])]
        # As the following rotate code is for left-handed coordinates,
        # so we plus -1 before the heading to adapt it to right-handed coordinates
        angle = -np.rad2deg(heading)
        box = [pygame.math.Vector2(p) for p in [(-h / 2, -w / 2), (-h / 2, w / 2), (h / 2, w / 2), (h / 2, -w / 2)]]
        box_rotate = [p.rotate(angle) + position for p in box]

        pygame.draw.polygon(surface, color, box_rotate)
        if draw_contour and pygame.ver.startswith("2"):
            pygame.draw.polygon(surface, cls.BLACK, box_rotate, width=contour_width)  # , 1)

        # Label
        if label:
            if cls.font is None:
                cls.font = pygame.font.Font(None, 15)
            text = "#{}".format(id(object) % 1000)
            text = cls.font.render(text, 1, (10, 10, 10), (255, 255, 255))
            surface.blit(text, position)

    @classmethod
    def get_color(cls, vehicle) -> Tuple[int]:
        if vehicle.crashed:
            color = cls.RED
        else:
            color = cls.BLUE
        return color


class LaneGraphics:
    """A visualization of a lane."""

    STRIPE_SPACING: float = 5
    """ Offset between stripes [m]"""

    STRIPE_LENGTH: float = 3
    """ Length of a stripe [m]"""

    STRIPE_WIDTH: float = 0.3
    """ Width of a stripe [m]"""

    LANE_LINE_WIDTH: float = 1

    @classmethod
    def display(cls, lane, surface, two_side=True, use_line_color=False, color=None) -> None:
        """
        Display a lane on a surface.

        :param lane: the lane to be displayed
        :param surface: the pygame surface
        :param two_side: draw two sides of the lane, or only one side
        """
        side = 2 if two_side else 1
        stripes_count = int(2 * (surface.get_height() + surface.get_width()) / (cls.STRIPE_SPACING * surface.scaling))
        s_origin, _ = lane.local_coordinates(surface.origin)
        s0 = (int(s_origin) // cls.STRIPE_SPACING - stripes_count // 2) * cls.STRIPE_SPACING
        for side in range(side):
            if use_line_color:
                if lane.line_colors[side] == PGLineColor.YELLOW and lane.line_types[side] == PGLineType.CONTINUOUS:
                    color = (255, 175, 35)
                elif lane.line_types[side] == PGLineType.SIDE:
                    color = (95, 95, 95)
                else:
                    color = (175, 175, 175)
            if lane.line_types[side] == PGLineType.BROKEN:
                cls.striped_line(lane, surface, stripes_count, s0, side, color=color)
            # circular side or continuous, it is same now
            elif lane.line_types[side] == PGLineType.CONTINUOUS and isinstance(lane, CircularLane):
                cls.continuous_curve(lane, surface, stripes_count, s0, side, color=color)
            elif (lane.line_types[side] == PGLineType.SIDE
                  or lane.line_types[side] == PGLineType.GUARDRAIL) and isinstance(lane, CircularLane):
                cls.continuous_curve(lane, surface, stripes_count, s0, side, color=color)
            # the line of continuous straight and side straight is same now
            elif (lane.line_types[side] == PGLineType.CONTINUOUS) and isinstance(lane, StraightLane):
                cls.continuous_line(lane, surface, stripes_count, s0, side, color=color)
            elif (lane.line_types[side] == PGLineType.SIDE
                  or lane.line_types[side] == PGLineType.GUARDRAIL) and isinstance(lane, StraightLane):
                cls.continuous_line(lane, surface, stripes_count, s0, side, color=color)
            # special scenario
            elif lane.line_types[side] == PGLineType.NONE:
                continue
            else:
                raise ValueError("I don't know how to draw this line type: {}".format(lane.line_types[side]))

    @classmethod
    def display_scenario_line(cls, polyline, type, surface, line_sample_interval=2) -> None:
        """
        Display a lane on a surface.

        :param lane: the lane to be displayed
        :param surface: the pygame surface
        :param two_side: draw two sides of the lane, or only one side
        """
        if MetaDriveType.is_yellow_line(type):
            color = (255, 175, 35)
        elif MetaDriveType.is_road_boundary_line(type):
            color = (100, 100, 100)
        else:
            color = (175, 175, 175)
        if MetaDriveType.is_road_line(type) or MetaDriveType.is_road_boundary_line(type):
            # if len(waymo_poly_line.segment_property) < 1:
            #     return
            if MetaDriveType.is_broken_line(type):
                points_to_skip = math.floor(PGDrivableAreaProperty.STRIPE_LENGTH * 2 / line_sample_interval) * 2
            else:
                points_to_skip = 1
            for index in range(0, len(polyline) - 1, points_to_skip):
                if index + 1 < len(polyline):
                    s_p = polyline[index]
                    e_p = polyline[index + 1]
                    pygame.draw.line(
                        surface,
                        color,
                        surface.vec2pix([s_p[0], s_p[1]]),
                        surface.vec2pix([e_p[0], e_p[1]]),
                        # max(surface.pix(LaneGraphics.STRIPE_WIDTH),
                        surface.pix(PGDrivableAreaProperty.LANE_LINE_WIDTH) * 2
                    )
        elif type == "center_lane" or type is None:
            pass

    @classmethod
    def striped_line(cls, lane, surface, stripes_count: int, longitudinal: float, side: int, color=None) -> None:
        """
        Draw a striped line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes to draw
        :param longitudinal: the longitudinal position of the first stripe [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        starts = longitudinal + np.arange(stripes_count) * cls.STRIPE_SPACING
        ends = longitudinal + np.arange(stripes_count) * cls.STRIPE_SPACING + cls.STRIPE_LENGTH
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats, color=color)

    @classmethod
    def continuous_curve(cls, lane, surface, stripes_count: int, longitudinal: float, side: int, color=None) -> None:
        """
        Draw a striped line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes to draw
        :param longitudinal: the longitudinal position of the first stripe [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        starts = longitudinal + np.arange(stripes_count) * cls.STRIPE_SPACING
        ends = longitudinal + np.arange(stripes_count) * cls.STRIPE_SPACING + cls.STRIPE_SPACING
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats, color=color)

    @classmethod
    def continuous_line(cls, lane, surface, stripes_count: int, longitudinal: float, side: int, color=None) -> None:
        """
        Draw a continuous line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes that would be drawn if the line was striped
        :param longitudinal: the longitudinal position of the start of the line [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        starts = [longitudinal + 0 * cls.STRIPE_SPACING]
        ends = [longitudinal + stripes_count * cls.STRIPE_SPACING + cls.STRIPE_LENGTH]
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats, color=color)

    @classmethod
    def draw_stripes(cls, lane, surface, starts: List[float], ends: List[float], lats: List[float], color=None) -> None:
        """
        Draw a set of stripes along a lane.

        :param lane: the lane
        :param surface: the surface to draw on
        :param starts: a list of starting longitudinal positions for each stripe [m]
        :param ends: a list of ending longitudinal positions for each stripe [m]
        :param lats: a list of lateral positions for each stripe [m]
        """
        if color is None:
            color = surface.LANE_LINE_COLOR
        starts = np.clip(starts, 0, lane.length)
        ends = np.clip(ends, 0, lane.length)
        for k, _ in enumerate(starts):
            if abs(starts[k] - ends[k]) > 0.5 * cls.STRIPE_LENGTH:
                pygame.draw.line(
                    surface, color, (surface.vec2pix(lane.position(starts[k], lats[k]))),
                    (surface.vec2pix(lane.position(ends[k], lats[k]))),
                    surface.pix(2 * PGDrivableAreaProperty.LANE_LINE_WIDTH)
                )

    @classmethod
    def draw_drivable_area(cls, lane, surface, color=(255, 255, 255)):
        from metadrive.component.pgblock.pg_block import PGBlock
        segment_num = int(lane.length / PGBlock.LANE_SEGMENT_LENGTH)
        width = lane.width
        for segment in range(segment_num):
            p_1 = lane.position(segment * PGBlock.LANE_SEGMENT_LENGTH, -width / 2)
            p_2 = lane.position(segment * PGBlock.LANE_SEGMENT_LENGTH, width / 2)
            p_3 = lane.position((segment + 1) * PGBlock.LANE_SEGMENT_LENGTH, width / 2)
            p_4 = lane.position((segment + 1) * PGBlock.LANE_SEGMENT_LENGTH, -width / 2)
            pygame.draw.polygon(
                surface, color,
                [surface.pos2pix(*p_1),
                 surface.pos2pix(*p_2),
                 surface.pos2pix(*p_3),
                 surface.pos2pix(*p_4)]
            )

        # # for last part
        p_1 = lane.position(segment_num * PGBlock.LANE_SEGMENT_LENGTH, -width / 2)
        p_2 = lane.position(segment_num * PGBlock.LANE_SEGMENT_LENGTH, width / 2)
        p_3 = lane.position(lane.length, width / 2)
        p_4 = lane.position(lane.length, -width / 2)
        pygame.draw.polygon(
            surface, color,
            [surface.pos2pix(*p_1),
             surface.pos2pix(*p_2),
             surface.pos2pix(*p_3),
             surface.pos2pix(*p_4)]
        )


class ObservationWindowMultiChannel:
    CHANNEL_NAMES = ["road_network", "traffic_flow", "target_vehicle", "past_pos"]

    def __init__(self, names, max_range, resolution):
        assert isinstance(names, list)
        assert set(self.CHANNEL_NAMES)
        self.sub_observations = {
            k: ObservationWindow(max_range=max_range, resolution=resolution)
            for k in ["traffic_flow", "target_vehicle"]
        }
        self.sub_observations["road_network"] = ObservationWindow(
            max_range=max_range,
            resolution=(resolution[0] * 2, resolution[1] * 2)
            # max_range=max_range, resolution=resolution
        )

        self.resolution = (resolution[0] * 2, resolution[1] * 2)
        self.canvas_display = None

    def get_canvas_display(self):
        if self.canvas_display is None:
            self.canvas_display = pygame.Surface(self.resolution)
        self.canvas_display.fill(COLOR_BLACK)
        return self.canvas_display

    def reset(self, canvas_runtime):
        for k, sub in self.sub_observations.items():
            sub.reset(canvas_runtime)

    def render(self, canvas_dict, position, heading):
        assert isinstance(canvas_dict, dict)
        assert set(canvas_dict.keys()) == set(self.sub_observations.keys())
        ret = dict()
        for k, canvas in canvas_dict.items():
            ret[k] = self.sub_observations[k].render(canvas, position, heading)
        return self.get_observation_window(ret)

    def get_observation_window(self, canvas_dict=None):
        if canvas_dict is None:
            canvas_dict = {k: v.get_observation_window() for k, v in self.sub_observations.items()}
        return canvas_dict

    def get_size(self):
        return next(iter(self.sub_observations.values())).get_size()

    def get_screen_window(self):
        canvas = self.get_canvas_display()
        ret = self.get_observation_window()

        for k in ret.keys():
            if k == "road_network":
                continue
            ret[k] = pygame.transform.scale2x(ret[k])

        def _draw(canvas, key, color):
            mask = pygame.mask.from_threshold(ret[key], (0, 0, 0, 0), (10, 10, 10, 255))
            mask.to_surface(canvas, setcolor=None, unsetcolor=color)

        if "navigation" in ret:
            _draw(canvas, "navigation", pygame.Color("Blue"))
        _draw(canvas, "road_network", pygame.Color("White"))
        _draw(canvas, "traffic_flow", pygame.Color("Red"))
        _draw(canvas, "target_vehicle", pygame.Color("Green"))
        return canvas
