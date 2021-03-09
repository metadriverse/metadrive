import os
import sys
from typing import List, Tuple

import numpy as np

from pgdrive.scene_creator.lane.abs_lane import LineType
from pgdrive.scene_creator.lane.circular_lane import CircularLane
from pgdrive.scene_creator.lane.straight_lane import StraightLane
from pgdrive.utils import import_pygame
from pgdrive.utils.constans import Decoration
from pgdrive.world.constants import PG_EDITION
from pgdrive.world.highway_render.world_surface import WorldSurface

pygame = import_pygame()

COLOR_BLACK = pygame.Color("black")


class ObservationWindow:
    def __init__(self, max_range, resolution):
        self.max_range = max_range
        self.resolution = resolution
        self.receptive_field = None
        self.receptive_field_double = None

        self.canvas_rotate = None
        self.canvas_unscaled = None
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
        self.canvas_unscaled = pygame.Surface(self.receptive_field)
        self.canvas_unscaled.fill(COLOR_BLACK)
        self.canvas_display.fill(COLOR_BLACK)

    def render(self, canvas, position, heading):
        # Prepare a runtime canvas for rotation
        # Assume max_range is only the radius!
        self.canvas_rotate.blit(
            canvas, (0, 0), (
                position[0] - self.receptive_field_double[0] / 2, position[1] - self.receptive_field_double[1] / 2,
                self.receptive_field_double[0], self.receptive_field_double[1]
            )
        )

        # Rotate the image so that ego is always heading top
        rotation = np.rad2deg(heading) + 90
        new_canvas = pygame.transform.rotate(self.canvas_rotate, rotation)
        # new_canvas = pygame.transform.rotozoom(self.canvas_rotate, rotation, 1)  # Optional choice! Not so efficient!

        # Crop the rotated image and then resize to the desired resolution
        self.canvas_unscaled.blit(
            new_canvas,
            (0, 0),
            (
                new_canvas.get_size()[0] / 2 - self.receptive_field[0] / 2,  # Left
                new_canvas.get_size()[1] / 2 - self.receptive_field[1] / 2,  # Top
                self.receptive_field[0],  # Width
                self.receptive_field[1]  # Height
            )
        )
        pygame.transform.smoothscale(self.canvas_unscaled, self.canvas_display.get_size(), self.canvas_display)

        # print(
        #     "Current canvas size, canvas display size",
        #     new_canvas.get_size(),
        #     canvas.get_size(),
        #     self.receptive_field,
        #     self.receptive_field_double,
        #     canvas.get_size(),
        #     self.canvas_display.get_size()
        # )
        return self.canvas_display

    def get_observation_window(self):
        return self.canvas_display

    def get_size(self):
        assert self.canvas_rotate is not None
        return self.canvas_rotate.get_size()


class HighwayRender:
    """
    Most of the source code is from Highway-Env, we only optimize and integrate it in PGDrive
    See more information on its Github page: https://github.com/eleurent/highway-env
    """
    RESOLUTION = (200, 200)  # pix x pix
    MAP_RESOLUTION = (2000, 2000)  # pix x pix
    MAX_RANGE = (50, 50)  # maximum detection distance = 50 M

    # CAM_REGION = 100  # 50m x (50m * HEIGHT/WIDTH)
    # FPS = 60
    # ROTATE = True

    def __init__(self, onscreen: bool, main_window_position=None):
        self.resolution = self.RESOLUTION
        self.onscreen = onscreen

        self._center_pos = None  # automatically change, don't set the value
        self._should_draw_map = True
        self._canvas_to_display_scaling = 0.0

        # scene
        self.road_network = None
        self.scene_mgr = None

        # initialize
        pygame.init()
        pygame.display.set_caption(PG_EDITION + " (Top-down)")
        # main_window_position means the left upper location.
        os.environ['SDL_VIDEO_WINDOW_POS'] = \
            '%i,%i' % (main_window_position[0] - self.RESOLUTION[0], main_window_position[1])
        self.screen = pygame.display.set_mode(self.resolution) if onscreen else None  # Used for display only!

        # canvas
        self.canvas_runtime = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_background = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.obs_window = ObservationWindow(self.MAX_RANGE, self.RESOLUTION)

    def set_scene_mgr(self, scene_mgr):
        self.scene_mgr = scene_mgr

    def set_map(self, map):
        """
        Initialize the most big map surface to save calculation time, this func is called in map class automatically
        :param map: Map class
        :return: None
        """
        self.road_network = map.road_network
        self._should_draw_map = True

    def render(self) -> np.ndarray:
        if self.onscreen:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        sys.exit()

        if self._should_draw_map:
            self.draw_map()

        self.draw_scene()

        if self.onscreen:
            self.screen.fill(COLOR_BLACK)
            self.screen.blit(self.obs_window.get_observation_window(), (0, 0))
            pygame.display.flip()

    def get_screenshot(self, name="screenshot.jpg"):
        pygame.image.save(self.screen, name)

    def draw_map(self) -> pygame.Surface:
        """
        :return: a big map surface, clip  and rotate to use a piece of it
        """
        surface = self.canvas_background

        # Setup the maximize size of the canvas
        # scaling and center can be easily found by bounding box
        b_box = self.road_network.get_bounding_box()
        self.canvas_background.set_colorkey(self.canvas_background.BLACK)
        x_len = b_box[1] - b_box[0]
        y_len = b_box[3] - b_box[2]
        max_len = max(x_len, y_len) + 20  # Add more 20 meters
        scaling = self.MAP_RESOLUTION[1] / max_len - 0.1
        assert scaling > 0

        # real-world distance * scaling = pixel in canvas
        self.canvas_background.scaling = scaling
        self.canvas_runtime.scaling = scaling
        # self._scaling = scaling

        centering_pos = ((b_box[0] + b_box[1]) / 2, (b_box[2] + b_box[3]) / 2)
        # self._center_pos = centering_pos
        self.canvas_runtime.move_display_window_to(centering_pos)

        surface.move_display_window_to(centering_pos)
        for _from in self.road_network.graph.keys():
            decoration = True if _from == Decoration.start else False
            for _to in self.road_network.graph[_from].keys():
                for l in self.road_network.graph[_from][_to]:
                    two_side = True if l is self.road_network.graph[_from][_to][-1] or decoration else False
                    LaneGraphics.LANE_LINE_WIDTH = 0.5
                    LaneGraphics.display(l, self.canvas_background, two_side)

        self.obs_window.reset(self.canvas_runtime)

        self._should_draw_map = False

    def draw_scene(self):
        # Set the active area that can be modify to accelerate
        pos = self.canvas_runtime.pos2pix(*self.scene_mgr.ego_vehicle.position)
        clip_size = (int(self.obs_window.get_size()[0] * 1.1), int(self.obs_window.get_size()[0] * 1.1))
        self.canvas_runtime.set_clip((pos[0] - clip_size[0] / 2, pos[1] - clip_size[1] / 2, clip_size[0], clip_size[1]))
        self.canvas_runtime.fill(COLOR_BLACK)
        self.canvas_runtime.blit(self.canvas_background, (0, 0))

        # Draw vehicles
        VehicleGraphics.display(self.scene_mgr.ego_vehicle, self.canvas_runtime)
        for v in self.scene_mgr.traffic_mgr.vehicles:
            if v is self.scene_mgr.ego_vehicle:
                continue
            VehicleGraphics.display(v, self.canvas_runtime)

        # Prepare a runtime canvas for rotation
        return self.obs_window.render(
            canvas=self.canvas_runtime, position=pos, heading=self.scene_mgr.ego_vehicle.heading_theta
        )

    @staticmethod
    def blit_rotate(
        surf: pygame.SurfaceType,
        image: pygame.SurfaceType,
        pos,
        angle: float,
    ) -> Tuple:
        """Many thanks to https://stackoverflow.com/a/54714144."""
        # calculate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

        # calculate the translation of the pivot
        origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (
            pos[0] - origin_pos[0] + min_box[0] - pivot_move[0], pos[1] - origin_pos[1] - max_box[1] + pivot_move[1]
        )
        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        # rotate and blit the image
        surf.blit(rotated_image, origin)
        return origin

    def get_observation_window(self):
        return self.obs_window.get_observation_window()


class VehicleGraphics:
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN

    @classmethod
    def display(cls, vehicle, surface, label: bool = False) -> None:
        """
        Display a vehicle on a pygame surface.

        The vehicle is represented as a colored rotated rectangle.

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param label: whether a text label should be rendered
        """
        from pgdrive.scene_creator.ego_vehicle.base_vehicle import BaseVehicle
        if not surface.is_visible(vehicle.position):
            return

        v = vehicle
        tire_length, tire_width = 1, 0.3

        # Vehicle rectangle
        length = v.LENGTH + 2 * tire_length

        # FIXME we should not initialize Surface here! Maintain a library of pre-generated surface in renderer!
        vehicle_surface = pygame.Surface(
            (surface.pix(length), surface.pix(length)), flags=pygame.SRCALPHA
        )  # per-pixel alpha
        rect = (
            surface.pix(tire_length), surface.pix(length / 2 - v.WIDTH / 2), surface.pix(v.LENGTH),
            surface.pix(v.WIDTH)
        )
        # TODO(pzh) remove the dependency here and pass in the color you like as input argument!
        pygame.draw.rect(vehicle_surface, cls.BLUE if not isinstance(vehicle, BaseVehicle) else cls.GREEN, rect, 0)

        # FIXME PZH: @LQY what this sentence used for?
        # FIXME LQY: @PZH It is from highway-env, Used to add a black exterior frame for vehicles
        # pygame.draw.rect(vehicle_surface, cls.BLACK, rect, 1)

        # Centered rotation
        if not isinstance(vehicle, BaseVehicle):
            h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
        else:
            h = v.heading_theta if abs(v.heading_theta) > 2 * np.pi / 180 else 0
        position = [*surface.pos2pix(v.position[0], v.position[1])]
        cls.blit_rotate(surface, vehicle_surface, position, np.rad2deg(-h))

        # Label
        if label:
            # TODO PZH: Do not initialize font here! Waste of time!
            font = pygame.font.Font(None, 15)
            text = "#{}".format(id(v) % 1000)
            text = font.render(text, 1, (10, 10, 10), (255, 255, 255))
            surface.blit(text, position)

    @staticmethod
    def blit_rotate(
        surf: pygame.SurfaceType,
        image: pygame.SurfaceType,
        pos,
        angle: float,
        origin_pos=None,
        show_rect: bool = False
    ) -> None:
        """Many thanks to https://stackoverflow.com/a/54714144."""
        # calculate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

        # calculate the translation of the pivot
        if origin_pos is None:
            origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (
            pos[0] - origin_pos[0] + min_box[0] - pivot_move[0], pos[1] - origin_pos[1] - max_box[1] + pivot_move[1]
        )
        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        # rotate and blit the image
        surf.blit(rotated_image, origin)
        # draw rectangle around the image
        if show_rect:
            pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()), 2)

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
    def display(cls, lane, surface, two_side=True) -> None:
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
            if lane.line_types[side] == LineType.BROKEN:
                cls.striped_line(lane, surface, stripes_count, s0, side)
            # circular side or continuous, it is same now
            elif lane.line_types[side] == LineType.CONTINUOUS and isinstance(lane, CircularLane):
                cls.continuous_curve(lane, surface, stripes_count, s0, side)
            elif lane.line_types[side] == LineType.SIDE and isinstance(lane, CircularLane):
                cls.continuous_curve(lane, surface, stripes_count, s0, side)
            # the line of continuous straight and side straight is same now
            elif (lane.line_types[side] == LineType.CONTINUOUS) and isinstance(lane, StraightLane):
                cls.continuous_line(lane, surface, stripes_count, s0, side)
            elif (lane.line_types[side] == LineType.SIDE) and isinstance(lane, StraightLane):
                cls.continuous_line(lane, surface, stripes_count, s0, side)
            # special case
            elif lane.line_types[side] == LineType.NONE:
                continue
            else:
                raise ValueError("I don't know how to draw this line type: {}".format(lane.line_types[side]))

    @classmethod
    def striped_line(cls, lane, surface, stripes_count: int, longitudinal: float, side: int) -> None:
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
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def continuous_curve(cls, lane, surface, stripes_count: int, longitudinal: float, side: int) -> None:
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
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def continuous_line(cls, lane, surface, stripes_count: int, longitudinal: float, side: int) -> None:
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
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def draw_stripes(cls, lane, surface, starts: List[float], ends: List[float], lats: List[float]) -> None:
        """
        Draw a set of stripes along a lane.

        :param lane: the lane
        :param surface: the surface to draw on
        :param starts: a list of starting longitudinal positions for each stripe [m]
        :param ends: a list of ending longitudinal positions for each stripe [m]
        :param lats: a list of lateral positions for each stripe [m]
        """
        starts = np.clip(starts, 0, lane.length)
        ends = np.clip(ends, 0, lane.length)
        for k, _ in enumerate(starts):
            if abs(starts[k] - ends[k]) > 0.5 * cls.STRIPE_LENGTH:
                pygame.draw.line(
                    surface, surface.WHITE, (surface.vec2pix(lane.position(starts[k], lats[k]))),
                    (surface.vec2pix(lane.position(ends[k], lats[k]))),
                    max(surface.pix(cls.STRIPE_WIDTH), surface.pix(cls.LANE_LINE_WIDTH))
                )

    @classmethod
    def simple_draw(cls, lane, surface, color=(255, 255, 255)):
        from pgdrive.scene_creator.blocks.block import Block
        segment_num = int(lane.length / Block.CIRCULAR_SEGMENT_LENGTH)
        width = lane.width
        for segment in range(segment_num):
            p_1 = lane.position(segment * Block.CIRCULAR_SEGMENT_LENGTH, -width / 2)
            p_2 = lane.position(segment * Block.CIRCULAR_SEGMENT_LENGTH, width / 2)
            p_3 = lane.position((segment + 1) * Block.CIRCULAR_SEGMENT_LENGTH, width / 2)
            p_4 = lane.position((segment + 1) * Block.CIRCULAR_SEGMENT_LENGTH, -width / 2)
            pygame.draw.polygon(
                surface, color,
                [surface.pos2pix(*p_1),
                 surface.pos2pix(*p_2),
                 surface.pos2pix(*p_3),
                 surface.pos2pix(*p_4)]
            )

        # # for last part
        p_1 = lane.position(segment_num * Block.CIRCULAR_SEGMENT_LENGTH, -width / 2)
        p_2 = lane.position(segment_num * Block.CIRCULAR_SEGMENT_LENGTH, width / 2)
        p_3 = lane.position(lane.length, width / 2)
        p_4 = lane.position(lane.length, -width / 2)
        pygame.draw.polygon(
            surface, color,
            [surface.pos2pix(*p_1),
             surface.pos2pix(*p_2),
             surface.pos2pix(*p_3),
             surface.pos2pix(*p_4)]
        )

    @classmethod
    def draw_ground(cls, lane, surface, color: Tuple[float], width: float, draw_surface: pygame.Surface = None) -> None:
        draw_surface = draw_surface or surface
        stripes_count = int(2 * (surface.get_height() + surface.get_width()) / (cls.STRIPE_SPACING * surface.scaling))
        s_origin, _ = lane.local_coordinates(surface.origin)
        s0 = (int(s_origin) // cls.STRIPE_SPACING - stripes_count // 2) * cls.STRIPE_SPACING
        dots = []
        for side in range(2):
            longis = np.clip(s0 + np.arange(stripes_count) * cls.STRIPE_SPACING, 0, lane.length)
            lats = [2 * (side - 0.5) * width for _ in longis]
            new_dots = [surface.vec2pix(lane.position(longi, lat)) for longi, lat in zip(longis, lats)]
            new_dots = reversed(new_dots) if side else new_dots
            dots.extend(new_dots)
        pygame.draw.polygon(draw_surface, color, dots, 0)
