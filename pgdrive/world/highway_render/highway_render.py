import os
from typing import List, Tuple

import numpy as np
from pgdrive.scene_creator.basic_utils import Decoration
from pgdrive.scene_creator.lanes.circular_lane import CircularLane
from pgdrive.scene_creator.lanes.lane import LineType
from pgdrive.scene_creator.lanes.straight_lane import StraightLane
from pgdrive.utils import import_pygame
from pgdrive.world.constants import PG_EDITION
from pgdrive.world.highway_render.world_surface import WorldSurface

pygame = import_pygame()


class HighwayRender:
    """
    Most of the source code is from Highway-Env, we only optimize and integrate it in PG-Drive
    See more information on its Github page: https://github.com/eleurent/highway-env
    """
    RESOLUTION = (200, 200)  # pix x pix
    MAP_RESOLUTION = (2000, 2000)  # pix x pix

    # CAM_REGION = 100  # 50m x (50m * HEIGHT/WIDTH)
    # FPS = 60
    # ROTATE = True

    def __init__(self, onscreen: bool, main_window_position=None):
        self.resolution = self.RESOLUTION
        self.frame_surface = None
        self._scaling = None  # automatically change, don't set the value
        self._center_pos = None  # automatically change, don't set the value
        self.onscreen = onscreen

        # map
        self.map = None
        self.map_surface = None

        # traffic
        self.scene_mgr = None

        pygame.init()
        pygame.display.set_caption(PG_EDITION + " (Top-down)")
        self.clock = None
        # if self.onscreen:
        # main_window_position means the left upper location.
        os.environ['SDL_VIDEO_WINDOW_POS'] = \
            '%i,%i' % (main_window_position[0] - self.RESOLUTION[0], main_window_position[1])
        self.screen = pygame.display.set_mode(self.resolution)
        self.clock = pygame.time.Clock()

        self.surface = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.frame_surface = pygame.Surface(self.RESOLUTION)

    def render(self) -> np.ndarray:
        # for event in pygame.event.get():
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_j:
        #             self.CAM_REGION -= 5 if 10 < self.CAM_REGION else 0
        #         if event.key == pygame.K_k:
        #             self.CAM_REGION += 5 if self.CAM_REGION < 100 else 0
        #         if event.key == pygame.K_ESCAPE:
        #             sys.exit()

        self.draw_scene()
        self.screen.fill(pygame.Color("black"))
        self.screen.blit(self.frame_surface, (0, 0))
        # if self.clock is not None:
        #     self.clock.tick(self.FPS)
        pygame.display.flip()

    def set_scene_mgr(self, scene_mgr):
        self.scene_mgr = scene_mgr

    def set_map(self, map):
        """
        Initialize the most big map surface to save calculation time, this func is called in map class automatically
        :param map: Map class
        :return: None
        """
        self.map = map
        self.draw_map()

    def draw_map(self) -> pygame.Surface:
        """
        :return: a big map surface, clip  and rotate to use a piece of it
        """
        surface = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        surface.set_colorkey(surface.BLACK)
        b_box = self.map.get_map_bound_box(self.map.road_network)
        x_len = b_box[1] - b_box[0]
        y_len = b_box[3] - b_box[2]
        max_len = max(x_len, y_len)
        # scaling and center can be easily found by bounding box
        scaling = self.MAP_RESOLUTION[1] / max_len - 0.1
        surface.scaling = scaling
        self._scaling = scaling
        centering_pos = ((b_box[0] + b_box[1]) / 2, (b_box[2] + b_box[3]) / 2)
        self._center_pos = centering_pos
        surface.move_display_window_to(centering_pos)
        for _from in self.map.road_network.graph.keys():
            decoration = True if _from == Decoration.start else False
            for _to in self.map.road_network.graph[_from].keys():
                for l in self.map.road_network.graph[_from][_to]:
                    two_side = True if l is self.map.road_network.graph[_from][_to][-1] or decoration else False
                    LaneGraphics.LANE_LINE_WIDTH = 0.5
                    LaneGraphics.display(l, surface, two_side)
        self.map_surface = surface

    def draw_scene(self):
        self.surface.fill(pygame.Color("black"))
        surface = self.surface
        surface.scaling = self._scaling
        surface.move_display_window_to(self._center_pos)
        surface.blit(self.map_surface, (0, 0))
        VehicleGraphics.display(self.scene_mgr.ego_vehicle, surface)
        for v in self.scene_mgr.vehicles:
            if v is self.scene_mgr.ego_vehicle:
                continue
            VehicleGraphics.display(v, surface)
        pos = surface.pos2pix(*self.scene_mgr.ego_vehicle.position)
        width = self.MAP_RESOLUTION[0] / 2
        height = width * self.RESOLUTION[1] / self.RESOLUTION[0]

        scale_surface = pygame.Surface((width, height))
        scale_surface.blit(surface, (0, 0), (pos[0] - width / 2, pos[1] - height / 2, width, height))
        #  scale the frame surface to window size
        rotate_surface = pygame.Surface((width, height))
        self.blit_rotate(
            rotate_surface,
            scale_surface, (width / 2, height / 2),
            angle=np.rad2deg(self.scene_mgr.ego_vehicle.heading_theta) + 90
        )

        final_cut_surface = pygame.Surface((width / 2, height / 2))
        final_cut_surface.blit(
            rotate_surface, (0, 0),
            (rotate_surface.get_width() / 4, rotate_surface.get_height() / 4, width / 2, height / 2)
        )
        pygame.transform.scale(final_cut_surface, self.RESOLUTION, self.frame_surface)

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


class VehicleGraphics(object):
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN

    @classmethod
    def display(cls, vehicle, surface, offscreen: bool = False, label: bool = False) -> None:
        """
        Display a vehicle on a pygame surface.

        The vehicle is represented as a colored rotated rectangle.

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param offscreen: whether the rendering should be done offscreen or not
        :param label: whether a text label should be rendered
        """
        from pgdrive.scene_creator.ego_vehicle.base_vehicle import BaseVehicle
        if not surface.is_visible(vehicle.position):
            return

        v = vehicle
        tire_length, tire_width = 1, 0.3

        # Vehicle rectangle
        length = v.LENGTH + 2 * tire_length
        vehicle_surface = pygame.Surface(
            (surface.pix(length), surface.pix(length)), flags=pygame.SRCALPHA
        )  # per-pixel alpha
        rect = (
            surface.pix(tire_length), surface.pix(length / 2 - v.WIDTH / 2), surface.pix(v.LENGTH),
            surface.pix(v.WIDTH)
        )
        pygame.draw.rect(vehicle_surface, cls.BLUE if not isinstance(vehicle, BaseVehicle) else cls.GREEN, rect, 0)
        pygame.draw.rect(vehicle_surface, cls.BLACK, rect, 1)

        # Centered rotation
        if not isinstance(vehicle, BaseVehicle):
            h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
        else:
            h = v.heading_theta if abs(v.heading_theta) > 2 * np.pi / 180 else 0
        position = [*surface.pos2pix(v.position[0], v.position[1])]
        if not offscreen:
            # convert_alpha throws errors in offscreen mode
            # see https://stackoverflow.com/a/19057853
            vehicle_surface = pygame.Surface.convert_alpha(vehicle_surface)
        cls.blit_rotate(surface, vehicle_surface, position, np.rad2deg(-h))

        # Label
        if label:
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


class LaneGraphics(object):
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
            if lane.line_types[side] == LineType.STRIPED:
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


class RoadGraphics(object):
    """A visualization of a road lanes."""
    @staticmethod
    def display(road, surface):
        """
        Display the road lanes on a surface.

        :param road: the road to be displayed
        :param surface: the pygame surface
        """
        surface.fill(surface.GREY)
        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for l in road.network.graph[_from][_to]:
                    LaneGraphics.display(l, surface)

    @staticmethod
    def display_traffic(road, surface, simulation_frequency: int = 15,
                        offscreen: bool = False) \
            -> None:
        """
        Display the road vehicles on a surface.

        :param road: the road to be displayed
        :param surface: the pygame surface
        :param simulation_frequency: simulation frequency
        :param offscreen: render without displaying on a screen
        """
        if road.record_history:
            for v in road.vehicles:
                VehicleGraphics.display_history(v, surface, simulation=simulation_frequency, offscreen=offscreen)
        for v in road.vehicles:
            VehicleGraphics.display(v, surface, offscreen=offscreen)
