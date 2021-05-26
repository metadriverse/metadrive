from typing import Optional, Union, Iterable

import cv2
import numpy as np
from pgdrive.constants import Decoration
from pgdrive.obs.top_down_obs_impl import WorldSurface, VehicleGraphics, LaneGraphics
from pgdrive.utils.utils import import_pygame

pygame = import_pygame()
from pgdrive.constants import TARGET_VEHICLES

color_white = (255, 255, 255)


def draw_top_down_map(
    map,
    resolution: Iterable = (512, 512),
    simple_draw=True,
    return_surface=False,
    film_size=None,
    reverse_color=False
) -> Optional[Union[np.ndarray, pygame.Surface]]:
    film_size = film_size or map.film_size
    surface = WorldSurface(film_size, 0, pygame.Surface(film_size))
    if reverse_color:
        surface.WHITE, surface.BLACK = surface.BLACK, surface.WHITE
        surface.__init__(film_size, 0, pygame.Surface(film_size))
    b_box = map.road_network.get_bounding_box()
    x_len = b_box[1] - b_box[0]
    y_len = b_box[3] - b_box[2]
    max_len = max(x_len, y_len)
    # scaling and center can be easily found by bounding box
    scaling = film_size[1] / max_len - 0.1
    surface.scaling = scaling
    centering_pos = ((b_box[0] + b_box[1]) / 2, (b_box[2] + b_box[3]) / 2)
    surface.move_display_window_to(centering_pos)
    for _from in map.road_network.graph.keys():
        decoration = True if _from == Decoration.start else False
        for _to in map.road_network.graph[_from].keys():
            for l in map.road_network.graph[_from][_to]:
                if simple_draw:
                    LaneGraphics.simple_draw(l, surface)
                else:
                    two_side = True if l is map.road_network.graph[_from][_to][-1] or decoration else False
                    LaneGraphics.display(l, surface, two_side)

    if return_surface:
        return surface
    ret = cv2.resize(pygame.surfarray.pixels_red(surface), resolution, interpolation=cv2.INTER_LINEAR)
    return ret


def draw_top_down_trajectory(
    surface: WorldSurface, episode_data: dict, entry_differ_color=False, exit_differ_color=False, color_list=None
):
    if entry_differ_color or exit_differ_color:
        assert color_list is not None
    color_map = {}
    if not exit_differ_color and not entry_differ_color:
        color_type = 0
    elif exit_differ_color ^ entry_differ_color:
        color_type = 1
    else:
        color_type = 2

    for frame in episode_data["frame"]:
        for k, state, in frame[TARGET_VEHICLES].items():
            if color_type == 0:
                color = color_white
            elif color_type == 1:
                key = state["destination"][1] if exit_differ_color else state["spawn_road"][0]
                if key not in color_map:
                    color_map[key] = color_list.pop()
                color = color_map[key]
            else:
                key_1 = state["spawn_road"][0]
                key_2 = state["destination"][1]
                if key_1 not in color_map:
                    color_map[key_1] = dict()
                if key_2 not in color_map[key_1]:
                    color_map[key_1][key_2] = color_list.pop()
                color = color_map[key_1][key_2]

            start = state["position"]
            pygame.draw.circle(surface, color, surface.pos2pix(start[0], start[1]), 1)
    return surface


class TopDownRenderer:
    def __init__(self, map, film_size=None, screen_size=None, light_background=True, zoomin=None):
        film_size = film_size or (1000, 1000)
        self._zoomin = zoomin or 1.0
        self._screen_size = screen_size
        self._map = map

        self._background = draw_top_down_map(map, simple_draw=False, return_surface=True, film_size=film_size)
        self._film_size = self._background.get_size()

        self._light_background = light_background
        if self._light_background:
            pixels = pygame.surfarray.pixels2d(self._background)
            pixels ^= 2**32 - 1
            del pixels

        self._runtime = self._background.copy()

        # self._runtime.blit(self._background, (0, 0))
        self._size = tuple(self._background.get_size())
        self._screen = pygame.display.set_mode(self._screen_size if self._screen_size is not None else self._film_size)
        self._screen.set_alpha(None)
        self._screen.fill(color_white)

        screen_size = self._screen_size or self._film_size
        self._blit_size = (int(screen_size[0] * self._zoomin), int(screen_size[1] * self._zoomin))
        self._blit_rect = (
            -(self._blit_size[0] - screen_size[0]) / 2, -(self._blit_size[1] - screen_size[1]) / 2, screen_size[0],
            screen_size[1]
        )
        self.blit()

    def refresh(self):
        # self._runtime.blit(self._background, self._blit_rect)
        self._runtime.blit(self._background, (0, 0))

    def render(self, vehicles, *args, **kwargs):
        self.refresh()
        self._draw_vehicles(vehicles)
        self.blit()

    def blit(self):
        if self._screen_size is None and self._zoomin is None:
            self._screen.blit(self._runtime, (0, 0))
        else:
            self._screen.blit(
                pygame.transform.smoothscale(self._runtime, self._blit_size), (self._blit_rect[0], self._blit_rect[1])
            )
        pygame.display.update()

    def _draw_vehicles(self, vehicles):
        for v in vehicles:
            h = v.heading_theta
            h = h if abs(h) > 2 * np.pi / 180 else 0

            VehicleGraphics.display(
                vehicle=v, surface=self._runtime, heading=h, color=VehicleGraphics.BLUE, draw_countour=True
            )


class PheromoneRenderer(TopDownRenderer):
    def __init__(self, map, film_size=(2000, 2000), screen_size=(1000, 1000), zoomin=1.3, draw_vehicle_first=False):
        super(PheromoneRenderer, self).__init__(
            map, film_size=film_size, screen_size=screen_size, light_background=True, zoomin=zoomin
        )
        self._pheromone_surface = None
        self._bounding_box = self._map.road_network.get_bounding_box()
        self._draw_vehicle_first = draw_vehicle_first
        self._color_map = None

    def render(self, vehicles, pheromone_map):
        self.refresh()

        # It is also OK to draw pheromone first! But we should wait a while until the vehicles leave the cells to
        # see them!
        if self._draw_vehicle_first:
            self._draw_vehicles(vehicles)
            self._draw_pheromone_map(pheromone_map)
        else:
            self._draw_pheromone_map(pheromone_map)
            self._draw_vehicles(vehicles)
        self.blit()

    def _draw_pheromone_map(self, pheromone_map):
        phero = pheromone_map.get_map(*self._bounding_box)
        if self._pheromone_surface is None:
            self._pheromone_surface = pygame.Surface(phero.shape[:2])

        if self._color_map is None:
            color_map = np.zeros(phero.shape[:2] + (3, ), dtype=np.uint8)
            color_map[0:100, :70] = (255, 150, 255)
            color_map[100:120, :70] = (155, 92, 155)
            color_map[120:140, :70] = (55, 32, 55)
            color_map[140:160, :70] = (5, 3, 5)
            self._color_map = color_map

        phero = np.squeeze(phero, -1)

        mask = phero > 0
        # self._color_map.fill(0)
        self._color_map[160:, :] = 0
        self._color_map[:, 70:] = 0

        self._color_map[mask, 0] = 255 * phero[mask]
        self._color_map[mask, 1] = 150 * phero[mask]
        self._color_map[mask, 2] = 255 * phero[mask]

        pygame.surfarray.blit_array(self._pheromone_surface, self._color_map)
        tmp = pygame.transform.scale(self._pheromone_surface, self._background.get_size())
        self._runtime.blit(
            tmp, (0, 0), special_flags=pygame.BLEND_RGB_SUB if self._light_background else pygame.BLEND_RGB_ADD
        )
