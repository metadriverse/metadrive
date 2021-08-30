from collections import deque, namedtuple
from typing import Optional, Union, Iterable

import cv2
import numpy as np

from metadrive.constants import Decoration, TARGET_VEHICLES
from metadrive.obs.top_down_obs_impl import WorldSurface, VehicleGraphics, LaneGraphics
from metadrive.utils.utils import import_pygame

pygame = import_pygame()

color_white = (255, 255, 255)
history_vehicle = namedtuple("history_vehicle", "name position heading_theta WIDTH LENGTH color done")


def draw_top_down_map(
    map,
    resolution: Iterable = (512, 512),
    simple_draw=True,
    return_surface=False,
    film_size=None,
    reverse_color=False,
    road_color=color_white
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
                    LaneGraphics.simple_draw(l, surface, color=road_color)
                else:
                    two_side = True if l is map.road_network.graph[_from][_to][-1] or decoration else False
                    LaneGraphics.display(l, surface, two_side, color=road_color)

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

    if entry_differ_color:
        # init only once
        if "spawn_roads" in episode_data:
            spawn_roads = episode_data["spawn_roads"]
        else:
            spawn_roads = set()
            first_frame = episode_data["frame"][0]
            for state in first_frame[TARGET_VEHICLES].values():
                spawn_roads.add((state["spawn_road"][0], state["spawn_road"][1]))
        keys = [road[0] for road in list(spawn_roads)]
        keys.sort()
        color_map = {key: color for key, color in zip(keys, color_list)}

    for frame in episode_data["frame"]:
        for k, state, in frame[TARGET_VEHICLES].items():
            if color_type == 0:
                color = color_white
            elif color_type == 1:
                if exit_differ_color:
                    key = state["destination"][1]
                    if key not in color_map:
                        color_map[key] = color_list.pop()
                    color = color_map[key]
                else:
                    color = color_map[state["spawn_road"][0]]
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
    for step, frame in enumerate(episode_data["frame"]):
        for k, state in frame[TARGET_VEHICLES].items():
            if not state["done"]:
                continue
            start = state["position"]
            if state["done"]:
                pygame.draw.circle(surface, (0, 0, 0), surface.pos2pix(start[0], start[1]), 5)
    return surface


class TopDownRenderer:
    def __init__(
        self,
        env,
        map,
        film_size=None,
        screen_size=None,
        light_background=True,
        zoomin=None,
        num_stack=15,
        history_smooth=0,
        road_color=(255, 255, 255),
        show_agent_name=False,
        track=False
    ):
        self.follow_agent = track
        self.show_agent_name = show_agent_name
        if show_agent_name:
            pygame.init()
        self.pygame_font = None

        film_size = film_size or (1000, 1000)
        self._env = env
        self._zoomin = zoomin or 1.0
        self._screen_size = screen_size
        self.map = map
        self.stack_frames = deque(maxlen=num_stack)
        self.history_vehicles = deque(maxlen=num_stack)
        self.history_smooth = history_smooth

        self._background = draw_top_down_map(
            map, simple_draw=False, return_surface=True, film_size=film_size, road_color=road_color
        )
        self._film_size = self._background.get_size()
        self.road_color = road_color

        self._light_background = light_background
        if self._light_background:
            pixels = pygame.surfarray.pixels2d(self._background)
            pixels ^= 2**32 - 1
            del pixels

        self._runtime = self._background.copy()
        self._runtime_output = self._background.copy()

        # self._runtime.blit(self._background, (0, 0))
        self._size = tuple(self._background.get_size())

        self._screen = pygame.display.set_mode(self._screen_size if self._screen_size is not None else self._film_size)
        self.canvas = pygame.Surface(self._screen.get_size())

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
        self.canvas.fill((255, 255, 255))

    def render(self, vehicles, agent_manager, *args, **kwargs):
        self.refresh()
        this_frame_vehicles = self._append_frame_vehicles(vehicles, agent_manager)
        self.history_vehicles.append(this_frame_vehicles)
        self._draw_history_vehicles()
        self.blit()
        ret = self.canvas.copy()
        ret = ret.convert(24)
        return ret

    def blit(self):
        # if self._screen_size is None and self._zoomin is None:
        #     self._screen.blit(self._runtime, (0, 0))
        # else:
        #     self._screen.blit(
        #         pygame.transform.smoothscale(self._runtime, self._blit_size), (self._blit_rect[0], self._blit_rect[1])
        #     )

        self._screen.blit(self.canvas, (0, 0))

        pygame.display.update()

    def _append_frame_vehicles(self, vehicles, agent_manager):
        frame_vehicles = []
        # for i, v in enumerate(vehicles, 1):
        #     name = self._env.agent_manager.object_to_agent(v.name)

        for v in agent_manager._active_objects.values():
            name = agent_manager.object_to_agent(v.name)
            frame_vehicles.append(
                history_vehicle(
                    name=name,
                    heading_theta=v.heading_theta,
                    WIDTH=v.WIDTH,
                    LENGTH=v.LENGTH,
                    position=v.position,
                    color=v.top_down_color,
                    done=False
                )
            )

        for (v, _) in agent_manager._dying_objects.values():
            name = agent_manager.object_to_agent(v.name)
            frame_vehicles.append(
                history_vehicle(
                    name=name,
                    heading_theta=v.heading_theta,
                    WIDTH=v.WIDTH,
                    LENGTH=v.LENGTH,
                    position=v.position,
                    color=v.top_down_color,
                    done=True
                )
            )
        return frame_vehicles

    def _draw_history_vehicles(self):
        if len(self.history_vehicles) == 0:
            return
        for i, vehicles in enumerate(self.history_vehicles):
            i = len(self.history_vehicles) - i
            if self.history_smooth != 0 and (i % self.history_smooth != 0):
                continue
            for v in vehicles:
                c = v.color
                h = v.heading_theta
                h = h if abs(h) > 2 * np.pi / 180 else 0
                x = abs(int(i))
                alpha_f = x / len(self.history_vehicles)
                VehicleGraphics.display(
                    vehicle=v,
                    surface=self._runtime,
                    heading=h,
                    color=(c[0] + alpha_f * (255 - c[0]), c[1] + alpha_f * (255 - c[1]), c[2] + alpha_f * (255 - c[2])),
                    draw_countour=False
                )

        # i = int(len(self.history_vehicles) / 2)
        # i = int(len(self.history_vehicles)) - 1
        i = -1
        for v in self.history_vehicles[i]:
            h = v.heading_theta
            c = v.color
            h = h if abs(h) > 2 * np.pi / 180 else 0
            # x = abs(int(i))
            # alpha_f = x / len(self.history_vehicles)
            alpha_f = 0
            VehicleGraphics.display(
                vehicle=v,
                surface=self._runtime,
                heading=h,
                color=(c[0] + alpha_f * (255 - c[0]), c[1] + alpha_f * (255 - c[1]), c[2] + alpha_f * (255 - c[2])),
                draw_countour=True,
                contour_width=2
            )

        if not hasattr(self, "_deads"):
            self._deads = []

        for v in self._deads:
            pygame.draw.circle(
                self._runtime,
                (255, 0, 0),
                self._runtime.pos2pix(v.position[0], v.position[1]),
                # self._runtime.pix(v.WIDTH)
                5
            )

        for v in self.history_vehicles[i]:
            if v.done:
                pygame.draw.circle(
                    self._runtime,
                    (255, 0, 0),
                    self._runtime.pos2pix(v.position[0], v.position[1]),
                    # self._runtime.pix(v.WIDTH)
                    5
                )
                self._deads.append(v)

        # Tracking Vehicle
        # heading = 30
        # rotation = np.rad2deg(heading) + 90
        # heading = self._env.current_track_vehicle.heading_theta
        # rotation = np.rad2deg(heading) + 90

        if self.follow_agent:
            v = self._env.current_track_vehicle
            canvas = self._runtime
            field = self.canvas.get_width()
            position = self._runtime.pos2pix(*v.position)
            off = (position[0] - field / 2, position[1] - field / 2)
            self.canvas.blit(canvas, (0, 0), (off[0], off[1], field, field))
        else:
            self.canvas.blit(self._runtime, (0, 0))
            off = (0, 0)

        # heading = self._env.current_track_vehicle.heading_theta
        # rotation = np.rad2deg(heading) + 90
        # rotated = pygame.transform.rotate(self.canvas, rotation)
        # size = self.canvas.get_size()
        # new_canvas = rotated
        # self.canvas.blit(
        #     rotated,
        #     (0, 0),
        #     (
        #         new_canvas.get_size()[0] / 2 - size[0] / 2,  # Left
        #         new_canvas.get_size()[1] / 2 - size[1] / 2,  # Top
        #         size[0],  # Width
        #         size[1]  # Height
        #     )
        # )

        if self.show_agent_name:
            if self.pygame_font is None:
                self.pygame_font = pygame.font.SysFont("Arial.ttf", 30)
            for v in self.history_vehicles[i]:
                position = self._runtime.pos2pix(*v.position)
                new_position = (position[0] - off[0], position[1] - off[1])
                img = self.pygame_font.render(
                    v.name,
                    True,
                    (0, 0, 0, 128),
                    # (0, 255, 0, 230)
                    # None
                    # pygame.color.Color("black"),
                    # (255, 255, 255)
                )
                # img.set_alpha(None)
                self.canvas.blit(
                    img,
                    (new_position[0] - img.get_width() / 2, new_position[1] - img.get_height() / 2),
                    # special_flags=pygame.BLEND_RGBA_MULT
                )

    def close(self):
        pygame.quit()

    def reset(self, map):
        self._background = draw_top_down_map(
            map, simple_draw=False, return_surface=True, film_size=self._film_size, road_color=self.road_color
        )
        self._film_size = self._background.get_size()

        self._light_background = self._light_background
        if self._light_background:
            pixels = pygame.surfarray.pixels2d(self._background)
            pixels ^= 2**32 - 1
            del pixels

        self._runtime = self._background.copy()
        self._runtime_output = self._background.copy()

        # self._runtime.blit(self._background, (0, 0))
        self._size = tuple(self._background.get_size())
