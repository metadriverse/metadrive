import copy
from metadrive.utils.interpolating_line import InterpolatingLine

from collections import deque, namedtuple
import copy
from typing import Optional, Union, Iterable

import cv2
import numpy as np

from metadrive.component.map.waymo_map import WaymoMap
from metadrive.constants import Decoration, TARGET_VEHICLES
from metadrive.constants import LineColor
from metadrive.constants import WaymoLaneProperty

from metadrive.engine.engine_utils import get_engine
from metadrive.obs.top_down_obs_impl import WorldSurface, VehicleGraphics, LaneGraphics
from metadrive.utils.map_utils import is_map_related_instance
from metadrive.utils.utils import import_pygame
from metadrive.utils.waymo_utils.waymo_utils import RoadLineType, RoadEdgeType, convert_polyline_to_metadrive

pygame = import_pygame()

color_white = (255, 255, 255)
history_object = namedtuple("history_object", "name position heading_theta WIDTH LENGTH color done")


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

    if isinstance(map, WaymoMap):
        assert not simple_draw, "Simple Draw does not support now"
        for data in map.blocks[-1].waymo_map_data.values():
            if WaymoLaneProperty.POLYLINE not in data:
                continue
            type = data.get("type", None)
            waymo_line = InterpolatingLine(convert_polyline_to_metadrive(data[WaymoLaneProperty.POLYLINE]))
            LaneGraphics.display_waymo(waymo_line, type, surface)
    else:
        for _from in map.road_network.graph.keys():
            decoration = True if _from == Decoration.start else False
            for _to in map.road_network.graph[_from].keys():
                for l in map.road_network.graph[_from][_to]:
                    if simple_draw:
                        LaneGraphics.simple_draw(l, surface, color=road_color)
                    else:
                        two_side = True if l is map.road_network.graph[_from][_to][-1] or decoration else False
                        LaneGraphics.display(l, surface, two_side, use_line_color=True)
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
        film_size=(1000, 1000),
        screen_size=(1000, 1000),
        light_background=True,
        num_stack=15,
        history_smooth=0,
        road_color=(80, 80, 80),
        show_agent_name=False,
        camera_position=None,
        track_target_vehicle=False,
        **kwargs
        # current_track_vehicle=None
    ):
        # Setup some useful flags
        self.position = camera_position
        self.track_target_vehicle = track_target_vehicle
        self.show_agent_name = show_agent_name
        if self.show_agent_name:
            pygame.init()
        self.engine = get_engine()
        # self._screen_size = screen_size
        self.pygame_font = None
        self.map = self.engine.current_map
        self.stack_frames = deque(maxlen=num_stack)
        self.history_objects = deque(maxlen=num_stack)
        self.history_smooth = history_smooth
        # self.current_track_vehicle = current_track_vehicle
        if self.track_target_vehicle:
            assert self.current_track_vehicle is not None, "Specify which vehicle to track"
        self.road_color = road_color
        self._light_background = light_background
        self._text_render_pos = [50, 50]
        self._font_size = 25
        self._text_render_interval = 20

        # Setup the canvas
        # (1) background is the underlying layer. It is fixed and will never change unless the map changes.
        self._background_canvas = draw_top_down_map(
            self.map, simple_draw=False, return_surface=True, film_size=film_size, road_color=road_color
        )
        if self._light_background:
            pixels = pygame.surfarray.pixels2d(self._background_canvas)
            pixels ^= 2**32 - 1
            del pixels
        # (2) runtime is a copy of the background so you can draw movable things on it. It is super large
        # and our vehicles can draw on this large canvas.
        self._runtime_canvas = self._background_canvas.copy()
        # self._runtime_output = self._background_canvas.copy()  # TODO(pzh) what is this?

        # Setup some runtime variables
        self._render_size = screen_size
        self._background_size = tuple(self._background_canvas.get_size())
        # screen_size = self._screen_size or self._render_size
        # self._blit_size = (int(screen_size[0] * self._zoomin), int(screen_size[1] * self._zoomin))
        # self._blit_rect = (
        #     -(self._blit_size[0] - screen_size[0]) / 2, -(self._blit_size[1] - screen_size[1]) / 2, screen_size[0],
        #     screen_size[1]
        # )

        # screen and canvas are a regional surface where only part of the super large background will draw.
        # (3) screen is the popup window and canvas is a wrapper to screen but with more features
        self._render_canvas = pygame.display.set_mode(self._render_size)
        self._render_canvas.set_alpha(None)
        self._render_canvas.fill(color_white)

        # self.canvas = self._render_canvas
        # self.canvas = pygame.Surface(self._render_canvas.get_size())

        # Draw
        self.blit()
        self.kwargs = kwargs

        # key accept
        self.need_reset = False

    @property
    def canvas(self):
        return self._render_canvas

    def refresh(self):
        self._runtime_canvas.blit(self._background_canvas, (0, 0))
        self.canvas.fill(color_white)

    def render(self, text, *args, **kwargs):
        self.need_reset = False
        key_press = pygame.key.get_pressed()
        if key_press[pygame.K_r]:
            self.need_reset = True

        # Record current target vehicle
        objects = self.engine.get_objects(lambda obj: not is_map_related_instance(obj))
        this_frame_objects = self._append_frame_objects(objects)
        self.history_objects.append(this_frame_objects)

        self._handle_event()
        self.refresh()
        self._draw(*args, **kwargs)
        self._add_text(text)
        self.blit()
        ret = self.canvas.copy()
        ret = ret.convert(24)
        return ret

    def _add_text(self, text: dict):
        if not text:
            return
        if not pygame.get_init():
            pygame.init()
        font2 = pygame.font.SysFont('didot.ttc', 25)
        # pygame do not support multiline text render
        count = 0
        for key, value in text.items():
            line = str(key) + ":" + str(value)
            img2 = font2.render(line, True, (0, 0, 0))
            this_line_pos = copy.copy(self._text_render_pos)
            this_line_pos[-1] += count * self._text_render_interval
            self._render_canvas.blit(img2, this_line_pos)
            count += 1

    def blit(self):
        # self._render_canvas.blit(self._runtime_canvas, (0, 0))
        pygame.display.update()

    def close(self):
        pygame.quit()

    def reset(self, map):
        # Reset the super large background
        self._background_canvas = draw_top_down_map(
            map, simple_draw=False, return_surface=True, film_size=self._background_size, road_color=self.road_color
        )
        self._light_background = self._light_background
        if self._light_background:
            pixels = pygame.surfarray.pixels2d(self._background_canvas)
            pixels ^= 2**32 - 1
            del pixels

        # Reset several useful variables.
        # self._render_size = self._background_canvas.get_size()
        # Maybe we can optimize here! We don't need to copy but just blit new background on it.

        self._runtime_canvas = self._background_canvas.copy()
        # self._runtime_output = self._background_canvas.copy()
        self._background_size = tuple(self._background_canvas.get_size())

    @property
    def current_track_vehicle(self):
        return self.engine.current_track_vehicle

    def _append_frame_objects(self, objects):
        frame_objects = []
        for name, obj in objects.items():
            frame_objects.append(
                history_object(
                    name=name,
                    heading_theta=obj.heading_theta,
                    WIDTH=obj.top_down_width,
                    LENGTH=obj.top_down_length,
                    position=obj.position,
                    color=obj.top_down_color,
                    done=False
                )
            )
        return frame_objects

    def _draw(self, *args, **kwargs):
        """
        This is the core function to process the
        """
        if len(self.history_objects) == 0:
            return
        for i, objects in enumerate(self.history_objects):
            i = len(self.history_objects) - i
            if self.history_smooth != 0 and (i % self.history_smooth != 0):
                continue
            for v in objects:
                c = v.color
                h = v.heading_theta
                h = h if abs(h) > 2 * np.pi / 180 else 0
                x = abs(int(i))
                alpha_f = x / len(self.history_objects)
                VehicleGraphics.display(
                    vehicle=v,
                    surface=self._runtime_canvas,
                    heading=h,
                    color=(c[0] + alpha_f * (255 - c[0]), c[1] + alpha_f * (255 - c[1]), c[2] + alpha_f * (255 - c[2])),
                    draw_countour=False
                )

        # Use this line if you wish to draw "future" trajectory.
        # i is the index of vehicle that we will render a black box for it.
        # i = int(len(self.history_vehicles) / 2)
        i = -1
        for v in self.history_objects[i]:
            h = v.heading_theta
            c = v.color
            h = h if abs(h) > 2 * np.pi / 180 else 0
            alpha_f = 0
            VehicleGraphics.display(
                vehicle=v,
                surface=self._runtime_canvas,
                heading=h,
                color=(c[0] + alpha_f * (255 - c[0]), c[1] + alpha_f * (255 - c[1]), c[2] + alpha_f * (255 - c[2])),
                draw_countour=True,
                contour_width=2
            )

        if not hasattr(self, "_deads"):
            self._deads = []

        for v in self._deads:
            pygame.draw.circle(
                surface=self._runtime_canvas,
                color=(255, 0, 0),
                center=self._runtime_canvas.pos2pix(v.position[0], v.position[1]),
                radius=5
            )

        for v in self.history_objects[i]:
            if v.done:
                pygame.draw.circle(
                    surface=self._runtime_canvas,
                    color=(255, 0, 0),
                    center=self._runtime_canvas.pos2pix(v.position[0], v.position[1]),
                    radius=5
                )
                self._deads.append(v)

        v = self.current_track_vehicle
        canvas = self._runtime_canvas
        field = self._render_canvas.get_size()
        if self.position is not None or self.track_target_vehicle:
            cam_pos = v.position if self.track_target_vehicle else self.position
            position = self._runtime_canvas.pos2pix(*cam_pos)
        else:
            position = (field[0] / 2, field[1] / 2)
        off = (position[0] - field[0] / 2, position[1] - field[1] / 2)
        self.canvas.blit(source=canvas, dest=(0, 0), area=(off[0], off[1], field[0], field[1]))

        if "traffic_light_msg" in kwargs:
            if kwargs["traffic_light_msg"] < 0.5:
                traffic_light_color = (0, 255, 0)
            else:
                traffic_light_color = (255, 0, 0)
            pygame.draw.circle(
                surface=self.canvas,
                color=traffic_light_color,
                center=(self.canvas.get_size()[0] * 0.1, self.canvas.get_size()[1] * 0.1),
                radius=20
            )

        if self.show_agent_name:
            raise ValueError()
            # FIXME check this later
            if self.pygame_font is None:
                self.pygame_font = pygame.font.SysFont("Arial.ttf", 30)
            agents = [agent.name for agent in list(self.engine.agents.values())]
            for v in self.history_objects[i]:
                if v.name in agents:
                    position = self._runtime_canvas.pos2pix(*v.position)
                    new_position = (position[0] - off[0], position[1] - off[1])
                    img = self.pygame_font.render(
                        text=self.engine.object_to_agent(v.name),
                        antialias=True,
                        color=(0, 0, 0, 128),
                    )
                    # img.set_alpha(None)
                    self.canvas.blit(
                        source=img,
                        dest=(new_position[0] - img.get_width() / 2, new_position[1] - img.get_height() / 2),
                        # special_flags=pygame.BLEND_RGBA_MULT
                    )

    def _handle_event(self) -> None:
        """
        Handle pygame events for moving and zooming in the displayed area.
        """
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    import sys
                    sys.exit()
